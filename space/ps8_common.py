"""Shared helpers for every notebook: paths, config, task format, parser, SGLang lifecycle + client."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("PS8_ROOT", "/content/drive/MyDrive/ps8"))
REPO = PROJECT_ROOT / "repo"                 # code + small outputs (data/, results/, figures/)
ART = PROJECT_ROOT / "artifacts"             # big outputs (curator runs, adapters, merged models)
DATA, RESULTS, FIGURES = REPO / "data", REPO / "results", REPO / "figures"
LOGS = Path("/content/logs")
SGLANG_PY = "/content/venvs/sglang/bin/python"


SGLANG_PKGS = ["sglang[all]==0.5.22", "ninja"]
PIPELINE_PKGS = [
    "curatorkit[generation,hf] @ git+https://github.com/Lexsi-Labs/CuratorKIT.git@dd9e835301fc",
    "auditkit[openai,requests] @ git+https://github.com/Lexsi-Labs/AuditKIT.git@71316da34578",
    "aligntune==0.1.28",
]


def _sh(cmd: str) -> None:
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def ensure_env() -> None:
    """Idempotent install: SGLang in its own venv, Lexsi toolkits in the kernel env."""
    if not Path(SGLANG_PY).exists():
        _sh("uv venv -q --python 3.12 /content/venvs/sglang")
        _sh(f"uv pip install -q --python {SGLANG_PY} " + " ".join(f"'{p}'" for p in SGLANG_PKGS))
    try:
        import aligntune, auditkit, curatorkit  # noqa: F401
    except ImportError:
        _sh("uv pip install -q --system " + " ".join(f"'{p}'" for p in PIPELINE_PKGS))


def load_config(path: Path | None = None) -> dict:
    cfg = yaml.safe_load(open(path or REPO / "configs" / "experiment.yaml"))
    for d in (DATA, RESULTS, FIGURES, ART, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def gpu_profile(cfg: dict) -> tuple[str, dict]:
    name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    key = next((k for k in cfg["profiles"] if k.lower() in name.lower()), "t4")
    return key, cfg["profiles"][key]


# ── Unified task format ──────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Read the abstract and answer the research question "
    "using only the abstract. Give 1–3 sentences of reasoning, then finish with a final line exactly "
    "in the form `Answer: yes`, `Answer: no`, or `Answer: maybe`."
)
LABELS = ("yes", "no", "maybe")
_ANSWER_RE = re.compile(r"(?i)answer:\s*(yes|no|maybe)")


def user_message(context_text: str, question: str) -> str:
    return f"Abstract:\n{context_text}\n\nQuestion: {question}"


def build_messages(context_text: str, question: str, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(context_text, question)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def parse_answer(text: str | None) -> str | None:
    """Last `Answer: yes|no|maybe` in the text, lower-cased; None = format failure."""
    m = _ANSWER_RE.findall(text or "")
    return m[-1].lower() if m else None


def join_contexts(context: dict) -> str:
    return "\n".join(context["contexts"])


# ── CuratorKIT prompt templates (str.format: {context}/{num_questions} are filled by CuratorKIT) ──
def qa_prompt_template(fewshot: list[dict]) -> str:
    """fewshot rows: {question, long_answer, final_decision} from DEV (never TEST)."""
    shots = "\n\n".join(f"Question: {r['question']}\nAnswer field: {r['long_answer']}\\nAnswer: {r['final_decision']}"
                        for r in fewshot).replace("{", "{{").replace("}", "}}")
    return (
        "You are creating training data for a biomedical research question-answering model.\n"
        "Read the PubMed abstract below and write exactly {num_questions} NEW research questions that can be "
        "answered yes, no, or maybe using ONLY this abstract.\n\n"
        "Rules:\n"
        "- Each question is a standalone yes/no research question in the style of the examples.\n"
        "- Use a MIX of labels across your questions. Include questions whose correct answer is 'no' "
        "(e.g. an effect the study did not find, or a claim the results contradict), and use 'maybe' when "
        "the abstract's evidence is inconclusive, mixed, or too limited to decide.\n"
        "- The \"answer\" field is 1-3 sentences of reasoning grounded in the abstract's findings, then a final "
        "line exactly `Answer: yes`, `Answer: no`, or `Answer: maybe`.\n\n"
        f"Examples of the expected style (from other abstracts):\n{shots}\n\n"
        "Abstract:\n---\n{context}\n---\n\n"
        "Respond with a JSON array ONLY, no other text:\n"
        '[{{"question": "...", "answer": "<reasoning>\\nAnswer: <yes|no|maybe>"}}]'
    )


HALLUCINATION_PROMPT = """You are an expert biomedical fact-checker. Judge whether the answer to a yes/no/maybe research question is fully supported by the source abstract.

Source abstract:
---
{source_text}
---

Question: {question}
Answer: {answer}

Check:
1. Every factual claim in the reasoning (numbers, outcomes, populations, comparisons) is stated in the abstract.
2. The final label (yes/no/maybe) is the one the abstract's evidence supports. A confident yes/no when the evidence is inconclusive, or the opposite label, is a serious error.
3. The answer does not rely on outside medical knowledge that the abstract does not state.

Respond in JSON ONLY:
{{"grounding_score": 0.XX, "label_supported": true, "unsupported_claims": ["..."]}}

Score guide: 1.0 = all claims supported and correct label; 0.7-0.9 = minor paraphrase issues, correct label; 0.4-0.6 = some unsupported claims or a debatable label; 0.0-0.3 = wrong label or claims not in / contradicting the abstract."""


# ── IO ───────────────────────────────────────────────────────────────────────
def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# ── SGLang server lifecycle ──────────────────────────────────────────────────
_SERVER: subprocess.Popen | None = None


def _get(url: str, timeout: float = 3) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def start_server(model_path: str, served_name: str, extra_args: list[str] | None = None,
                 port: int = 30000, mem_fraction: float = 0.8, timeout: int = 900) -> str:
    """Launch SGLang in its own venv as a background process group; block until /health is ok."""
    global _SERVER
    stop_server(port)
    log_path = LOGS / f"sglang_{served_name}.log"
    cmd = [SGLANG_PY, "-m", "sglang.launch_server", "--model-path", str(model_path),
           "--host", "0.0.0.0", "--port", str(port), "--mem-fraction-static", str(mem_fraction),
           "--served-model-name", served_name, *(extra_args or [])]
    # venv bin on PATH: flashinfer/CUDA-graph JIT needs the venv's `ninja`.
    env = dict(os.environ, PATH=f"{Path(SGLANG_PY).parent}:{os.environ['PATH']}")
    _SERVER = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                               start_new_session=True, env=env)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _SERVER.poll() is not None:
            break
        if _get(f"http://localhost:{port}/health"):
            print(f"[sglang] {served_name} ready on :{port} in {time.time() - t0:.0f}s")
            return f"http://localhost:{port}/v1"
        time.sleep(5)
    tail = "".join(open(log_path).readlines()[-40:])
    stop_server(port)
    raise RuntimeError(f"SGLang failed to start (rc={_SERVER.poll() if _SERVER else None}). Log tail:\n{tail}")


def stop_server(port: int = 30000) -> None:
    """Kill our server's process group (and anything else holding the port), wait for GPU memory to free."""
    global _SERVER
    if _SERVER is not None and _SERVER.poll() is None:
        os.killpg(_SERVER.pid, signal.SIGTERM)
        try:
            _SERVER.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(_SERVER.pid, signal.SIGKILL)
    _SERVER = None
    # Servers started by an earlier kernel are not tracked in _SERVER.
    pattern = f"sglang.launch_server.*--port {port}"
    subprocess.run(["pkill", "-f", pattern])
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode != 0:
            time.sleep(3)  # let the driver release the process's memory
            return
        time.sleep(2)
    subprocess.run(["pkill", "-9", "-f", pattern])
    print(f"[sglang] warning: had to SIGKILL the server on :{port}")


# ── SGLang client ────────────────────────────────────────────────────────────
def chat(messages: list[dict], model: str, base_url: str = "http://localhost:30000/v1",
         temperature: float = 0.0, max_tokens: int = 256, timeout: float = 300) -> dict:
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return {"text": out["choices"][0]["message"]["content"], "latency_s": time.time() - t0,
            "completion_tokens": out["usage"]["completion_tokens"]}


def chat_many(batch: list[list[dict]], model: str, concurrency: int = 32, **kw) -> list[dict]:
    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(lambda m: chat(m, model, **kw), batch))


def sglang_callable(model: str, **kw):
    """AuditKIT `model=` callable: list of user messages -> list of completions (system prompt added here)."""
    def _call(prompts: list[str]) -> list[str]:
        return [r["text"] for r in chat_many(
            [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}] for p in prompts],
            model, **kw)]
    return _call


def auditkit_scorers() -> list:
    """Per-sample AuditKIT scorers: label accuracy (exact match of the parsed label) and format validity."""
    from auditkit.scorers import scorer
    from auditkit.types import Direction

    @scorer(direction=Direction.MAXIMIZE, name="label_accuracy")
    def label_accuracy(sample, output):
        return float(parse_answer(output) == sample.target)

    @scorer(direction=Direction.MAXIMIZE, name="format_ok")
    def format_ok(sample, output):
        return float(parse_answer(output) is not None)

    return [label_accuracy, format_ok]


def label_metrics(gold: list[str], pred: list[str | None]) -> dict:
    """Accuracy, macro-F1 (over yes/no/maybe), per-class F1, confusion matrix, format-failure rate."""
    from sklearn.metrics import confusion_matrix, f1_score
    p = [x if x is not None else "invalid" for x in pred]
    per = f1_score(gold, p, labels=list(LABELS), average=None, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": sum(g == x for g, x in zip(gold, p)) / len(gold),
        "macro_f1": float(per.mean()),
        "f1_per_class": dict(zip(LABELS, map(float, per))),
        "format_failure_rate": sum(x is None for x in pred) / len(pred),
        "confusion": {"labels": [*LABELS, "invalid"],
                      "matrix": confusion_matrix(gold, p, labels=[*LABELS, "invalid"]).tolist()},
    }


def bootstrap_ci(gold: list[str], pred: list[str | None], n: int = 1000, seed: int = 42) -> dict:
    import random
    rng, idx = random.Random(seed), range(len(gold))
    acc, mf1 = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        m = label_metrics([gold[i] for i in s], [pred[i] for i in s])
        acc.append(m["accuracy"]); mf1.append(m["macro_f1"])
    q = lambda v: [sorted(v)[int(0.025 * n)], sorted(v)[int(0.975 * n) - 1]]
    return {"accuracy_ci95": q(acc), "macro_f1_ci95": q(mf1)}


def evaluate_on_test(model_id: str, served_name: str, cfg: dict) -> dict:
    """Run TEST through the SGLang-served model with AuditKIT; save RunResult summary + per-sample predictions."""
    import auditkit as ak
    from auditkit.metrics.rag import LexicalGroundedness
    from auditkit.runspec import RunConfig
    test = read_jsonl(ART / "data" / "test.jsonl")
    samples = [ak.Sample(input=user_message(r["context_text"], r["question"]), target=r["final_decision"],
                         id=str(r["pubid"]), retrieval_context=[r["context_text"]]) for r in test]
    E = cfg["eval"]
    t0 = time.time()
    run = ak.evaluate(samples, model=sglang_callable(served_name, max_tokens=E["max_tokens"],
                                                     temperature=E["temperature"], concurrency=E["concurrency"]),
                      scorers=[*auditkit_scorers(), LexicalGroundedness()],
                      config=RunConfig(temperature=E["temperature"], max_tokens=E["max_tokens"], seed=cfg["seed"]),
                      experiment_name="ps8_pubmedqa", tags=[model_id])
    wall = time.time() - t0
    by_id = {p.sample_id: p for p in run.predictions}
    preds = []
    for r in test:
        p = by_id[str(r["pubid"])]
        preds.append({"pubid": r["pubid"], "gold": r["final_decision"], "output": p.raw_output,
                      "pred": parse_answer(p.raw_output),
                      "scores": {s["name"]: s["value"] for s in p.metadata.get("scores", [])}})
    gold, pred = [x["gold"] for x in preds], [x["pred"] for x in preds]
    res = {"model": model_id, **label_metrics(gold, pred), **bootstrap_ci(gold, pred, E["bootstrap_resamples"], cfg["seed"]),
           "lexical_groundedness": sum(x["scores"].get("lexical_groundedness", 0) for x in preds) / len(preds),
           "auditkit_headline": run.headline, "auditkit_run_id": run.run_id, "auditkit_fingerprint": run.fingerprint,
           "auditkit_errors": len(run.errors), "eval_wall_s": wall}
    write_jsonl(preds, RESULTS / f"predictions_{model_id}.jsonl")
    json.dump(res, open(RESULTS / f"eval_{model_id}.json", "w"), indent=1, default=str)
    (ART / "auditkit").mkdir(exist_ok=True)
    run.save(str(ART / "auditkit" / f"run_{model_id}.json"))
    print(f"[{model_id}] acc={res['accuracy']:.3f} {res['accuracy_ci95']} macroF1={res['macro_f1']:.3f} "
          f"fmt_fail={res['format_failure_rate']:.3f} wall={wall:.0f}s")
    return res


if __name__ == "__main__":
    assert parse_answer("reasoning...\nAnswer: Yes") == "yes"
    assert parse_answer("Answer: no ... later Answer: maybe.") == "maybe"
    assert parse_answer("I think yes") is None
    assert build_messages("ctx", "q?", "r\nAnswer: no")[2]["role"] == "assistant"
    p = qa_prompt_template([{"question": "Q {x}?", "long_answer": "L", "final_decision": "no"}])
    p = p.format(context="ABSTRACT", num_questions=3, difficulty="medium")
    assert "ABSTRACT" in p and "Q {x}?" in p and '[{"question"' in p
    HALLUCINATION_PROMPT.format(source_text="s", question="q", answer="a")
    print("ps8_common self-check ok")
