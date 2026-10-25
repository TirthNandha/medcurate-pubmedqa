"""MedCurate demo Space: two SGLang servers (D curated, B raw) behind a Gradio UI. Same prompt/parser as the experiments."""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

import gradio as gr

import ps8_common as C

MODELS = {"D": ("TirthNandha/medcurate-qwen2.5-1.5b-pubmedqa-curated", 30000),
          "B": ("TirthNandha/medcurate-qwen2.5-1.5b-pubmedqa-raw", 30001)}
# T4 (sm75) flags, identical to the Colab experiments (DECISIONS E6).
T4_ARGS = ["--dtype", "float16", "--attention-backend", "triton", "--sampling-backend", "pytorch",
           "--disable-prefill-cuda-graph", "--mem-fraction-static", "0.4"]
status = {k: "starting" for k in MODELS}


def launch(name, repo, port):
    log = open(f"/tmp/sglang_{name}.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", "--model-path", repo, "--host", "127.0.0.1",
                          "--port", str(port), "--served-model-name", name, *T4_ARGS], stdout=log, stderr=subprocess.STDOUT)
    while p.poll() is None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            status[name] = "ready"
            return
        except Exception:
            time.sleep(5)
    status[name] = "failed: " + "".join(open(f"/tmp/sglang_{name}.log").readlines()[-5:])


def start_all():
    # Sequential: two concurrent model loads + CUDA-graph captures can race for the 0.4 memory fractions.
    for name, (repo, port) in MODELS.items():
        launch(name, repo, port)


threading.Thread(target=start_all, daemon=True).start()


def answer(abstract, question):
    if not abstract.strip() or not question.strip():
        raise gr.Error("Paste an abstract and a question.")
    msgs = C.build_messages(abstract.strip(), question.strip())
    out = []
    for name, (_, port) in MODELS.items():
        if status[name] != "ready":
            out += [f"(model {name} is {status[name]} — the Space loads both models after waking, ~3-5 min)", ""]
            continue
        r = C.chat(msgs, name, base_url=f"http://127.0.0.1:{port}/v1", max_tokens=256)
        out += [r["text"], f"**{(C.parse_answer(r['text']) or 'unparseable').upper()}** · {r['latency_s']:.2f}s"]
    return out


with gr.Blocks(title="MedCurate — PubMedQA") as demo:
    gr.Markdown(
        "## MedCurate · curated synthetic data vs raw synthetic data (PubMedQA)\n"
        "Qwen2.5-1.5B fine-tuned with **AlignTune** on synthetic yes/no/maybe QA generated and filtered by **CuratorKIT**, "
        "served by **SGLang**. **D** trained on the 388 curated samples; **B** on all 858 raw samples. "
        "On 500 expert-labelled PubMedQA test items: accuracy D 0.670 vs B 0.604. Answers only from the abstract — research demo, not medical advice.")
    state = gr.Markdown()
    abstract = gr.Textbox(label="Abstract", lines=10)
    question = gr.Textbox(label="Research question")
    go = gr.Button("Answer", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### D · trained on CURATED data")
            d_txt, d_lab = gr.Textbox(label="Reasoning", lines=5), gr.Markdown()
        with gr.Column():
            gr.Markdown("### B · trained on RAW generated data")
            b_txt, b_lab = gr.Textbox(label="Reasoning", lines=5), gr.Markdown()
    go.click(answer, [abstract, question], [d_txt, d_lab, b_txt, b_lab])
    gr.Examples(json.load(open("examples.json")), [abstract, question])
    demo.load(lambda: "Model status: " + ", ".join(f"{k} {v}" for k, v in status.items()), None, state)

demo.queue().launch(server_name="0.0.0.0", server_port=7860)
