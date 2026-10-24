# DECISIONS

Every deviation from the project brief, plus workarounds, version pins and threshold choices, each with its reason. Entries are added as they happen.

## Environment

| # | Decision | Reason |
|---|---|---|
| E1 | **GPU = Tesla T4 (16 GB, sm75)** | `colab new` turned down L4, A100, H100 and G4 for this account (no entitlement). Only T4 would allocate. The T4 profile from §6 is used. |
| E2 | **Notebooks are run by the agent with the `colab` CLI** (`colab exec -f nb.ipynb`) on a Colab VM, not from the IDE extension | The user chose this. Each run saves an `*_output.ipynb` with outputs next to its notebook. These are git-ignored, and the notebooks stay runnable top to bottom by hand. |
| E3 | **Persistence = Google Drive** at `/content/drive/MyDrive/ps8/` (checked writable in notebook 00) | `colab drivemount` was run once by the user, because it needs an interactive login. Every notebook's setup cell calls `drive.mount` only if Drive isn't already mounted. |
| E4 | **Code reaches the runtime by syncing the repo to Drive** (`ps8/repo/`: `src/`, `configs/`, `notebooks/`) instead of `%%writefile` | There's one source of truth for `ps8_common.py`, with no copy pasted into a notebook. Notebooks add `ps8/repo/src` to `sys.path`. To reproduce, copy the repo to `MyDrive/ps8/repo` (or set `PS8_ROOT`). |
| E5 | **SGLang lives in its own venv** (`/content/venvs/sglang`, Python 3.12, `sglang[all]==0.5.22`, its own torch 2.14.1). The Lexsi toolkits go in the kernel env (torch 2.11). | This is the brief's isolation strategy. Installing `aligntune` also pulls `trl[vllm]==1.7.1` → vLLM 0.22.1, which can't share an env with SGLang. The toolkits only reach SGLang over HTTP. |
| E6 | **SGLang flags on T4:** `--dtype float16 --attention-backend triton --sampling-backend pytorch --disable-prefill-cuda-graph`, and the venv's `bin/` (for `ninja`) added to the server's PATH | 1) The default FlashInfer backend JIT-compiles kernels for sm75 and crashed with `FileNotFoundError: ninja`. The root cause was that the venv's `ninja` wasn't on PATH. 2) The T4 has no bf16. 3) Capturing the prefill CUDA graph took 141 s on every launch, and we launch about 8 times; turning it off brought startup from 201 s to 56 s. Decode CUDA graphs stay on. |
| E7 | **Generator on T4 = `Qwen/Qwen2.5-7B-Instruct-AWQ` with `--quantization awq`**, falling back to `Qwen2.5-3B-Instruct` | `awq_marlin` needs sm80+. The choice actually used is recorded in `data/generator_choice.json` by notebook 03. |

## Pinned versions (verified working together)

- SGLang venv: see `requirements-sglang.txt` (sglang 0.5.22, torch 2.14.1+cu130, transformers 5.19.0, flashinfer-python 0.7.1, ninja 1.13.2).
- Pipeline / kernel env: see `requirements-pipeline.txt` and `requirements-train.txt` (curatorkit 0.1.21 @ `dd9e835301fc`, auditkit 1.1.3 @ `71316da34578`, aligntune 0.1.28, trl 1.7.1, transformers 5.20.2, torch 2.11.0+cu130, peft 0.21.2, litellm 1.106.3).
- The brief mentions `transformers==5.14.1`. AlignTune 0.1.28 actually requires `transformers>=5.15,<6`, and Colab's preinstalled 5.20.2 satisfies that.

## Data curation (CuratorKIT)

| # | Decision | Reason |
|---|---|---|
| D1 | A **custom `qa_prompt_template`** asks for standalone yes/no/maybe research questions with a mix of labels, and answers in the exact training/eval format (`reasoning\nAnswer: x`). It includes 5 DEV few-shot examples (2 yes, 2 no, 1 maybe; conclusions ≤60 words). | Generated targets have to match the unified task format. Few-shots come only from DEV. |
| D2 | A **custom `hallucination_prompt_template`** keeps the required `grounding_score` key, and additionally asks the judge to check that the **final yes/no/maybe label** is the one the abstract supports | For this task the most harmful synthetic error is a wrong label, not only an unsupported claim. |
| D3 | **Dedup:** CuratorKIT applies `dedup="minhash"` to the *source* rows, before generation. For near-duplicate *generated* QA, the post-generation **DiversityGate** (`diversity_threshold=0.92`, all-MiniLM-L6-v2) is enabled | This follows CuratorKIT's pipeline order (normalizers → generator → Hallucination → Reward → Diversity). 0.92 is the value from CuratorKIT's docs. DiversityGate compares within one run (one 100-abstract chunk), and QA from different abstracts are rarely near-duplicates. |
| D4 | **Generation runs in 100-abstract chunks**, one CuratorKIT run each | This makes it resumable after a disconnect. It mirrors tutorial 09's chunking. |
| D5 | **RAW = passed + rejected QA** (CuratorKIT's `RejectedSample` keeps the full question, answer and source abstract), so no second, ungated generation run was needed | Verified in the CuratorKIT source: `RejectedSample(**sample.model_dump(), ...)`, and tutorial 09 trains on `result.rejected`. Source-level generation failures (unparseable generator output) contain no QA and are counted separately. |

## Fine-tuning (AlignTune)

| # | Decision | Reason |
|---|---|---|
| F1 | **Loss is on the full sequence** (system + abstract + question + answer), not assistant tokens only | AlignTune's Unsloth SFT backend formats `messages` into a single `text` field, so it can't mask the prompt. With the TRL backend, its DataManager flattens conversational `prompt`/`completion` into strings and rebuilds `messages`, and TRL's `assistant_only_loss` needs `{% generation %}` tags that Qwen2.5's chat template doesn't have. Both were tested in a probe on 2026-10-24. The brief allows this ("only if AlignTune supports it"). The setting is identical for B, C and D. |
| F2 | **Unsloth backend, LoRA on the fp16 base model (no 4-bit)** | The 1.5B student fits easily on a T4 in fp16. Merging LoRA into an fp16 base is lossless, while merging into a 4-bit base is not, so the merged model SGLang serves is exactly the trained one. |
| F3 | **LoRA dropout 0.05 kept** even though Unsloth warns that only dropout=0 gets its fast patching | The brief specifies 0.05. It's identical across arms and only costs speed. |
| F4 | **Merge with AlignTune `PEFTMerger.merge_lora(..., torch_dtype="float16")`**, then serve the merged folder with SGLang | Verified end to end in a probe: the merged smoke model was served and answered in the right format. |
| F5 | The TRL and Unsloth backends **can't be used in the same Python process** (once imported, Unsloth patches TRL's `SFTTrainer` globally → `RuntimeError: Unsloth: You must specify a formatting_func`) | Notebook 05 only uses Unsloth. Restart the kernel between experiments that mix them. |

## Generation and curation settings (pilot-driven)

| # | Decision | Reason |
|---|---|---|
| G1 | **N = 300 source abstracts** (the brief's T4 default is 600) | Pilot timing on the T4 with the 7B-AWQ generator and judge was 311 s per 30 abstracts (generation plus both LLM gates), so 600 abstracts would take ~1 h 50 m. The user chose N=300 to fit the hackathon deadline; the brief's time box allows this cut. A smaller (3B) generator/judge was considered and rejected. It would weaken both the teacher signal and the gate's ability to tell good from bad, which is the thing PS #8 measures. |
| G2 | **Thresholds: grounding ≥ 0.80, reward ≥ 0.75** (both started at 0.7) | Pilot on 30 abstracts → 86 QA (`data/pilot_runs.json`). At 0.7/0.7 only 19% were rejected (below the 30–60% target). The judge scores sit on a coarse grid. Simulated sweep over the pilot's recorded scores: 0.8/0.75 → 40% rejected (+2 diversity), 0.85/0.70 → 57%, any reward ≥0.80 → ≥74%. We picked 0.8/0.75, mid-band, with both judges contributing. |
| G3 | **The generator is also the judge** (one SGLang server, Qwen2.5-7B-Instruct-AWQ) | Only one 7B model fits on the T4 at a time. CuratorKIT recommends a separate judge to avoid self-leniency, and the pilot shows that leniency: at 0.7, grounding rejected only 3/86. This is a limitation, and the reward threshold compensates for it. |
| G4 | The sampling seed is passed to SGLang through `llm_extra_body={"seed": 42}` (generator temperature 0.7, judge 0.1) | For reproducibility. The output is not guaranteed bit-identical across batch compositions. |
