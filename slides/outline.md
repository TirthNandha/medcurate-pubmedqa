# MedCurate: 5-minute pitch (Syntax Error 2026 · Local LLM Track · PS #8)

## 1. The problem: expert labels are scarce (30 s)
- PubMedQA: **1,000** expert-labelled yes/no/maybe questions, **61,000** unlabelled abstracts.
- Medical labels cost doctors' time, so synthetic data from a local LLM is the obvious lever.
- **The real question:** is raw generated data good enough, or does *curating* it matter? And is any gain from quality or just from quantity?

## 2. The pipeline: three Lexsi toolkits, all inference on SGLang (45 s)
- Diagram (README mermaid): unlabelled abstracts → **CuratorKIT** (generate + gate) → **AlignTune** (LoRA SFT + merge) → **SGLang** serving → **AuditKIT** eval.
- One SGLang server at a time on a single **T4**: 7B-AWQ generator/judge, then each 1.5B student.
- Everything is reproducible: 9 idempotent notebooks, pinned versions, and every deviation logged in `DECISIONS.md`.

## 3. Data and the curation funnel (40 s)
- Figure: `figures/curation_funnel.png`. 300 abstracts → **858 QA** → **388 curated** (55% rejected).
  - RewardGate: 395
  - Grounding + label check: 67
  - Diversity: 8
- Thresholds were tuned on a 30-abstract pilot to land in the 30–60% band.
- TEST: 500 expert-labelled items, never seen by the generator or trainer (pubid-isolated, asserted).

## 4. Experimental design: the fairness control (30 s)
- Five models:
  - **A** base
  - **B** RAW (858)
  - **C** RAW-MATCHED (388, a random sample of raw)
  - **D** CURATED (388)
  - T teacher (reference)
- Identical hyperparameters, prompt, parser and decoding for every model.
- **C is the key control:** same size as D, so any D > C gain comes from quality, not quantity.

## 5. Results (60 s)
- Figure: `figures/results_bars.png`.
- **Accuracy:** D **0.670** vs C 0.594 / B 0.604 / A 0.592.
  - D vs C **+7.6 pts**, 95% CI [+3.6, +11.4], McNemar **p = 0.0003**.
  - D beats B too, despite B having **2.2× more data** (p = 0.003).
  - C ≈ B: the extra 470 raw samples added nothing.
- **The 1.5B student beats its 7B teacher on accuracy:** 0.670 vs 0.618, McNemar p = 0.036.
- Fine-tuning brings format compliance to 100% and lexical groundedness from 0.55 to about 0.79.
- **Honest caveat:** macro-F1 is a tie (D 0.493 vs C 0.506, CI includes 0). Curation removed most "maybe" answers, so D rarely says "maybe".

## 6. What curation caught (40 s)
- Raw data teaches **hedging**: B and C answer "maybe" on 108–120 TEST items whose experts said yes or no.
- Three caught samples:
  - A null result labelled "maybe" instead of "no"
  - Reasoning about the wrong subgroup
  - "94% abnormal" misread when the key result was NPV = 100%
- Honest note: the gate also rejected some good samples (the generator judging itself is lenient and noisy).

## 7. Demo and serving (30 s)
- Live Gradio app (`08_demo_gradio`): paste an abstract and question, and see the **D vs B** answers side by side, both served by SGLang.
- Serving D on a single T4:
  - **2.4 req/s, 307 output tok/s, 1,538 total tok/s** at 32 concurrent requests
  - p50 / p95 latency 13.1 / 13.3 s
  - inter-token latency 36 ms

## 8. Takeaways (25 s)
- **Curation > raw at equal size** (+7.6 accuracy points, p < 0.001). Quality, not quantity, and the 1.5B student overtakes its 7B teacher on accuracy.
- Curation can **shift the label distribution**, so check per-class metrics. The next step is label-aware thresholds or CuratorKIT resampling to keep "maybe".
- The whole loop runs locally on one T4: SGLang + CuratorKIT + AlignTune + AuditKIT.
- We also found and documented an uncentred p-value in AuditKIT's paired bootstrap.
