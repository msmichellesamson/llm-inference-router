# llm-inference-router

> Can a simple heuristic router approximate RouteLLM's learned router?

## The question I'm exploring

LMSYS's [RouteLLM](https://arxiv.org/abs/2406.18665) showed that a learned
router can send ~50% of queries to a cheaper/smaller model with negligible
quality loss on MT-Bench. [FrugalGPT](https://arxiv.org/abs/2305.05176)
made a similar argument earlier. The shared insight: **most queries don't
need your strongest model.**

But a learned router is its own deployment problem — you have to train it,
keep it fresh as model behaviour changes, and monitor it for drift. I wanted
to know how much of RouteLLM's win you can capture with a much simpler
**heuristic** complexity analyzer (token count, regex for code/reasoning
patterns) and aggressive load balancing.

## Why I care

Two reasons, both practical:

1. **Cost.** API spend on frontier models grows linearly with usage and
   doesn't slow down. Even a 30% reduction at scale matters.
2. **Sustainability.** Smaller models burn meaningfully less GPU-time per
   token. Routing is one of the few inference-time levers that improves
   both economics and emissions at the same time.

I'm specifically interested in the **failure modes** of cheap-first routing —
what categories of query get silently downgraded, and how would you catch it
in production?

## What's in here

A reasonably complete async router service:

- `src/core/complexity_analyzer.py` — heuristic scoring (token count,
  code patterns, reasoning keywords)
- `src/core/router.py` — model selection given complexity + availability
- `src/core/load_balancer.py`, `circuit_breaker.py`, `health_checker.py` —
  the production hygiene
- `src/observability/` — Prometheus exporter, OpenTelemetry tracing,
  structured logs with correlation IDs
- `src/api/` — FastAPI server with metrics + benchmark endpoints

Plus terraform for GKE, a Dockerfile, k8s manifests, and an OpenAPI spec.

## What I'm finding (so far)

I ran an honest first pass against the [MT-Bench](https://huggingface.co/datasets/philschmid/mt-bench)
prompts (n=80, 8 categories). Full writeup in
[`experiments/findings.md`](experiments/findings.md). The short version:

| Method | AUC (complex vs simple) |
|---|---|
| token count only | 0.322 |
| heuristic (token + regex) | **0.417** |
| random | 0.496 |
| MiniLM embeddings + logistic regression (LOO-CV) | **0.938** |

- **The current heuristic is worse than random.** The intuition that
  "longer prompts are harder" is wrong on MT-Bench — math and reasoning
  prompts tend to be *short* (`"What is the integral of x sin(x)?"`),
  and extraction prompts are *long* (paragraph + question).
- **A 22M-parameter learned baseline beats the heuristic by ~0.5 AUC.**
  That's a meaningful gap, not noise. It's not a SOTA result — it's
  the obvious baseline that the current heuristic should be replaced
  with, full stop.
- The regex `"explain|analyze|compare"` fires on extraction prompts
  constantly, which is why extraction has the highest heuristic
  complexity score (0.250) despite being one of the easier categories.
- I haven't yet run actual models against these prompts to measure
  the *quality* difference between routing decisions. That's the
  experiment that matters next — AUC on a proxy label isn't
  cost-savings-at-equal-quality.

## What I'd do next

- **Replace the heuristic with the learned classifier in
  `src/core/complexity_analyzer.py`.** Keep the heuristic as a fallback
  for when the embedding service is down. This is the smallest change
  with the largest expected impact based on what I just measured.
- Build the **shadow eval**: actually answer each prompt with a small
  model (Llama-3.2-1B-Instruct) and a strong one, judge with a third
  model, report cost-savings at fixed quality. AUC on a proxy label
  is not the number that actually matters for production routing.
- Run the same experiment on 500-1000 prompts from a more realistic
  distribution ([LMSYS Chatbot Arena conversations](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
  is the obvious starting point).
- Compare against [RouteLLM's released routers](https://github.com/lm-sys/RouteLLM)
  on the same prompts.
- Measure GPU-seconds-saved per query, not just dollar cost — that's
  the sustainability angle I actually care about.

## Status

The router service runs end-to-end. The complexity analyzer it currently
uses is **demonstrably worse than random** on MT-Bench (see findings),
so the next code change should be replacing it. Reproducible eval
harness exists at [`experiments/`](experiments/).

## References

- Ong et al., [*RouteLLM: Learning to Route LLMs with Preference Data*](https://arxiv.org/abs/2406.18665) (2024)
- Chen et al., [*FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*](https://arxiv.org/abs/2305.05176) (2023)
- Šakota et al., [*Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling*](https://arxiv.org/abs/2308.06077) (2024)
- [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench) — the eval suite RouteLLM trained against
- [Martian's model router](https://withmartian.com/) — commercial implementation in the same space
