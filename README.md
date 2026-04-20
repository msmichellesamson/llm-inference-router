# llm-inference-router

> Can a simple heuristic router approximate RouteLLM's learned router?

## The question I'm exploring

LMSYS's [RouteLLM](https://arxiv.org/abs/2406.18665) showed that a learned
router can send ~50% of queries to a cheaper/smaller model with negligible
quality loss on MT-Bench. FrugalGPT made a similar argument earlier. The
shared insight: **most queries don't need your strongest model.**

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

I've only run this against synthetic queries, not a real eval set, so take
this with appropriate skepticism:

- The regex-based "reasoning detector" misses anything that isn't phrased
  in obvious English (`"why does X?"` works; a math problem stated in LaTeX
  doesn't). The heuristic is a starting point, not a finished classifier.
- Token count alone is a surprisingly strong feature for a first pass —
  most short queries genuinely are simpler. The interesting failures are in
  the long-but-easy and short-but-hard tails.
- The complexity threshold is a single knob, which is wrong. Different
  query categories probably want different thresholds (e.g. code queries
  should bias toward the stronger model; small-talk should always go to
  the cheap one).
- I haven't built the part that actually matters: a **shadow eval** that
  compares router decisions to a "send everything to the strong model"
  baseline on the same traffic. Without that I can't claim quality parity.

## What I'd do next

- Build the shadow-eval pipeline. Without ground truth on routing decisions,
  the rest is theatre.
- Replace the regex reasoning detector with a small classifier (DistilBERT
  fine-tuned on MT-Bench prompts would be a fair comparison to RouteLLM)
- Add per-category thresholds and an A/B test framework
- Measure GPU-seconds-saved per query, not just dollar cost — because the
  sustainability angle is the one I actually care about

## Status

The router runs end-to-end. The eval harness that would tell me whether
the routing decisions are any good doesn't exist yet.

## References

- Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data* (2024)
- Chen et al., *FrugalGPT: How to Use Large Language Models While Reducing
  Cost and Improving Performance* (2023)
- Šakota et al., *Fly-Swat or Cannon? Cost-Effective Language Model Choice
  via Meta-Modeling* (2024)
