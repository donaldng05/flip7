---
name: flip7-agent-evaluation
description: Benchmark Flip 7 baseline, learned, self-play, and unseen-opponent agents with repeatable tournaments and transparent metrics. Use when comparing policies or producing evaluation reports.
disable-model-invocation: true
---

# Flip 7 Agent Evaluation

1. Verify the environment, action legality, scoring, and seeded transitions
   before running a tournament.
2. Define the agent roster, opponent provenance, player counts, game count,
   evaluation seed set, and whether opponents were seen during training.
3. Compare available policies against random, fixed-threshold, risk-based,
   expected-value, and algorithmic baselines.
4. Run repeated games with fixed evaluation seeds; never infer strength from a
   one-off match.
5. Calculate win rates with sample sizes and uncertainty, plus the documented
   secondary metrics.
6. Preserve machine-readable results and write a concise report that identifies
   training, known-evaluation, and unseen-opponent results separately.

Use [reference.md](reference.md) for the evaluation record and report format.
