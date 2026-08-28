# Project Plan

## 1. Project Goal

Build and evaluate a reinforcement learning agent capable of playing Flip 7 competitively against heuristic, algorithmic, and learned opponents.

The project will progress from a reliable game simulator to increasingly sophisticated agents, ending with robust self-play evaluation and optional deployment.

---

## 2. Development Phases

### Phase 0 — Engineering Foundation

**Goal:** Establish a reproducible, testable, and CI-validated repository before
implementing game behavior.

* Set up the Python package and locked development environment.
* Add formatting, linting, typing, testing, and pre-commit quality gates.
* Add cross-platform local checks and GitHub Actions CI.
* Add optional Docker support for reproducible development.
* Define configuration, experiment-artifact, architecture, and release policies.
* Automate versioned package artifact releases without premature production hosting.

**Deliverable:** A clean-clone development environment with passing local checks,
CI validation, and reproducible package builds.

### Phase 1 — Game Rules & Specification

**Goal:** Establish an unambiguous definition of Flip 7.

* Document complete game rules.
* Define round and game flow.
* Define cards, actions, scoring, and winning conditions.
* Resolve edge cases and rule ambiguities.
* Establish the authoritative game specification.

**Deliverable:** `game-rules-specification.md`

---

### Phase 2 — Game Engine

**Goal:** Build a correct and testable Flip 7 simulator.

* Implement deck and card mechanics.
* Implement rounds and game progression.
* Implement scoring.
* Implement action cards.
* Implement multi-player turn handling.
* Add deterministic testing and rule validation.

**Deliverable:** Functional Flip 7 simulation engine.

---

### Phase 3 — RL Environment

**Goal:** Expose the game as a standardized RL environment.

* Define observations.
* Define action spaces.
* Implement legal-action handling and action masking.
* Define rewards.
* Support multi-agent interaction.
* Validate environment transitions.

**Deliverable:** A PettingZoo AEC environment wrapping the core engine, plus a
Gymnasium vs-opponents wrapper. Layouts are defined in
`environment-specification.md` and ADRs `0001`–`0005`.

---

### Phase 4 — Baseline Agents

**Goal:** Establish meaningful performance benchmarks before using RL.

Implement:

* Random agent.
* Fixed-threshold heuristic.
* Risk/bust-probability heuristic.
* Expected-value agent.
* Monte Carlo or DP-based strategy where practical.

Evaluate baseline strength through repeated tournaments.

**Deliverable:** Baseline agent suite and benchmark results.

Phase 4 uses an observation-only callable policy interface. The initial suite
contains random-legal, fixed-threshold, bust-probability, expected-value, and
bounded round-level dynamic-programming agents. Evaluation runs through the
PettingZoo AEC environment with fixed seeds and reports win share, ties, final
score, round score, bust rate, Flip 7 frequency, and action distributions.

---

### Phase 5 — Initial RL Agent

**Goal:** Train the first learned policy.

* Implement PPO or an equivalent RL algorithm.
* Train against baseline opponents.
* Establish training and evaluation pipelines.
* Track learning curves and game performance.
* Compare RL performance against baseline agents.

**Deliverable:** First competitive RL policy.

Phase 5 uses a custom PyTorch PPO actor-critic with masked discrete actions,
the Gymnasium learner-seat wrapper, frozen Phase 4 baseline opponents,
reproducible checkpoints, and JSON training/evaluation artifacts. Self-play,
recurrent policies, and vectorized training remain deferred.
The initial bounded run is documented in
[`phase-5-initial-rl.md`](phase-5-initial-rl.md).

---

### Phase 6 — State & Information Experiments

**Goal:** Determine what information actually improves strategic performance.

Experiment with progressively richer observations:

```text
Current round state
        ↓
+ Opponent scores
        ↓
+ Visible/depleted deck information
        ↓
+ Historical information
        ↓
+ Recurrent / learned memory
```

Measure the effect of each representation on performance.

**Deliverable:** State-representation ablation study.

The Phase 6 implementation uses the six-condition matrix documented in
[`phase-6-state-info-experiments.md`](phase-6-state-info-experiments.md):
`basic`, `competitive`, and `deck_aware` observations with fixed or randomized
learner seating. Historical observations, recurrent policies, and reward
shaping remain deferred until this controlled comparison is complete.

---

### Phase 7 — Self-Play & Multi-Agent Learning

**Goal:** Move from learning against fixed opponents to competitive adaptation.

* Introduce self-play.
* Maintain a population of opponent policies.
* Train against diverse strategies.
* Evaluate policy strength using tournaments/Elo.
* Prevent overfitting to a single opponent strategy.
* Investigate MAPPO or other MARL approaches where appropriate.

**Deliverable:** Robust competitive RL agent.

Phase 7 begins with league PPO: a randomized-seat active policy trains against
frozen historical policy snapshots and diverse Phase 4 baselines. The initial
experiment compares a Phase 6 control, a mixed snapshot league, and a
latest-snapshot ablation using held-out opponent variants and seat-rotated
tournaments with Elo. MAPPO remains evidence-gated until centralized-critic
credit assignment is justified by the league results.

The Phase 7 follow-up strengthens that evidence boundary with fixed-bank
behavioural diversity metrics, novelty-aware retention, paired multi-game
tournaments, five-seed confirmation, and a targeted MAPPO pilot. Phase 8
should begin only after the population-diversity and stable-adaptation gates
are reported, with MAPPO adopted, deferred, or explicitly sent to another
focused investigation.

The stability follow-up adds deterministic seat-balanced rollouts, a gradual
opponent curriculum, training-only response signatures, paired held-out
comparisons, and focused final-versus-warmup tournaments. Its seat-aware
fallback and MAPPO pilot remain gated by the balanced-basic confirmation
results.

---

### Phase 8 — Robustness & Generalization

**Goal:** Determine whether the agent has learned a general strategy.

Evaluate against:

* Previously unseen heuristic agents.
* Previously unseen RL checkpoints.
* Different player counts.
* Different game states and deck conditions.
* Different opponent behavior distributions.

Analyze where the agent succeeds and fails.

**Deliverable:** Comprehensive evaluation report.

---

### Phase 9 — Optimization & Infrastructure

**Goal:** Scale experimentation where necessary.

* Optimize simulation performance.
* Add vectorized/parallel environments.
* Introduce Ray if distributed training provides value.
* Add hyperparameter optimization with Optuna.
* Improve experiment tracking.
* Containerize reproducible training environments.

**Deliverable:** Scalable training and experimentation infrastructure.

---

### Phase 10 — Deployment & Demonstration

**Goal:** Make the final agent usable outside the training environment.

Potential components:

* Export trained policy with ONNX.
* Build inference service with FastAPI.
* Create a simple human-vs-agent interface.
* Display agent decisions and relevant statistics.
* Containerize the application.

**Deliverable:** Playable Flip 7 RL agent.

---

## 3. Evaluation Strategy

Every major phase should produce measurable results.

Primary metric:

> **Game win rate**

Secondary metrics:

* Average final score
* Round win rate
* Bust rate
* Flip 7 frequency
* Action distribution
* Performance by player count
* Performance against different opponent classes
* Training efficiency
* Inference latency where relevant

The final evaluation should compare the strongest learned agent against all major baseline classes.

---

## 4. Development Principles

The project will follow four principles:

1. **Correctness before optimization**
   Validate the game engine before training agents.

2. **Baselines before RL**
   Establish what simpler methods can achieve before claiming an RL advantage.

3. **Measure before adding complexity**
   New algorithms or infrastructure should be justified by experiments.

4. **Separate training from evaluation**
   Final performance should be measured against opponents and scenarios not used during training.

---

## 5. High-Level Roadmap

```text
Rules Specification
        ↓
Game Engine
        ↓
RL Environment
        ↓
Baseline Agents
        ↓
Initial RL Agent
        ↓
State / Information Experiments
        ↓
Self-Play & MARL
        ↓
Robustness & Generalization
        ↓
Scaling & Optimization
        ↓
Deployment
```

The project should remain usable after each phase, allowing experimentation to begin early rather than waiting for the entire system to be completed.
