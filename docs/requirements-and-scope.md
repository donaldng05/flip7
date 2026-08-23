# Requirements & Scope Specification

## 1. Purpose

This document defines the functional, technical, and experimental requirements for the Flip 7 RL Agent project.

The system must provide a reliable Flip 7 simulation environment, support competitive agent training, and provide reproducible evaluation of learned strategies.

---

## 2. Project Scope

### In Scope

* Complete Flip 7 game simulation.
* Multi-player games.
* Card and deck management.
* Round and game progression.
* Scoring and winning conditions.
* Action cards and targeting.
* Structured RL observations and actions.
* Heuristic and algorithmic baseline agents.
* Reinforcement learning agents.
* Self-play and opponent populations.
* Experiment tracking.
* Performance benchmarking.
* Reproducible training and evaluation.
* Optional model serving and human gameplay interface.

### Out of Scope — Initial Version

* Physical card recognition.
* Computer vision.
* Automated interaction with a physical game.
* Mobile application.
* Large-scale cloud deployment.
* Guaranteed optimal play against arbitrary opponents.
* General-purpose game-playing AI beyond Flip 7.

These may be considered future extensions.

---

## 3. Functional Requirements

### 3.1 Game Engine

The engine must:

* Represent the complete game state.
* Initialize and manage the deck.
* Draw cards without replacement.
* Track visible and discarded cards.
* Detect duplicate-number busts.
* Handle special cards.
* Calculate round scores.
* Track cumulative player scores.
* Detect the Flip 7 condition.
* Detect the game-winning condition.
* Support configurable player counts.
* Produce deterministic results when provided with a fixed random seed.

---

### 3.2 Game State

The system must represent, where applicable:

* Current player.
* Active players.
* Current round.
* Player scores.
* Cards held by each player.
* Remaining deck composition.
* Discarded cards.
* Active modifiers.
* Special-card effects.
* Current turn/phase.
* Relevant game history.

The representation must support both complete internal game state and restricted player observations.

---

### 3.3 Agent Interface

Agents must be able to:

* Receive an observation of the current game state.
* Select a legal action.
* Handle action masking where required.
* Receive rewards/results.
* Participate in repeated games without manual intervention.

The interface should allow different agent implementations to use the same environment.

---

### 3.4 Action Space

The environment must support the actions required by the game, including:

* **Hit**
* **Stay**
* Appropriate special-card actions
* Target selection where required

The action interface must prevent agents from selecting invalid actions.

---

### 3.5 Observation Space

The system must support multiple observation configurations, including progressively richer information.

At minimum:

1. **Basic state**

   * Current player's cards
   * Current round score
   * Relevant game status

2. **Competitive state**

   * Opponent scores
   * Opponent visible cards
   * Game progress

3. **Deck-aware state**

   * Remaining card counts
   * Discard/history information
   * Derived probability features

4. **Historical/recurrent state**

   * Sequential game information suitable for recurrent policies

The exact numerical representation will be defined separately in the environment specification.

---

## 4. RL Requirements

The project must support:

* Reinforcement learning training.
* Self-play.
* Training against fixed opponents.
* Training against populations of opponents.
* Checkpointing and restoring policies.
* Independent evaluation of trained policies.
* Reproducible training runs.

The initial RL implementation should use **PyTorch** and a suitable policy-gradient or value-based algorithm, with PPO as the initial candidate.

---

## 5. Baseline Requirements

Before evaluating RL performance, the project must implement baseline agents representing different levels of strategy:

* Random.
* Fixed-threshold.
* Bust-probability/risk-based.
* Expected-value based.
* Monte Carlo or dynamic-programming based where practical.

Baseline agents must be evaluated using the same environment and evaluation framework as RL agents.

---

## 6. Multi-Agent Requirements

The environment must support competitive multi-player interaction.

It must allow agents to:

* Observe the information available to them.
* Act sequentially.
* Respond to other players' actions.
* Make targeting decisions where applicable.
* Maintain independent policies.
* Participate in self-play.

The architecture should support future experimentation with MARL algorithms such as MAPPO.

---

## 7. Training Requirements

The training system should support:

* Configurable environments.
* Configurable opponents.
* Parallel game simulation.
* Training checkpoints.
* Evaluation checkpoints.
* Hyperparameter configuration.
* Experiment logging.
* Random seed control.
* Resumable training.

Training infrastructure should initially prioritize correctness and reproducibility over distributed scale.

---

## 8. Evaluation Requirements

The evaluation system must measure:

### Primary Metric

* **Game win rate**

### Secondary Metrics

* Average final score.
* Round score.
* Bust rate.
* Flip 7 frequency.
* Hit/Stay frequency.
* Special-card usage.
* Performance by player count.
* Performance against opponent types.
* Training efficiency.

Evaluation must distinguish between:

* Training opponents.
* Known evaluation opponents.
* Previously unseen opponents.

---

## 9. Experimentation Requirements

The project should support controlled experiments investigating:

* Observation representation.
* Card-counting information.
* Opponent information.
* Reward design.
* RL algorithms.
* Self-play strategies.
* Opponent populations.
* Player count.
* Hyperparameters.

Experiments should be reproducible and record sufficient configuration and results to allow comparison between runs.

---

## 10. Performance Requirements

The simulator should be optimized for **high-throughput game generation**, since simulation volume is expected to be more important than neural-network computation.

The system should eventually support:

* Parallel environments.
* Batched simulation.
* Efficient state representation.
* Efficient random card sampling.
* Optional JIT/compiled optimization if profiling demonstrates a bottleneck.

GPU acceleration is optional and should primarily be used for neural-network training rather than game simulation.

---

## 11. Software Engineering Requirements

The project must maintain:

* A reproducible Python development environment with a locked dependency set.
* Modular architecture.
* Unit tests for game rules.
* Integration tests for complete games.
* Deterministic seeded tests.
* Clear separation between environment, agents, training, and evaluation.
* Configuration-driven experiments.
* Version-controlled source code.
* Reproducible development/training environments.

Continuous integration should eventually run automated tests on changes to the project.
The engineering foundation establishes CI before game-rule implementation and
initial CD is limited to validated package artifacts.

---

## 12. Optional Deployment Requirements

Deployment is not required for the core research project.

If implemented, the system may provide:

* Exported trained models.
* ONNX inference.
* FastAPI inference service.
* Human-vs-agent interface.
* Dockerized deployment.
* Decision visualization.

Deployment should not interfere with the primary research and evaluation workflow.

---

## 13. Non-Functional Requirements

The system should be:

| Requirement      | Description                                                                           |
| ---------------- | ------------------------------------------------------------------------------------- |
| **Correct**      | Game mechanics must accurately follow the authoritative rules specification.          |
| **Reproducible** | Experiments should be repeatable using recorded configurations and seeds.             |
| **Modular**      | Agents, environments, algorithms, and evaluation should be independently replaceable. |
| **Scalable**     | Simulation and training should support increasing workloads when required.            |
| **Testable**     | Core mechanics and environment transitions must have automated tests.                 |
| **Extensible**   | New agents, algorithms, observations, and game variants should be easy to add.        |
| **Observable**   | Training and evaluation should produce sufficient metrics for analysis.               |

---

## 14. Scope Boundaries

The project is fundamentally concerned with:

> **Learning and evaluating strategic decision-making in Flip 7.**

It is not primarily concerned with:

* Building a commercial Flip 7 product.
* Recreating the publisher's digital game.
* Computer vision.
* Physical robotics.
* Massive-scale GPU training.
* Guaranteed theoretical optimality for the complete game.

The system should remain focused on using Flip 7 as a practical environment for **reinforcement learning, multi-agent learning, stochastic optimization, and decision-making under uncertainty**.

---

## 15. Definition of Done

The core project is complete when:

1. The Flip 7 rules are formally specified.
2. The game engine passes its rule and integration tests.
3. A standardized RL environment is operational.
4. Baseline agents are implemented and benchmarked.
5. At least one RL agent can be trained reproducibly.
6. RL performance can be compared against the baselines.
7. Self-play or population-based training can be evaluated.
8. The strongest agent demonstrates measurable competitive performance.
9. Experiments and results are documented.
10. The complete training/evaluation pipeline can be reproduced from the repository.
