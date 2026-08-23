# Project Overview

## 1. Project Identity

* **Project:** Flip 7 RL Agent
* **Domain:** Reinforcement Learning, Multi-Agent Systems, Game AI, Sequential Decision-Making

This project aims to develop a reinforcement learning agent capable of playing Flip 7 strategically and competing against human and artificial opponents.

The project uses Flip 7 as a controlled environment for studying decision-making under uncertainty, risk, incomplete information, and competition.

---

## 2. Vision

The long-term vision is to build an agent that can learn and execute strong, robust strategies for Flip 7 rather than relying on a fixed set of hand-crafted rules.

The agent should be capable of adapting its decisions based on:

* Current cards and scores
* Remaining deck composition
* Opponent states
* Game progress toward 200 points
* Risk and potential reward
* Opponent behavior and strategy

Ultimately, the goal is to create an agent that can consistently outperform strong baseline and previously unseen strategies.

---

## 3. Problem Statement

At first glance, Flip 7 appears to be a simple push-your-luck game. However, optimal play requires sequential decisions under uncertainty.

At every decision point, an agent must determine whether to:

* Continue drawing and accept additional risk
* Stop and preserve its current score
* Use an action card strategically
* Select an appropriate opponent when targeting is possible

The optimal decision depends not only on the current hand, but also on the remaining deck, opponents' scores, and the overall race toward the winning condition.

The project therefore treats Flip 7 as a sequential decision-making and competitive multi-agent learning problem.

---

## 4. Project Objectives

### Primary Objectives

* Build an accurate and reproducible Flip 7 game simulator.
* Develop a reinforcement learning environment for the game.
* Implement strong heuristic and mathematical baseline agents.
* Train RL agents through self-play and controlled opponent populations.
* Evaluate whether learned policies can outperform established strategies.
* Investigate how information such as deck state and game history affects performance.
* Evaluate the robustness of learned strategies against previously unseen opponents.

### Secondary Objectives

* Develop practical experience with RL and MARL.
* Explore partial observability and state representation.
* Develop scalable simulation and training infrastructure.
* Establish a reusable framework for experimentation with competitive game agents.

---

## 5. Core Research & Engineering Questions

The project will investigate questions such as:

* Can an RL agent learn a strong Flip 7 strategy without explicitly encoding a complete strategy?
* How much does knowledge of the remaining deck improve decision-making?
* How does explicit card counting compare with learned memory-based representations?
* How should an agent balance short-term score against the probability of winning the overall game?
* Can self-play produce stronger strategies than training against fixed opponents?
* Can an agent trained against a population of opponents generalize to unseen strategies?
* Which combination of game-state information, algorithms, and training approaches produces the strongest policy?

---

## 6. Proposed Solution

The project will be developed as a layered system:

```text
                Flip 7 Game Simulator
                         │
                         ▼
              RL / Multi-Agent Environment
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Baseline Agents          RL Agents
       Heuristics / EV          PPO / MARL
              │                     │
              └──────────┬──────────┘
                         ▼
                 Self-Play Training
                         │
                         ▼
                Opponent Population
                         │
                         ▼
                  Evaluation System
                         │
                         ▼
                Strong Playing Agent

```

Mathematical and heuristic strategies will be implemented alongside learned agents. This provides meaningful baselines and allows the project to determine whether RL actually provides an advantage over simpler approaches.

The architecture should remain modular so that game mechanics, observations, policies, training algorithms, and evaluation methods can evolve independently.

---

## 7. Project Scope

The initial project will focus on:

* Accurate simulation of Flip 7.
* Multi-player game support.
* Turn-based decision-making.
* Hit/Stay decisions.
* Action-card decisions and targeting.
* Deck depletion and card-history tracking.
* Heuristic and mathematical agents.
* RL-based agents.
* Self-play training.
* Opponent populations.
* Quantitative evaluation and benchmarking.
* Experiment tracking and reproducibility.

The project will initially focus on decision-making from structured game state, rather than visual recognition of physical cards. Computer vision, physical-game integration, and other advanced interfaces are considered future extensions rather than core requirements.

---

## 8. Success Criteria

The project will be considered successful if it produces:

* A reliable and validated Flip 7 simulator.
* A standardized RL environment suitable for large-scale simulation.
* Multiple competitive baseline agents.
* An RL agent that demonstrates measurable improvement over appropriate baselines.
* Evaluation against opponents not encountered during training.
* Evidence showing which state information and learning approaches contribute to performance.
* Reproducible experiments and documented results.

The primary performance metric will be game win rate, supported by secondary metrics such as average score, bust rate, round performance, and performance against different opponent strategies.

---

## 9. Technology & System Direction

The project foundation uses a small, reproducible Python stack:

* **Python 3.12+** for the primary implementation.
* **uv** for dependency resolution and lockfiles.
* **pytest**, **Ruff**, **Pyright**, and **pre-commit** for quality automation.
* **GitHub Actions** and optional **Docker** for reproducible CI and development.

Later research phases may add:

* **NumPy** for simulation and numerical computation.
* **PettingZoo / Gymnasium** for RL environment interfaces.
* **PyTorch** for neural-network policies and training.
* **PPO / MARL algorithms** for policy learning.
* **Monte Carlo / dynamic programming** for analytical and algorithmic baselines.
* **Ray** where distributed simulation or training becomes beneficial.
* **MLflow / Weights & Biases** for experiment tracking.
* **Optuna** for hyperparameter optimization.
* **Docker** for reproducible environments.
* **FastAPI / ONNX** for potential model serving.
* **Automated testing and CI/CD** for software reliability.

Technology choices may evolve as the project develops. Infrastructure will be added where it provides measurable value rather than being required by the game itself.

---

## 10. Future Potential

The project can eventually expand beyond the initial RL agent into a broader game-AI platform.

Potential extensions include:

* Human-vs-agent gameplay.
* A web-based game interface.
* Advanced recurrent or Transformer-based agents.
* POMDP and belief-state modeling.
* Advanced multi-agent reinforcement learning.
* Population-based and league training.
* Opponent modeling and behavioral adaptation.
* Automated strategy analysis and visualization.
* High-performance distributed simulation.
* Generalization of the environment architecture to other stochastic board and card games.

The longer-term goal is not simply to create a Flip 7 bot, but to use Flip 7 as a complete experimental platform for studying learned decision-making in competitive, stochastic environments.
