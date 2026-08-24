# ADR 0001: PettingZoo AEC as the native environment API

## Status

Accepted.

## Context

Phase 3 must expose the Flip 7 engine as a standardized RL environment.
The engine is already a sequential multi-agent process: 3 to 18 players, one
legal decision at a time, with Hit/Stay turns and Action-card targeting mixed
in the same episode.

Gymnasium models a single `reset()` / `step()` loop. That matches Phase 5 PPO
against frozen opponents, but it hides every other seat inside the environment
and would force a rewrite for self-play and MAPPO.

PettingZoo's Parallel API assumes simultaneous actions. Flip 7 players do not
act at the same time.

## Decision

The native environment is `Flip7AECEnv`, a PettingZoo Agent Environment Cycle
environment wrapping `Flip7Engine`.

A thin Gymnasium wrapper, `Flip7VsOpponentsEnv`, presents one learning seat
against frozen opponent callables. It is an adapter over the AEC environment,
not a second rules implementation.

## Consequences

- Phase 5 can train PPO against baselines without treating Flip 7 as a
  single-agent game.
- Phase 7 self-play and MARL can use the same AEC environment.
- Illegal simultaneous-action shortcuts are out of scope.
- SuperSuit, RLlib, and training libraries are not introduced by this decision.
