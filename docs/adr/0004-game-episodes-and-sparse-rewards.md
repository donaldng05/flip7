# ADR 0004: Game-length episodes and sparse win rewards

## Status

Accepted.

## Context

The primary evaluation metric is game win rate. Games end after a round in which
at least one player reaches 200 points. Ties are reported without a tie-breaker.
Shaped score rewards can teach point farming instead of winning.

## Decision

One environment episode is one complete game.

The default reward `sparse_win` is `0` until the game ends. Each winner then
receives `1 / k` where `k` is the number of tied winners; other seats receive
`0`.

`round_score` is an optional config that also adds each recorded round score
divided by 200. It is for later reward-design experiments, not the default.

There is no `max_rounds` truncation in Phase 3.

## Consequences

- Credit assignment is sparse. PPO against this default may need many games or
  a later shaping experiment.
- Evaluation can read `winning_player_ids` from the engine rather than inferring
  winners from shaped returns.
- Episode length varies with play; vectorized time limits are a later concern.
