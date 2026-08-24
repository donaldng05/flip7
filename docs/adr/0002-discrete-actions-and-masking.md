# ADR 0002: Flattened discrete actions with masks

## Status

Accepted.

## Context

The engine exposes typed actions: `HitAction`, `StayAction`, and
`TargetAction`. RL libraries expect a stable discrete or box space. The pending
Action card is already in the game state, so it must not also be chosen as part
of the integer action. Invalid actions are already rejected by the engine.

## Decision

Use one `Discrete(2 + player_count)` space for every seat:

- `0` Hit
- `1` Stay
- `2 + target_id` assign the pending Action card to that seated player

Publish `info["action_mask"]` as an `int8` vector aligned with that space.
Stepping an unmasked action raises the same domain error the engine would raise.
Policies must sample from the mask. The environment does not no-op, clip, or
repair illegal actions.

Absolute seat ids are used in Phase 3. Seat-relative action indices are deferred
to Phase 6.

## Consequences

- Action masking is required for training.
- The PettingZoo `api_test` helper, which samples the raw action space, is not
  a valid environment acceptance test.
- Changing player count changes the action-space size; a policy is not assumed
  to transfer across player counts until a later phase measures that.
