# Flip 7 Environment Specification

This document is the normative observation, action, reward, and adapter layout
for Phase 3. The rules engine remains [`game-rules-specification.md`](game-rules-specification.md).
Architecture choices are recorded under [`adr/`](adr/README.md).

## 1. Boundaries

- `flip7.core` owns deterministic rules, state, and seeded RNG objects.
- `flip7.envs` adapts `Flip7Engine` to PettingZoo and Gymnasium.
- Agents, training, and evaluation consume environment observations; they must
  not call engine internals except through the adapters.
- Draw-pile **order** is never part of an observation. Remaining-face **counts**
  appear only in the `deck_aware` family.

## 2. Native API: PettingZoo AEC

`Flip7AECEnv` is the source of truth.

- Agents: `"player_0"` … `"player_{n-1}"` matching `player_id`.
- `reset(seed, options)` builds `Flip7Engine(random.Random(seed), n, **options)`.
- Supported `options`: `dealer_id`, `draw_pile`, `discard_pile`,
  `starting_scores`.
- One `step(action)` applies one `engine.apply(...)`.
- `agent_selection` is the engine's current deciding player.
- After the game ends, remaining agents terminate and must `step(None)`.
- `info["action_mask"]` is `int8` and aligned with the discrete action space.

`Flip7VsOpponentsEnv` is a Gymnasium wrapper around that AEC environment. It
advances frozen opponent policies until the configured learner seat must act.

## 3. Action space

Size: `2 + player_count`.

| Index | Meaning |
| --- | --- |
| `0` | Hit |
| `1` | Stay |
| `2 + target_id` | Assign the pending Action card to seated player `target_id` |

The pending Action card is read from `state.pending_action`. It is not encoded
in the integer. Unmasked actions raise `InvalidActionError`.

## 4. Shared player features

Each seated player occupies 26 `float32` values:

| Index | Feature |
| --- | --- |
| `0`–`12` | Occupancy of Number cards 0–12 |
| `13` | Second Chance occupancy |
| `14`–`19` | Occupancy of `+2`, `+4`, `+6`, `+8`, `+10`, `x2` |
| `20`–`23` | Status one-hot: active, stayed, frozen, busted |
| `24` | Unique Number-card count / 7 |
| `25` | Cumulative score / 200 |

Pending-Action one-hot (4 values): none, Freeze, Flip Three, Second Chance.

## 5. Observation families

### 5.1 `basic` (length 33)

Independent of player count.

| Index | Feature |
| --- | --- |
| `0`–`25` | Observing player's features |
| `26` | Round number / 30 |
| `27`–`30` | Pending-Action one-hot |
| `31` | Observing player is dealer |
| `32` | Flip Three cards remaining / 3, or 0 |

### 5.2 `competitive` (length `30n + 6`, default)

Absolute seat order. Only the ego one-hot differs across observers.

| Index | Feature |
| --- | --- |
| `0`–`26n-1` | Player features for seats `0 … n-1` |
| `26n`–`27n-1` | Ego one-hot |
| `27n`–`28n-1` | Current-player one-hot, or zeros |
| `28n`–`29n-1` | Dealer one-hot |
| `29n`–`30n-1` | Pending-resolver one-hot, or zeros |
| `30n` | Round number / 30 |
| `30n+1`–`30n+4` | Pending-Action one-hot |
| `30n+5` | Flip Three cards remaining / 3, or 0 |

For `n = 3` the length is 96. For `n = 4` it is 126.

### 5.3 `deck_aware` (length `30n + 30`)

`competitive`, then:

| Offset after competitive | Feature |
| --- | --- |
| `0`–`12` | Remaining draw counts for Number cards 0–12 |
| `13`–`15` | Remaining draw counts for Freeze, Flip Three, Second Chance |
| `16`–`21` | Remaining draw counts for `+2`, `+4`, `+6`, `+8`, `+10`, `x2` |
| `22` | Draw-pile size |
| `23` | Discard-pile size |

Counts are order-independent. Two draw piles with the same face counts must
encode identically.

`historical` observations are not defined here.

## 6. Rewards and episodes

One episode is one game to the 200-point threshold.

- `sparse_win` (default): `0` until `GAME_ENDED`; each winner receives `1 / k`.
- `round_score`: also add each recorded round score / 200.

There is no truncation limit in Phase 3.

## 7. Configuration

`configs/defaults.yaml`:

```yaml
seed: 7
players: 3
env:
  observation: competitive
  reward: sparse_win
  learner_id: 0
```

`players` must be in `3 … 18`. Observation and reward names match
`ObservationFamily` and `RewardMode`.
