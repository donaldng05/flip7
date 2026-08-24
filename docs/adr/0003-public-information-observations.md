# ADR 0003: Public-information observation families

## Status

Accepted.

## Context

Requirements ask for progressively richer observations. The rules specification
makes cards in play and discards public, and it forbids leaking draw-pile order
into agent observations. Phase 6 will ablate information; Phase 3 must freeze a
layout so later vectors do not silently reshape.

## Decision

Observations are `float32` Gymnasium `Box` vectors selected by configuration:

- `basic` — observing player's public cards, status, score, round, pending
  Action, dealer flag, and Flip Three remainder.
- `competitive` (default) — `basic` plus every seat's public cards, ego /
  current / dealer / pending-resolver one-hots.
- `deck_aware` — `competitive` plus remaining draw-face counts and pile sizes,
  never draw order.

`historical` / recurrent observations are out of Phase 3. Exact indices live in
`docs/environment-specification.md`.

## Consequences

- Default competitive observations contain opponent cards because those cards
  are public, not because the agent is cheating.
- Card-counting experiments can switch `deck_aware` without changing the engine.
- A later seat-relative or history encoding is an additive family, not a silent
  rewrite of `competitive`.
