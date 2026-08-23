---
name: flip7-rules-specification
description: Author and review the authoritative Flip 7 rules specification before game-engine implementation. Use when defining game rules, resolving ambiguities, or preparing rule tests.
disable-model-invocation: true
---

# Flip 7 Rules Specification

Use this skill before implementing `flip7.core`.

1. Read `docs/project-overview.md`, `docs/requirements-and-scope.md`, and
   `docs/project-plan.md`.
2. Separate facts confirmed by the source rules from assumptions and unresolved
   ambiguities. Do not silently choose between conflicting interpretations.
3. Specify deck composition, card types, setup, round phases, turn order,
   actions, legal actions, targeting, scoring, busting, Flip 7, game victory,
   player counts, and edge cases.
4. Add worked examples for ordinary turns, special cards, busts, Flip 7, and
   game-ending conditions.
5. Write rule tests for each decision before implementing the engine.
6. Request review of unresolved questions and record each final decision.

The output is a reviewable `game-rules-specification.md` plus tests; it is not
an engine implementation. Use [reference.md](reference.md) as the checklist.
