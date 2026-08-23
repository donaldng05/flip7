# Rules Specification Checklist

## Rule coverage

- [ ] Source authority and version are identified.
- [ ] Deck composition and card distribution are explicit.
- [ ] Setup, player count, turn order, and round lifecycle are explicit.
- [ ] Every action has preconditions, effects, and legal-target behavior.
- [ ] Number-card duplicates, busts, special cards, and modifiers are defined.
- [ ] Round scoring, Flip 7 scoring, cumulative scoring, and victory are defined.
- [ ] Empty-deck, simultaneous, tie, and interruption cases are resolved.

## Terminology

Use one term consistently for each concept. Distinguish:

- card draw from action-card resolution
- a player’s round score from cumulative game score
- a legal action from an agent’s selected action
- visible information from complete internal state
- round completion from game completion

## Required examples

Include examples for a normal hit/stay sequence, a duplicate-number bust, every
special-card action, a Flip 7, a target-selection decision, and a game ending
at the winning threshold.
