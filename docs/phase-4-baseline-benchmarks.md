# Phase 4 Baseline Benchmark Results

## Status

Initial qualitative benchmark. These results are intended to establish a
reproducible reference point for Phase 4, not a final ranking or publication-
quality tournament.

## Method

- Environment: `Flip7AECEnv`
- Observation family: `deck_aware`
- Games per lineup: 100
- Seeds: `700` for the five-player lineup and `1700` for the three-player lineup
- Primary metric: win share, where tied winners split one win
- Secondary metrics: average final score, bust rate, action counts, and Flip 7
  frequency
- Agent seats were fixed within each lineup; seat rotation and confidence
  intervals are reserved for the next evaluation pass.

## Results

### Five-player lineup

Lineup: `random`, `threshold`, `risk`, `ev`, `dp`.

| Agent | Win share | Average final score | Bust rate |
| --- | ---: | ---: | ---: |
| Risk / bust probability | 69% | 202.82 | 12.33% |
| Fixed threshold | 27% | 179.49 | 11.97% |
| Random legal | 3% | 133.96 | 7.24% |
| Round DP | 1% | 115.59 | 0.79% |
| Expected value | 0% | 126.06 | 2.23% |

Flip 7 frequency was 4%; no ties occurred.

### Three-player lineup

Lineup: `risk`, `ev`, `dp`.

| Agent | Win share | Average final score | Bust rate |
| --- | ---: | ---: | ---: |
| Risk / bust probability | 96% | 212.44 | 9.49% |
| Expected value | 2% | 138.79 | 1.70% |
| Round DP | 2% | 120.01 | 0.67% |

No ties or Flip 7 events occurred.

## Interpretation

The risk agent is currently the strongest baseline because it directly models
Flip 7's central push-your-luck decision: whether the next draw is likely to
bust. The threshold agent is a useful transparent fallback and performs
meaningfully above random in this lineup.

The expected-value and DP agents should not yet be interpreted as strong
algorithmic strategies. Their current Phase 4 models are intentionally narrow:
they do not fully model future draws, bust consequences, modifiers, action
cards, targeting, opponent strategy, or cumulative game position.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe -c "from flip7.agents import *; from flip7.evaluation import run_matchup; roster={'random':lambda:RandomLegalAgent(seed=101),'threshold':lambda:FixedThresholdAgent(threshold=15),'risk':lambda:BustProbabilityAgent(risk_tolerance=.2),'ev':lambda:ExpectedValueAgent(),'dp':lambda:RoundDPAgent()}; print(run_matchup(('random','threshold','risk','ev','dp'),roster,games=100,seed=700).as_dict()); print(run_matchup(('risk','ev','dp'),roster,games=100,seed=1700).as_dict())"
```

The benchmark configuration is recorded in
[`configs/baselines.yaml`](../configs/baselines.yaml). Future reports should
rotate seats, use larger seed sets, and add uncertainty intervals before
making stronger claims about relative agent strength.
