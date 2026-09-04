"""Training orchestration for learned policies."""

from flip7.training.league import (
    LeagueConfig,
    LeaguePPOTrainer,
    PolicyLeague,
    PolicySnapshot,
    write_population,
)
from flip7.training.league_followup import (
    DiverseLeaguePPOTrainer,
    DiversePolicyLeague,
    FollowUpLeagueConfig,
    write_followup_population,
)
from flip7.training.mappo import (
    CentralizedCritic,
    MAPPOAgent,
    MAPPOConfig,
    MAPPORollout,
    MAPPOTrainer,
    SharedActor,
    write_mappo_history,
)
from flip7.training.ppo import (
    EpisodeLineup,
    OpponentProvider,
    PPOConfig,
    PPOTrainer,
    Rollout,
    SeatOpponentProvider,
    baseline_factories,
    compute_gae,
    write_history,
)
from flip7.training.stability import (
    BalancedBaselineProvider,
    SeatBalancedPPOTrainer,
    StabilityControlPPOTrainer,
    StabilityLeaguePPOTrainer,
    balanced_seat_quotas,
)

__all__ = [
    "PPOConfig",
    "PPOTrainer",
    "Rollout",
    "EpisodeLineup",
    "OpponentProvider",
    "SeatOpponentProvider",
    "LeagueConfig",
    "LeaguePPOTrainer",
    "PolicyLeague",
    "PolicySnapshot",
    "baseline_factories",
    "compute_gae",
    "write_history",
    "BalancedBaselineProvider",
    "SeatBalancedPPOTrainer",
    "StabilityControlPPOTrainer",
    "StabilityLeaguePPOTrainer",
    "balanced_seat_quotas",
    "write_population",
    "DiverseLeaguePPOTrainer",
    "DiversePolicyLeague",
    "FollowUpLeagueConfig",
    "write_followup_population",
    "CentralizedCritic",
    "MAPPOAgent",
    "MAPPOConfig",
    "MAPPORollout",
    "MAPPOTrainer",
    "SharedActor",
    "write_mappo_history",
]
