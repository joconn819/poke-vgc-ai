"""Generate synthetic supervised datasets from heuristic self-play."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.data import SyntheticBattleDataset
from src.training.synthetic_battles import BattleGenerator
from src.utils.config import REGULATION_CHAMPIONS_MB
from src.utils.team_pool import TeamPool


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic VGC training data")
    parser.add_argument("--num-battles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, default=Path("data/generated/synthetic_dataset.pt"))
    args = parser.parse_args()

    team_pool = TeamPool(
        teams_file="data/teams/champions_mb.json",
        regulation=REGULATION_CHAMPIONS_MB,
    )
    generator = BattleGenerator(
        regulation_config=REGULATION_CHAMPIONS_MB,
        seed=args.seed,
        player_agent_type="max_damage",
        opponent_agent_type="random",
    )
    dataset = SyntheticBattleDataset.from_generator(
        generator=generator,
        team_pool=team_pool,
        num_battles=args.num_battles,
        regulation_config=REGULATION_CHAMPIONS_MB,
    )
    dataset.save(args.output_path)
    print(f"saved {len(dataset)} examples to {args.output_path}")
    print(dataset.get_statistics())


if __name__ == "__main__":
    main()
