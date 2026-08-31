"""Alternating battle and team-selection training schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


Phase = Literal["battle", "team_selection"]


@dataclass(frozen=True)
class AlternatingConfig:
    battle_updates: int = 1
    team_selection_updates: int = 1

    def __post_init__(self) -> None:
        if self.battle_updates < 1 or self.team_selection_updates < 1:
            raise ValueError("alternating update counts must be positive")


class AlternatingTrainer:
    """Run a deterministic schedule over independent policy trainers."""

    def __init__(self, config: AlternatingConfig = AlternatingConfig()):
        self.config = config
        self.phase: Phase = "battle"
        self.update_count = 0

    def run(
        self,
        updates: int,
        train_battle: Callable[[], None],
        train_team_selection: Callable[[], None],
    ) -> list[Phase]:
        if updates < 0:
            raise ValueError("updates must be non-negative")
        phases: list[Phase] = []
        phase_updates = 0
        for _ in range(updates):
            phase_updates += 1
            if self.phase == "battle":
                train_battle()
                phases.append("battle")
                if phase_updates == self.config.battle_updates:
                    self.phase = "team_selection"
                    phase_updates = 0
            else:
                train_team_selection()
                phases.append("team_selection")
                if phase_updates == self.config.team_selection_updates:
                    self.phase = "battle"
                    phase_updates = 0
            self.update_count += 1
        return phases
