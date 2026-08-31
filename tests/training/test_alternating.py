import pytest

from src.training.alternating import AlternatingConfig, AlternatingTrainer


def test_alternates_training_phases() -> None:
    calls: list[str] = []
    trainer = AlternatingTrainer(AlternatingConfig(battle_updates=2, team_selection_updates=1))
    phases = trainer.run(5, lambda: calls.append("battle"), lambda: calls.append("team"))
    assert phases == ["battle", "battle", "team_selection", "battle", "battle"]
    assert calls == ["battle", "battle", "team", "battle", "battle"]


def test_rejects_invalid_schedule() -> None:
    with pytest.raises(ValueError):
        AlternatingConfig(battle_updates=0)
