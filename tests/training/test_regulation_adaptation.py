import numpy as np
import pytest

from src.training.regulation_adaptation import RegulationAdapter, ReplayExampleMetadata
from src.utils.config import REGULATION_CHAMPIONS_MB


def test_filter_action_mask_intersects_current_legality() -> None:
    adapter = RegulationAdapter(REGULATION_CHAMPIONS_MB)
    result = adapter.filter_action_mask([1, 1, 1, 0], [0, 2])
    np.testing.assert_array_equal(result, [1, 0, 1, 0])


def test_filter_action_mask_rejects_empty_intersection() -> None:
    adapter = RegulationAdapter(REGULATION_CHAMPIONS_MB)
    with pytest.raises(ValueError, match="every action"):
        adapter.filter_action_mask([1, 0], [1])


def test_historical_replay_is_downweighted() -> None:
    adapter = RegulationAdapter(REGULATION_CHAMPIONS_MB, historical_weight=0.2)
    metadata = [
        ReplayExampleMetadata("champions-mb", (0,)),
        ReplayExampleMetadata("future-regulation", (0,)),
    ]
    probabilities = adapter.weighted_indices(metadata)
    assert probabilities[0] > probabilities[1]
    assert np.isclose(probabilities.sum(), 1.0)
