"""Feature encoding system for Pokemon battle states."""

from src.features.tensors import (
    TensorShapes,
    DEFAULT_TENSOR_SHAPES,
    POKEMON_INDEX_MAP,
    MOVE_INDEX_MAP,
    ITEM_INDEX_MAP,
    ABILITY_INDEX_MAP,
    STATUS_INDEX_MAP,
    WEATHER_INDEX_MAP,
    TERRAIN_INDEX_MAP,
    TERA_TYPE_INDEX_MAP,
)
from src.features.encoding import (
    FeatureEncoder,
    EncodedBattleState,
)

__all__ = [
    "TensorShapes",
    "DEFAULT_TENSOR_SHAPES",
    "FeatureEncoder",
    "EncodedBattleState",
    "POKEMON_INDEX_MAP",
    "MOVE_INDEX_MAP",
    "ITEM_INDEX_MAP",
    "ABILITY_INDEX_MAP",
    "STATUS_INDEX_MAP",
    "WEATHER_INDEX_MAP",
    "TERRAIN_INDEX_MAP",
    "TERA_TYPE_INDEX_MAP",
]
