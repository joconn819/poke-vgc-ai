"""Stable token IDs for battle features beyond the original toy vocabularies."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Mapping

from poke_env.data.normalize import to_id_str

from src.features.tensors import (
    ABILITY_INDEX_MAP,
    ITEM_INDEX_MAP,
    MOVE_INDEX_MAP,
    POKEMON_INDEX_MAP,
)


LEGACY_VOCAB_SIZES = {
    "pokemon": 1025,
    "move": 656,
    "item": 376,
    "ability": 267,
}
_MAPS: dict[str, Mapping[str, int]] = {
    "pokemon": POKEMON_INDEX_MAP,
    "move": MOVE_INDEX_MAP,
    "item": ITEM_INDEX_MAP,
    "ability": ABILITY_INDEX_MAP,
}
_NORMALIZED_MAPS = {
    kind: {to_id_str(name): index for name, index in values.items()}
    for kind, values in _MAPS.items()
}


@lru_cache(maxsize=65536)
def token_id(kind: str, value: str | None, *, vocab_version: int) -> int:
    """Encode a token deterministically; zero remains reserved for padding.

    Version 0 reproduces the historical maps for old checkpoints. Version 1
    retains their non-padding IDs and hashes every other canonical Showdown
    token into the remaining embedding rows. Hashes are stable across Python
    processes and do not depend on hidden opponent set information.
    """
    if kind not in _MAPS:
        raise ValueError(f"unsupported battle token kind {kind!r}")
    if vocab_version not in {0, 1}:
        raise ValueError(f"unsupported token vocabulary version {vocab_version}")
    if not value:
        return 0
    canonical = to_id_str(str(value))
    if kind == "move" and canonical == "protect":
        canonical = "protected"
    if not canonical:
        return 0

    if vocab_version == 0:
        return int(_NORMALIZED_MAPS[kind].get(canonical, 0))

    known_id = _NORMALIZED_MAPS[kind].get(canonical)
    if known_id is not None and known_id > 0:
        return int(known_id)

    size = LEGACY_VOCAB_SIZES[kind]
    reserved = max(_MAPS[kind].values()) + 1
    if known_id == 0:
        # The first row was historically both a real token ID and the padding
        # row. Give that one known token a non-padding reserved row.
        return reserved
    first_hash_id = reserved + 1
    bucket_count = size - first_hash_id
    if bucket_count <= 0:
        raise RuntimeError(f"no hash buckets remain in {kind} vocabulary")
    digest = hashlib.blake2b(
        f"{kind}:{canonical}".encode("utf-8"), digest_size=8
    ).digest()
    return first_hash_id + int.from_bytes(digest, "little") % bucket_count
