"""Persistent per-Pokémon volatile flags carried from Showdown observations."""

# Keep this list fixed and versioned: observation positions 142:502 contain
# these 30 binary flags for each of the 12 roster members, in roster order.
# Perish and trapping retain their established dedicated observation fields.
POKEMON_VOLATILE_EFFECTS = (
    "CONFUSION",
    "DISABLE",
    "ENCORE",
    "TAUNT",
    "TORMENT",
    "SUBSTITUTE",
    "LEECHSEED",
    "YAWN",
    "PARTIALLYTRAPPED",
    "BIND",
    "WRAP",
    "FIRESPIN",
    "INFESTATION",
    "MAGMASTORM",
    "THUNDERCAGE",
    "WHIRLPOOL",
    "SANDTOMB",
    "INGRAIN",
    "MAGNETRISE",
    "TELEKINESIS",
    "SMACKDOWN",
    "AQUARING",
    "ROOST",
    "CHARGE",
    "MUSTRECHARGE",
    "LOCKEDMOVE",
    "FOCUSENERGY",
    "LASERFOCUS",
    "NORETREAT",
    "GLAIVERUSH",
)

OBSERVATION_VOLATILE_OFFSET = 142
OBSERVATION_VOLATILE_WIDTH = 12 * len(POKEMON_VOLATILE_EFFECTS)
EVALUATOR_BASE_FIELD_WIDTH = 100 + OBSERVATION_VOLATILE_WIDTH
FIELD_FEATURE_SCHEMA = "vgc_state_volatile_effects_v1"
