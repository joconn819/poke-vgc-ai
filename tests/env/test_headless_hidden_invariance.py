"""Exercise the player-facing baseline against a changed unrevealed set."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_showdown_self_play import generate_showdown_champions_team_data
from src.baselines.native_damage import choose_max_damage_action, choose_public_counterplay_action
from src.env.headless_showdown import HeadlessShowdownProcess
from src.env.showdown_env import ShowdownDoublesEnv
from src.features.public_mechanics import encode_public_mechanics
from src.models.policy import TransformerPolicy
from src.training.self_play import PolicyAgent


SHOWDOWN = Path(os.environ.get("POKEMON_SHOWDOWN_PATH", "/home/justin/p/poke_learning/pokemon-showdown"))
FORMAT = "gen9championsdoubles2026mcclosed"
CHECKPOINT = Path("training/outputs/phase12_headless/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt")


@pytest.fixture(scope="module")
def incumbent_model():
    if not CHECKPOINT.is_file():
        pytest.skip("Round430 checkpoint not available")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = TransformerPolicy.from_checkpoint_state(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


@pytest.mark.skipif(not (SHOWDOWN / "dist/sim").is_dir(), reason="local Showdown simulator unavailable")
@pytest.mark.parametrize("acting_side", [0, 1])
@pytest.mark.parametrize(
    "decision_point,hidden_slot",
    [("initial", 0), ("initial", 3), ("after_one_turn", 3)],
    ids=["initial-active", "initial-bench", "later-bench"],
)
@pytest.mark.parametrize("hidden_field", ["moves_item", "ability"])
def test_unrevealed_set_does_not_change_player_view_or_decision(
    acting_side, decision_point, hidden_slot, hidden_field, incumbent_model
):
    teams = [generate_showdown_champions_team_data(922001 + i, FORMAT) for i in range(2)]
    original = teams[1 - acting_side][hidden_slot]
    if hidden_field == "moves_item":
        original["item"] = "leftovers"
    else:
        # Neither ability announces itself when the Pokémon enters.
        original["ability"] = "innerfocus"
    alternative = deepcopy(teams)
    hidden = alternative[1 - acting_side][hidden_slot]
    if hidden_field == "moves_item":
        hidden["moves"] = ["protect", "tackle", "rest", "sleeptalk"]
        hidden["item"] = "sitrusberry"
    else:
        hidden["ability"] = "owntempo"
    observed = []
    transition_choices = None
    for world in (teams, alternative):
        with HeadlessShowdownProcess(*world, format_id=FORMAT, seed=32145, showdown_path=SHOWDOWN) as match:
            match.advance("team 1234", "team 1234")
            if decision_point == "after_one_turn":
                observation1, mask1, observation2, mask2 = match.observations()
                if transition_choices is None:
                    actions = [
                        choose_public_counterplay_action(
                            mask1, match.battles[0], match.public_history,
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        ),
                        choose_public_counterplay_action(
                            mask2, match.battles[1], match.public_history,
                            order_validator=ShowdownDoublesEnv.action_to_order,
                        ),
                    ]
                    transition_choices = match.choices_from_native_actions(actions)
                match.advance(*transition_choices)
            observation1, mask1, observation2, mask2 = match.observations()
            observation, mask = ((observation1, mask1), (observation2, mask2))[acting_side]
            battle = match.battles[acting_side]
            if getattr(battle, "_wait", False):
                # The other side's request is still pending; there is no
                # decision to make, and action zero is the protocol sentinel.
                max_damage = np.zeros(2, dtype=np.int64)
                public = np.zeros(2, dtype=np.int64)
            else:
                max_damage = choose_max_damage_action(
                    mask, battle, order_validator=ShowdownDoublesEnv.action_to_order
                )
                public = choose_public_counterplay_action(
                    mask, battle, match.public_history,
                    order_validator=ShowdownDoublesEnv.action_to_order,
                )
            public_opponent_roster = [
                {"species": str(member.get("species") or member.get("name"))}
                for member in world[1 - acting_side]
            ]
            public_context = encode_public_mechanics(
                match.public_mechanics()[acting_side],
                observation,
                world[acting_side],
                public_opponent_roster,
                FORMAT,
            )
            policy = PolicyAgent(
                incumbent_model,
                world[acting_side],
                world[1 - acting_side],
                deterministic=True,
                public_context_features=public_context,
            )
            policy_action, log_probability, value = policy.predict_native_with_value(observation, mask, use_critic=True)
            observed.append((
                observation.copy(), mask.copy(), max_damage.copy(), public.copy(),
                public_context.copy(), policy_action.copy(),
                np.asarray([log_probability, value]),
            ))
    for original, changed in zip(*observed, strict=True):
        np.testing.assert_array_equal(original, changed)
