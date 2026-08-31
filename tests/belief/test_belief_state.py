"""Tests for belief state modeling."""

import pytest
from src.belief.belief_state import PokemonBeliefState, OpponentBeliefState


class TestPokemonBeliefState:
    """Tests for PokemonBeliefState class."""

    def test_init_with_known_values(self):
        """Test initialization with known (certain) values."""
        belief = PokemonBeliefState(
            name="Salamence",
            hp_min=130,
            hp_max=130,
            moves={"Earthquake", "Outrage", "Protect", "Superpower"},
            items={"Choice Scarf"},
            abilities={"Intimidate"},
            tera_types={"Water"},
        )

        assert belief.name == "Salamence"
        assert belief.hp_min == 130
        assert belief.hp_max == 130
        assert belief.moves == {"Earthquake", "Outrage", "Protect", "Superpower"}
        assert belief.items == {"Choice Scarf"}
        assert belief.abilities == {"Intimidate"}
        assert belief.tera_types == {"Water"}

    def test_init_with_unknown_values(self):
        """Test initialization with unknown (uncertain) values."""
        belief = PokemonBeliefState(name="Kyogre")

        assert belief.name == "Kyogre"
        assert belief.hp_min == 0
        assert belief.hp_max == 255
        assert belief.moves == set()
        assert belief.items == set()
        assert belief.abilities == set()
        assert belief.tera_types == set()

    def test_init_with_stat_ranges(self):
        """Test initialization with stat min/max values."""
        belief = PokemonBeliefState(
            name="Groudon",
            attack_min=100,
            attack_max=200,
            defense_min=80,
            defense_max=150,
            spa_min=90,
            spa_max=180,
            spd_min=85,
            spd_max=160,
            speed_min=40,
            speed_max=120,
        )

        assert belief.attack_min == 100
        assert belief.attack_max == 200
        assert belief.defense_min == 80
        assert belief.defense_max == 150
        assert belief.spa_min == 90
        assert belief.spa_max == 180
        assert belief.spd_min == 85
        assert belief.spd_max == 160
        assert belief.speed_min == 40
        assert belief.speed_max == 120

    def test_expected_hp(self):
        """Test expected HP calculation."""
        belief = PokemonBeliefState(name="Salamence", hp_min=120, hp_max=140)
        expected = belief.expected_hp()
        assert expected == 130.0

    def test_expected_stat(self):
        """Test expected stat calculation."""
        belief = PokemonBeliefState(name="Salamence", attack_min=100, attack_max=200)
        expected = belief.expected_attack()
        assert expected == 150.0

    def test_variance_hp(self):
        """Test HP variance calculation."""
        belief = PokemonBeliefState(name="Salamence", hp_min=100, hp_max=150)
        variance = belief.variance_hp()
        # Variance of uniform distribution: (b - a)^2 / 12
        expected_variance = (150 - 100) ** 2 / 12
        assert abs(variance - expected_variance) < 0.01

    def test_variance_stat(self):
        """Test stat variance calculation."""
        belief = PokemonBeliefState(name="Salamence", attack_min=120, attack_max=180)
        variance = belief.variance_attack()
        expected_variance = (180 - 120) ** 2 / 12
        assert abs(variance - expected_variance) < 0.01

    def test_update_moves(self):
        """Test updating possible movesets."""
        belief = PokemonBeliefState(name="Salamence")
        belief.update_moves({"Earthquake", "Outrage", "Protect"})

        assert belief.moves == {"Earthquake", "Outrage", "Protect"}

    def test_update_moves_intersection(self):
        """Test that updating moves uses intersection when already set."""
        belief = PokemonBeliefState(
            name="Salamence", moves={"Earthquake", "Outrage", "Protect", "Superpower"}
        )
        # Discover it must have Earthquake or Outrage
        belief.update_moves({"Earthquake", "Outrage", "Recover"})

        # Should be intersection
        assert belief.moves == {"Earthquake", "Outrage"}

    def test_update_hp_range(self):
        """Test updating HP range."""
        belief = PokemonBeliefState(name="Salamence", hp_min=100, hp_max=200)
        belief.update_hp_range(120, 180)

        assert belief.hp_min == 120
        assert belief.hp_max == 180

    def test_update_attack_range(self):
        """Test updating attack stat range."""
        belief = PokemonBeliefState(name="Salamence", attack_min=100, attack_max=200)
        belief.update_attack_range(130, 170)

        assert belief.attack_min == 130
        assert belief.attack_max == 170

    def test_update_items(self):
        """Test updating possible items."""
        belief = PokemonBeliefState(name="Salamence")
        belief.update_items({"Choice Scarf", "Life Orb", "Assault Vest"})

        assert belief.items == {"Choice Scarf", "Life Orb", "Assault Vest"}

    def test_update_abilities(self):
        """Test updating possible abilities."""
        belief = PokemonBeliefState(name="Salamence")
        belief.update_abilities({"Intimidate", "Moxie"})

        assert belief.abilities == {"Intimidate", "Moxie"}

    def test_update_tera_types(self):
        """Test updating possible tera types."""
        belief = PokemonBeliefState(name="Salamence")
        belief.update_tera_types({"Water", "Flying", "Dragon"})

        assert belief.tera_types == {"Water", "Flying", "Dragon"}

    def test_all_stats_getters(self):
        """Test all stat getter methods."""
        belief = PokemonBeliefState(
            name="Salamence",
            hp_min=100,
            hp_max=150,
            attack_min=120,
            attack_max=180,
            defense_min=80,
            defense_max=140,
            spa_min=90,
            spa_max=150,
            spd_min=85,
            spd_max=145,
            speed_min=100,
            speed_max=160,
        )

        assert belief.expected_hp() == 125.0
        assert belief.expected_attack() == 150.0
        assert belief.expected_defense() == 110.0
        assert belief.expected_spa() == 120.0
        assert belief.expected_spd() == 115.0
        assert belief.expected_speed() == 130.0

    def test_certain_hp(self):
        """Test when HP is certain (min == max)."""
        belief = PokemonBeliefState(name="Salamence", hp_min=130, hp_max=130)
        assert belief.expected_hp() == 130.0
        assert belief.variance_hp() == 0.0

    def test_certain_stat(self):
        """Test when a stat is certain (min == max)."""
        belief = PokemonBeliefState(
            name="Salamence", attack_min=160, attack_max=160
        )
        assert belief.expected_attack() == 160.0
        assert belief.variance_attack() == 0.0


class TestOpponentBeliefState:
    """Tests for OpponentBeliefState class."""

    def test_init_empty(self):
        """Test initialization with empty team."""
        belief = OpponentBeliefState()
        assert belief.team == {}
        assert len(belief.team) == 0

    def test_init_with_team(self, test_team_preview):
        """Test initialization with known team."""
        opponent_data = test_team_preview["opponent"]
        team = {
            poke["name"]: PokemonBeliefState(name=poke["name"])
            for poke in opponent_data
        }
        belief = OpponentBeliefState(team=team)

        assert len(belief.team) == 6
        assert "Kyogre" in belief.team
        assert "Groudon" in belief.team

    def test_add_pokemon(self):
        """Test adding a Pokemon to the belief state."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(name="Salamence")
        belief.add_pokemon("Salamence", pokemon_belief)

        assert "Salamence" in belief.team
        assert belief.team["Salamence"].name == "Salamence"

    def test_get_pokemon(self):
        """Test retrieving a Pokemon belief."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(name="Salamence", hp_min=130, hp_max=130)
        belief.add_pokemon("Salamence", pokemon_belief)

        retrieved = belief.get_pokemon("Salamence")
        assert retrieved is not None
        assert retrieved.name == "Salamence"
        assert retrieved.hp_min == 130

    def test_get_pokemon_not_found(self):
        """Test retrieving a non-existent Pokemon."""
        belief = OpponentBeliefState()
        retrieved = belief.get_pokemon("Salamence")
        assert retrieved is None

    def test_update_pokemon_belief(self):
        """Test updating a Pokemon's belief state."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(name="Salamence")
        belief.add_pokemon("Salamence", pokemon_belief)

        belief.update_pokemon_belief("Salamence", hp_min=120, hp_max=140)
        updated = belief.get_pokemon("Salamence")
        assert updated.hp_min == 120
        assert updated.hp_max == 140

    def test_remove_pokemon(self):
        """Test removing a Pokemon from the team."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(name="Salamence")
        belief.add_pokemon("Salamence", pokemon_belief)

        assert "Salamence" in belief.team
        belief.remove_pokemon("Salamence")
        assert "Salamence" not in belief.team

    def test_get_all_pokemon_names(self):
        """Test retrieving all Pokemon names."""
        belief = OpponentBeliefState()
        belief.add_pokemon("Salamence", PokemonBeliefState(name="Salamence"))
        belief.add_pokemon("Groudon", PokemonBeliefState(name="Groudon"))
        belief.add_pokemon("Kyogre", PokemonBeliefState(name="Kyogre"))

        names = belief.get_all_pokemon_names()
        assert set(names) == {"Salamence", "Groudon", "Kyogre"}

    def test_team_size(self):
        """Test getting team size."""
        belief = OpponentBeliefState()
        belief.add_pokemon("Salamence", PokemonBeliefState(name="Salamence"))
        belief.add_pokemon("Groudon", PokemonBeliefState(name="Groudon"))

        assert belief.team_size() == 2

    def test_update_pokemon_moves(self):
        """Test updating a Pokemon's possible moves."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(name="Salamence")
        belief.add_pokemon("Salamence", pokemon_belief)

        belief.update_pokemon_moves("Salamence", {"Earthquake", "Outrage"})
        updated = belief.get_pokemon("Salamence")
        assert updated.moves == {"Earthquake", "Outrage"}

    def test_get_pokemon_expected_hp(self):
        """Test retrieving expected HP for a specific Pokemon."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(
            name="Salamence", hp_min=120, hp_max=140
        )
        belief.add_pokemon("Salamence", pokemon_belief)

        expected = belief.get_pokemon_expected_hp("Salamence")
        assert expected == 130.0

    def test_get_pokemon_expected_hp_not_found(self):
        """Test expected HP query for non-existent Pokemon."""
        belief = OpponentBeliefState()
        result = belief.get_pokemon_expected_hp("Salamence")
        assert result is None

    def test_uncertainty_all_unknown(self):
        """Test measuring uncertainty when all values are unknown."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(name="Salamence")
        belief.add_pokemon("Salamence", pokemon_belief)

        # For unknown stats (0-255 range), variance should be high
        uncertainty = belief.get_pokemon_uncertainty("Salamence")
        assert uncertainty is not None
        assert uncertainty > 1000  # Large uncertainty for completely unknown Pokemon

    def test_uncertainty_all_known(self):
        """Test measuring uncertainty when values are known with certainty."""
        belief = OpponentBeliefState()
        pokemon_belief = PokemonBeliefState(
            name="Salamence",
            hp_min=130,
            hp_max=130,
            attack_min=160,
            attack_max=160,
            defense_min=100,
            defense_max=100,
            spa_min=120,
            spa_max=120,
            spd_min=110,
            spd_max=110,
            speed_min=140,
            speed_max=140,
            moves={"Earthquake"},
            items={"Choice Scarf"},
            abilities={"Intimidate"},
            tera_types={"Water"},
        )
        belief.add_pokemon("Salamence", pokemon_belief)

        # With all stats known for certain, uncertainty should be 0
        uncertainty = belief.get_pokemon_uncertainty("Salamence")
        assert uncertainty == 0.0

    def test_team_uncertainty(self):
        """Test average uncertainty across the team."""
        belief = OpponentBeliefState()

        # Certain Pokemon
        certain = PokemonBeliefState(
            name="Salamence",
            hp_min=130,
            hp_max=130,
            attack_min=160,
            attack_max=160,
            defense_min=100,
            defense_max=100,
            spa_min=120,
            spa_max=120,
            spd_min=110,
            spd_max=110,
            speed_min=140,
            speed_max=140,
        )
        belief.add_pokemon("Salamence", certain)

        # Uncertain Pokemon
        uncertain = PokemonBeliefState(name="Kyogre")
        belief.add_pokemon("Kyogre", uncertain)

        avg_uncertainty = belief.average_team_uncertainty()
        assert avg_uncertainty > 0
        assert avg_uncertainty < uncertain.get_uncertainty()  # Average is between them

    def test_constraint_by_item_discovery(self):
        """Test constraining beliefs when an item is discovered."""
        belief = OpponentBeliefState()
        pokemon = PokemonBeliefState(
            name="Salamence",
            items={"Choice Scarf", "Life Orb", "Assault Vest"},
        )
        belief.add_pokemon("Salamence", pokemon)

        # Discover it has Choice Scarf
        belief.update_pokemon_belief("Salamence", items={"Choice Scarf"})
        updated = belief.get_pokemon("Salamence")
        assert updated.items == {"Choice Scarf"}

    def test_constraint_by_move_discovery(self):
        """Test constraining beliefs when a move is discovered."""
        belief = OpponentBeliefState()
        pokemon = PokemonBeliefState(
            name="Salamence",
            moves={"Earthquake", "Outrage", "Protect", "Superpower"},
        )
        belief.add_pokemon("Salamence", pokemon)

        # Discover it uses Earthquake
        belief.update_pokemon_belief("Salamence", moves={"Earthquake"})
        updated = belief.get_pokemon("Salamence")
        assert updated.moves == {"Earthquake"}
