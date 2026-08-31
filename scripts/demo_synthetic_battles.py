"""Example usage of the synthetic battle generator and replayer."""

from src.training import BattleGenerator, BattleReplayer
from src.utils.config import REGULATION_SV2024_1


def main():
    """Demonstrate synthetic battle generation and replay."""
    
    # Initialize generator with a regulation
    generator = BattleGenerator(
        regulation_config=REGULATION_SV2024_1,
        seed=42,
        player_agent_type="max_damage",
        opponent_agent_type="random",
    )
    
    # Create sample teams from legal Pokemon
    pokemon_list = list(REGULATION_SV2024_1.legal_pokemon)[:6]
    team = [
        {
            "name": pokemon,
            "level": 50,
            "item": "choice-band",
            "ability": "static",
            "moves": ["earthquake", "protected"],
            "tera_type": "normal",
        }
        for pokemon in pokemon_list
    ]
    
    # Generate battles
    print("Generating battles...")
    for i in range(3):
        record = generator.generate_battle(player_team=team, opponent_team=team)
        print(f"Battle {i+1}: {record.winner} won in {len(record.trajectory)} steps")
        
        # Replay the battle
        replayer = BattleReplayer(record)
        print(f"  Trajectory length: {replayer.get_trajectory_length()}")
        
        # Iterate through steps
        step_count = 0
        for state, action, done, reward in replayer:
            step_count += 1
        print(f"  Replayed {step_count} steps")
    
    # Print statistics
    stats = generator.get_battle_statistics()
    print("\nBattle Statistics:")
    print(f"  Total battles: {stats['total_battles']}")
    print(f"  Player wins: {stats['player_wins']}")
    print(f"  Opponent wins: {stats['opponent_wins']}")
    print(f"  Average battle length: {stats['average_battle_length']:.1f} steps")


if __name__ == "__main__":
    main()
