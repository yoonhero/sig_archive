import chess
import chess.engine
from typing import Optional, Dict, Tuple

from state import State

# Estimated ELO ranges for Stockfish skill levels
# These are approximate values based on community testing
SKILL_LEVEL_ELO_MAP: Dict[int, Tuple[int, int]] = {
    0: (1200, 1450),    # Based on master-skill-0: 1320.1
    1: (1350, 1600),    # Based on master-skill-1: 1467.6
    2: (1500, 1750),    # Based on master-skill-2: 1608.4
    3: (1600, 1900),    # Based on master-skill-3: 1742.3
    4: (1800, 2050),    # Based on master-skill-4: 1922.9
    5: (2050, 2350),    # Based on master-skill-5: 2203.7
    6: (2250, 2500),    # Based on master-skill-6: 2363.2
    7: (2350, 2650),    # Based on master-skill-7: 2499.5
    8: (2450, 2750),    # Based on master-skill-8: 2596.2
    9: (2550, 2850),    # Based on master-skill-9: 2702.8
    10: (2650, 2950),   # Based on master-skill-10: 2788.3
    11: (2700, 3000),   # Based on master-skill-11: 2855.5
    12: (2800, 3050),   # Based on master-skill-12: 2923.1
    13: (2850, 3100),   # Based on master-skill-13: 2972.9
    14: (2900, 3150),   # Based on master-skill-14: 3024.8
    15: (2950, 3200),   # Based on master-skill-15: 3069.5
    16: (3000, 3250),   # Based on master-skill-16: 3111.2
    17: (3000, 3300),   # Based on master-skill-17: 3141.3
    18: (3050, 3300),   # Based on master-skill-18: 3170.3
    19: (3050, 3350),   # Based on master-skill-19: 3191.1
    20: (3300, 3600),   # Max strength, estimation beyond table
}


# Example usage:
if __name__ == "__main__":
    from live_chess import BasicSearchAgent
    
    # Initialize game state and agent
    state = State()
    agent = BasicSearchAgent(state, max_depth=3)
    
    # Create Stockfish opponent with skill level 10 (Expert level)
    opponent = StockfishOpponent("/opt/homebrew/bin/stockfish", skill_level=4)
    
    # Get estimated ELO range
    min_elo, max_elo = opponent.get_estimated_elo()
    print(f"Opponent ELO range: {min_elo}-{max_elo}")
    
    # Play a game
    result = opponent.play_game(agent, state)
    print(f"Game result: {result}")