#!/usr/bin/env python3
"""
Generate self-play games using the chess engine.

Runs multiple engine instances to play games against themselves,
collecting (position, policy, value) tuples for training.
"""

import argparse
import subprocess
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.utils import load_config, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Generate self-play games')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to chess engine binary')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to engine weights NPZ')
    parser.add_argument('--config', type=str, default='config/selfplay.yaml',
                        help='Path to selfplay config')
    parser.add_argument('--num-games', type=int, default=100,
                        help='Number of games to generate')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for game data')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    return parser.parse_args()


def play_single_game(engine_path, weights_path, config):
    """
    Play a single self-play game.
    
    Returns:
        Game data: positions, policies, outcome
    """
    # TODO: Implement engine communication
    # - Start engine process
    # - Load weights
    # - Play game with specified time control
    # - Collect (position, search policy, move) for each position
    # - Return game data
    
    # Placeholder
    return {
        'positions': [],
        'policies': [],
        'moves': [],
        'result': 0.0
    }


def main():
    args = parse_args()
    
    # Setup
    logger = setup_logger('selfplay')
    config = load_config(args.config)
    selfplay_config = config['selfplay']
    
    logger.info(f"Generating {args.num_games} self-play games")
    logger.info(f"Using {args.workers} parallel workers")
    logger.info(f"Engine: {args.engine}")
    logger.info(f"Weights: {args.weights}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate games in parallel
    games_completed = 0
    all_games_data = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(play_single_game, args.engine, args.weights, selfplay_config)
            for _ in range(args.num_games)
        ]
        
        for future in futures:
            try:
                game_data = future.result()
                all_games_data.append(game_data)
                games_completed += 1
                
                if games_completed % 10 == 0:
                    logger.info(f"Completed {games_completed}/{args.num_games} games")
            except Exception as e:
                logger.error(f"Game failed: {e}")
    
    logger.info(f"\nCompleted {games_completed} games")
    
    # Save games
    games_file = output_dir / 'games.json'
    with open(games_file, 'w') as f:
        json.dump(all_games_data, f)
    
    logger.info(f"Saved games to {games_file}")
    
    # TODO: Convert to training buffer format
    # See sample_to_rl_buffer.py


if __name__ == '__main__':
    main()
