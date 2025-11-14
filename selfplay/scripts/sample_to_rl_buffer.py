#!/usr/bin/env python3
"""
Convert self-play games to RL training buffer.

Processes game data into (planes, policy, value) format.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.utils import fen_to_planes, setup_logger
from training.datasets import RollingBuffer


def parse_args():
    parser = argparse.ArgumentParser(description='Convert games to RL buffer')
    parser.add_argument('--games', type=str, required=True,
                        help='Path to games JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NPZ buffer file')
    parser.add_argument('--buffer-size', type=int, default=500000,
                        help='Maximum buffer size')
    return parser.parse_args()


def process_game(game_data):
    """
    Convert game data to training samples.
    
    Returns:
        List of (planes, policy, value) tuples
    """
    positions = game_data['positions']
    policies = game_data['policies']
    result = game_data['result']
    
    samples = []
    
    for i, (fen, policy) in enumerate(zip(positions, policies)):
        # Convert FEN to planes
        planes = fen_to_planes(fen)
        
        # Flip value based on side to move
        # (result is from white's perspective)
        value = result if (i % 2 == 0) else -result
        
        samples.append((planes, policy, value))
    
    return samples


def main():
    args = parse_args()
    
    logger = setup_logger('sample_to_buffer')
    
    # Load games
    logger.info(f"Loading games from {args.games}")
    with open(args.games) as f:
        games = json.load(f)
    
    logger.info(f"Loaded {len(games)} games")
    
    # Create buffer
    buffer = RollingBuffer(max_size=args.buffer_size)
    
    # Process games
    total_positions = 0
    for i, game in enumerate(games):
        samples = process_game(game)
        
        # Separate into components
        game_planes = [s[0] for s in samples]
        game_policies = [s[1] for s in samples]
        game_values = [s[2] for s in samples]
        
        buffer.add_game(game_planes, game_policies, game_values)
        total_positions += len(samples)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(games)} games, "
                       f"{total_positions} positions")
    
    logger.info(f"\nTotal positions collected: {total_positions}")
    logger.info(f"Buffer size: {len(buffer)}")
    
    # Save buffer
    buffer.save(args.output)
    logger.info(f"Saved buffer to {args.output}")


if __name__ == '__main__':
    main()
