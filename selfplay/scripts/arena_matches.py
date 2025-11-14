#!/usr/bin/env python3
"""
Run arena matches between two models to measure Elo difference.

Used for model selection: new model must beat old model by threshold.
"""

import argparse
import subprocess
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.utils import load_config, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Arena matches between models')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to chess engine binary')
    parser.add_argument('--model1', type=str, required=True,
                        help='Path to first model weights')
    parser.add_argument('--model2', type=str, required=True,
                        help='Path to second model weights')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--config', type=str, default='config/selfplay.yaml',
                        help='Path to config')
    return parser.parse_args()


def play_match(engine_path, weights1, weights2, num_games, config):
    """
    Play a match between two models.
    
    Returns:
        Result statistics: wins, losses, draws for model1
    """
    # TODO: Implement arena match
    # - Play num_games/2 with model1 as white
    # - Play num_games/2 with model2 as white
    # - Use specified time control
    # - Return results
    
    # Placeholder
    return {
        'model1_wins': 0,
        'model2_wins': 0,
        'draws': 0,
        'model1_score': 0.0,
        'model2_score': 0.0,
    }


def calculate_elo_difference(score1, score2, num_games):
    """
    Calculate Elo difference from match score.
    
    Args:
        score1: Score for model1 (wins + 0.5*draws)
        score2: Score for model2
        num_games: Total games played
        
    Returns:
        Elo difference (positive means model1 is stronger)
    """
    if score1 == 0:
        return -float('inf')
    if score1 == num_games:
        return float('inf')
    
    win_rate = score1 / num_games
    
    # Elo formula: Δ = 400 * log10(win_rate / (1 - win_rate))
    import math
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))
    
    return elo_diff


def main():
    args = parse_args()
    
    logger = setup_logger('arena')
    config = load_config(args.config)
    
    logger.info(f"Running arena match: {args.games} games")
    logger.info(f"Model 1: {args.model1}")
    logger.info(f"Model 2: {args.model2}")
    
    # Play match
    results = play_match(
        args.engine,
        args.model1,
        args.model2,
        args.games,
        config['selfplay']['evaluation']
    )
    
    # Report results
    logger.info("\n" + "=" * 50)
    logger.info("Arena Match Results")
    logger.info("=" * 50)
    logger.info(f"Model 1 wins: {results['model1_wins']}")
    logger.info(f"Model 2 wins: {results['model2_wins']}")
    logger.info(f"Draws: {results['draws']}")
    logger.info(f"Model 1 score: {results['model1_score']:.1f}/{args.games}")
    logger.info(f"Model 2 score: {results['model2_score']:.1f}/{args.games}")
    
    # Calculate Elo
    elo_diff = calculate_elo_difference(
        results['model1_score'],
        results['model2_score'],
        args.games
    )
    
    logger.info(f"\nEstimated Elo difference: {elo_diff:+.1f}")
    
    if elo_diff > 0:
        logger.info("→ Model 1 is stronger")
    elif elo_diff < 0:
        logger.info("→ Model 2 is stronger")
    else:
        logger.info("→ Models are equal")
    
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
