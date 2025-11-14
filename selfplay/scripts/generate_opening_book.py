#!/usr/bin/env python3
"""
Generate opening book for self-play games.

Creates diverse starting positions to improve exploration.
"""

import argparse
import chess
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate opening book')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file for opening book')
    parser.add_argument('--num-positions', type=int, default=1000,
                        help='Number of opening positions to generate')
    parser.add_argument('--max-moves', type=int, default=8,
                        help='Maximum moves from start position')
    return parser.parse_args()


def generate_opening_position(max_moves):
    """
    Generate a random opening position by playing random legal moves.
    
    Args:
        max_moves: Maximum number of moves to play
        
    Returns:
        FEN string of resulting position
    """
    board = chess.Board()
    
    num_moves = random.randint(1, max_moves)
    
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Play random legal move
        move = random.choice(legal_moves)
        board.push(move)
    
    return board.fen()


def main():
    args = parse_args()
    
    print(f"Generating {args.num_positions} opening positions")
    print(f"Max moves from start: {args.max_moves}")
    
    # Generate positions
    positions = set()
    
    while len(positions) < args.num_positions:
        fen = generate_opening_position(args.max_moves)
        positions.add(fen)
        
        if len(positions) % 100 == 0:
            print(f"Generated {len(positions)}/{args.num_positions} unique positions")
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for fen in sorted(positions):
            f.write(fen + '\n')
    
    print(f"\nSaved opening book to {output_path}")
    print(f"Total unique positions: {len(positions)}")


if __name__ == '__main__':
    main()
