#!/usr/bin/env python3
"""
Convert PGN files to dataset for supervised training.

Extracts positions from chess games and converts them to
(board_planes, move_index, game_result) format.
"""

import argparse
import numpy as np
import chess
import chess.pgn
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.scripts.utils.fen_encoding import fen_to_planes
from utils.move_encoding import move_to_index


def parse_args():
    parser = argparse.ArgumentParser(description='Convert PGN to training dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Input PGN file(s) (glob pattern)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NPZ file path')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Maximum number of games to process')
    parser.add_argument('--min-elo', type=int, default=2000,
                        help='Minimum Elo rating for players')
    return parser.parse_args()


def extract_positions_from_game(game):
    """
    Extract training positions from a single game.
    
    Returns:
        List of (planes, move_index, result) tuples
    """
    positions = []
    board = game.board()
    
    # Get game result
    result = game.headers.get('Result', '*')
    if result == '1-0':
        result_value = 1.0
    elif result == '0-1':
        result_value = -1.0
    elif result == '1/2-1/2':
        result_value = 0.0
    else:
        return []  # Skip games without result
    
    # Walk through moves
    for move in game.mainline_moves():
        # Convert position to planes
        planes = fen_to_planes(board.fen())
        
        # Get move index
        move_idx = move_to_index(move)
        
        # Adjust result based on side to move
        if board.turn == chess.WHITE:
            value = result_value
        else:
            value = -result_value
        
        positions.append((planes, move_idx, value))
        
        # Make move
        board.push(move)
    
    return positions


def main():
    args = parse_args()
    
    # Find PGN files
    input_path = Path(args.input)
    if input_path.is_file():
        pgn_files = [input_path]
    else:
        pgn_files = list(input_path.parent.glob(input_path.name))
    
    print(f"Found {len(pgn_files)} PGN file(s)")
    
    # Collect positions
    all_planes = []
    all_move_indices = []
    all_results = []
    
    games_processed = 0
    positions_collected = 0
    
    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}...")
        
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Check player ratings if available
                white_elo = game.headers.get('WhiteElo', '0')
                black_elo = game.headers.get('BlackElo', '0')
                try:
                    if int(white_elo) < args.min_elo or int(black_elo) < args.min_elo:
                        continue
                except ValueError:
                    pass  # Skip if Elo not available
                
                # Extract positions
                positions = extract_positions_from_game(game)
                
                for planes, move_idx, result in positions:
                    all_planes.append(planes)
                    all_move_indices.append(move_idx)
                    all_results.append(result)
                
                games_processed += 1
                positions_collected += len(positions)
                
                if args.max_games and games_processed >= args.max_games:
                    break
        
        if args.max_games and games_processed >= args.max_games:
            break
    
    print(f"\nProcessed {games_processed} games")
    print(f"Collected {positions_collected} positions")
    
    if positions_collected == 0:
        print("No positions collected!")
        return
    
    # Convert to numpy arrays
    planes_array = np.stack(all_planes, axis=0)
    move_indices_array = np.array(all_move_indices, dtype=np.int32)
    results_array = np.array(all_results, dtype=np.float32)
    
    print(f"\nDataset shapes:")
    print(f"  Planes: {planes_array.shape}")
    print(f"  Move indices: {move_indices_array.shape}")
    print(f"  Results: {results_array.shape}")
    
    # Save to NPZ
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        planes=planes_array,
        move_indices=move_indices_array,
        results=results_array
    )
    
    print(f"\nSaved dataset to {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == '__main__':
    main()
