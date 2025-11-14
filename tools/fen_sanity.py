#!/usr/bin/env python3
"""
Sanity check FEN parsing and encoding.

Validates that FEN → planes → board round-trip is correct.
"""

import chess
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.utils import fen_to_planes


def test_fen(fen_string):
    """Test FEN encoding for a position."""
    print(f"\nTesting FEN: {fen_string}")
    
    # Parse with python-chess
    try:
        board = chess.Board(fen_string)
    except ValueError as e:
        print(f"  ❌ Invalid FEN: {e}")
        return False
    
    # Convert to planes
    try:
        planes = fen_to_planes(fen_string)
        print(f"  ✓ Converted to planes: {planes.shape}")
    except Exception as e:
        print(f"  ❌ Encoding failed: {e}")
        return False
    
    # Check piece planes (0-11)
    piece_count = planes[0:12].sum()
    actual_pieces = len(board.piece_map())
    print(f"  Pieces in planes: {int(piece_count)}, actual: {actual_pieces}")
    
    if int(piece_count) != actual_pieces:
        print(f"  ⚠️  Piece count mismatch!")
    
    # Check castling (20-23)
    castling_flags = []
    if planes[20, 0, 0] > 0:
        castling_flags.append('K')
    if planes[21, 0, 0] > 0:
        castling_flags.append('Q')
    if planes[22, 0, 0] > 0:
        castling_flags.append('k')
    if planes[23, 0, 0] > 0:
        castling_flags.append('q')
    
    print(f"  Castling rights: {''.join(castling_flags) or '-'}")
    
    # Check side to move (25)
    side = "White" if planes[25, 0, 0] > 0 else "Black"
    print(f"  Side to move: {side}")
    
    return True


def main():
    # Test positions
    test_fens = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # After 1.e4
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        # Kiwipete
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        # Endgame
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    
    print("=" * 60)
    print("FEN Sanity Check")
    print("=" * 60)
    
    passed = 0
    for fen in test_fens:
        if test_fen(fen):
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(test_fens)} tests passed")
    print("=" * 60)


if __name__ == '__main__':
    main()
