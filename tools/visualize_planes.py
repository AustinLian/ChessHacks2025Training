#!/usr/bin/env python3
"""
Visualize board plane representation.

Displays individual planes for debugging encoding.
"""

import argparse
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.utils import fen_to_planes


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize board planes')
    parser.add_argument('--fen', type=str,
                        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        help='FEN string to visualize')
    parser.add_argument('--plane', type=int, default=None,
                        help='Show specific plane (0-26)')
    return parser.parse_args()


def visualize_plane(plane, name):
    """Print a single 8x8 plane."""
    print(f"\n{name}:")
    print("  " + " ".join("abcdefgh"))
    for rank in range(7, -1, -1):
        row = plane[rank, :]
        row_str = "".join(['■' if v > 0.5 else '□' for v in row])
        print(f"{rank + 1} {' '.join(row_str)}")


def main():
    args = parse_args()
    
    print(f"FEN: {args.fen}")
    
    # Convert to planes
    planes = fen_to_planes(args.fen)
    print(f"Planes shape: {planes.shape}")
    
    plane_names = [
        # Pieces (0-11)
        "White Pawns", "White Knights", "White Bishops",
        "White Rooks", "White Queens", "White King",
        "Black Pawns", "Black Knights", "Black Bishops",
        "Black Rooks", "Black Queens", "Black King",
        # History (12-19)
        "History 1A", "History 1B", "History 2A", "History 2B",
        "History 3A", "History 3B", "History 4A", "History 4B",
        # Metadata (20-26)
        "White O-O", "White O-O-O", "Black O-O", "Black O-O-O",
        "En Passant", "Side to Move", "Move Count"
    ]
    
    if args.plane is not None:
        # Show specific plane
        if 0 <= args.plane < 27:
            visualize_plane(planes[args.plane], f"Plane {args.plane}: {plane_names[args.plane]}")
        else:
            print(f"Invalid plane index: {args.plane} (must be 0-26)")
    else:
        # Show piece planes only
        print("\n" + "=" * 60)
        print("Piece Planes")
        print("=" * 60)
        for i in range(12):
            if planes[i].sum() > 0:
                visualize_plane(planes[i], f"Plane {i}: {plane_names[i]}")
        
        # Show metadata
        print("\n" + "=" * 60)
        print("Metadata Planes")
        print("=" * 60)
        for i in range(20, 27):
            if planes[i].sum() > 0 or i in [25, 26]:
                visualize_plane(planes[i], f"Plane {i}: {plane_names[i]}")


if __name__ == '__main__':
    main()
