import chess
from typing import List, Tuple, Optional

# ---------- Move encoding (same as your training pipeline) ----------
PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
NUM_PROMOS = len(PROMO_PIECES)
POLICY_DIM = 64 * 64 * NUM_PROMOS  # 20,480

def encode_move_components(from_sq: int, to_sq: int, promo_piece) -> int:
    promo_idx = PROMO_PIECES.index(promo_piece)
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx

def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    promo_piece = move.promotion if move.promotion is not None else None
    return encode_move_components(from_sq, to_sq, promo_piece)

# ---------- Legality testing function ----------
def legal_moves_indices(board: chess.Board, debug: bool = False) -> List[int]:
    """
    Returns a list of legal move indices for a given chess.Board.
    
    Args:
        board (chess.Board): the board position to evaluate
        debug (bool): if True, prints extra info for debugging
    
    Returns:
        List[int]: list of legal move indices compatible with policy dimension
    """
    legal_indices = []
    
    for move in board.legal_moves:
        idx = move_to_index(move)
        legal_indices.append(idx)
    
    if debug:
        print("========== DEBUG LEGAL MOVES ==========")
        print("FEN:", board.fen())
        print("Side to move:", "White" if board.turn == chess.WHITE else "Black")
        print("Number of legal moves:", len(legal_indices))
        print("Sample moves (SAN -> index):")
        for i, move in enumerate(board.legal_moves):
            if i >= 10:  # only print first 10 moves for sanity
                break
            print(f"  {board.san(move)} -> {move_to_index(move)}")
        print("======================================")
    
    return legal_indices

# ---------- Optional helper: legality mask ----------
def legality_mask(board: chess.Board) -> List[int]:
    """
    Returns a binary mask of size POLICY_DIM where legal moves are 1, illegal are 0.
    """
    mask = [0] * POLICY_DIM
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1
    return mask

# ---------- Example usage ----------
if __name__ == "__main__":
    board = chess.Board()  # starting position
    indices = legal_moves_indices(board, debug=True)
    mask = legality_mask(board)
    print("Mask sum (number of legal moves):", sum(mask))
