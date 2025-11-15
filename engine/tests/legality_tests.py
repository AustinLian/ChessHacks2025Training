import chess
import numpy as np
from typing import List, Tuple, Optional

# ---------------- MOVE ENCODING ----------------
PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
NUM_PROMOS = len(PROMO_PIECES)
POLICY_DIM = 64 * 64 * NUM_PROMOS  # 20,480

def encode_move_components(from_sq: int, to_sq: int, promo_piece) -> int:
    """Encode a move into an integer index for policy output."""
    promo_idx = PROMO_PIECES.index(promo_piece)
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx

def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move to a policy index."""
    from_sq = move.from_square
    to_sq = move.to_square
    promo_piece = move.promotion if move.promotion is not None else None
    return encode_move_components(from_sq, to_sq, promo_piece)

def index_to_move(index: int, board: Optional[chess.Board] = None) -> chess.Move:
    """
    Convert a policy index back to a chess.Move.
    If board is provided, ensures legality by returning a legal move.
    """
    promo_idx = index % NUM_PROMOS
    to_sq = (index // NUM_PROMOS) % 64
    from_sq = (index // NUM_PROMOS) // 64
    promo_piece = PROMO_PIECES[promo_idx]
    move = chess.Move(from_sq, to_sq, promotion=promo_piece)

    if board is not None and move not in board.legal_moves:
        # Defensive: return first legal move if original is illegal
        return next(iter(board.legal_moves))
    return move

# ---------------- LEGALITY HELPERS ----------------
def legal_moves_indices(board: chess.Board, debug: bool = False) -> List[int]:
    """
    Returns a list of legal move indices for the given board.
    """
    legal_indices = [move_to_index(m) for m in board.legal_moves]

    if debug:
        print("========== DEBUG LEGAL MOVES ==========")
        print("FEN:", board.fen())
        print("Side to move:", "White" if board.turn == chess.WHITE else "Black")
        print("Number of legal moves:", len(legal_indices))
        print("Sample moves (SAN -> index):")
        for i, move in enumerate(board.legal_moves):
            if i >= 10:  # only print first 10 moves
                break
            print(f"  {board.san(move)} -> {move_to_index(move)}")
        print("======================================")

    return legal_indices

def legality_mask(board: chess.Board) -> np.ndarray:
    """
    Returns a NumPy array of shape (POLICY_DIM,) where legal moves are 1, illegal are 0.
    """
    mask = np.zeros(POLICY_DIM, dtype=np.int8)
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1
    return mask

# ---------------- POLICY OUTPUT HELPERS ----------------
def filter_policy(policy_logits: np.ndarray, board: chess.Board) -> np.ndarray:
    """
    Zero out illegal moves in a policy output array.
    Returns a masked array of same shape.
    """
    mask = legality_mask(board)
    return policy_logits * mask

def sample_policy(policy_probs: np.ndarray, board: chess.Board) -> int:
    """
    Sample a legal move index from a probability distribution.
    Zeroes out illegal moves first.
    """
    masked_probs = filter_policy(policy_probs, board)
    if masked_probs.sum() == 0:
        # Fallback: pick random legal move
        legal_idxs = legal_moves_indices(board)
        return np.random.choice(legal_idxs)
    masked_probs = masked_probs / masked_probs.sum()
    return int(np.random.choice(len(masked_probs), p=masked_probs))

# ---------------- DEBUG EXAMPLE ----------------
if __name__ == "__main__":
    board = chess.Board()  # starting position
    print("Legal move indices:", legal_moves_indices(board, debug=True))
    print("Legality mask sum (number of legal moves):", legality_mask(board).sum())
    
    # Example policy
    policy = np.random.rand(POLICY_DIM)
    chosen = sample_policy(policy, board)
    print("Sampled policy move index:", chosen)
    print("Corresponding SAN:", board.san(index_to_move(chosen, board)))
