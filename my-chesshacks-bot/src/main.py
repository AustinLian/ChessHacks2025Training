from .utils import chess_manager, GameContext
from chess import Move
import chess
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

# Point this to the NEW best.pt from the patched trainer above.
MODEL_PATH = Path(
    r"C:\Users\ethan\Downloads\ChessHacks\ChessHacks2025\checkpoints_delta_resnet20\best.pt"
)

NUM_PLANES = 18
NUM_PROMOS = 5   # [None, Q, R, B, N]
POLICY_DIM = 64 * 64 * NUM_PROMOS
CP_SCALE = 200.0

MATE_SCORE = 100000.0
TIME_LIMIT_SECONDS = 0.25   # per-move budget (tune)
MAX_SEARCH_DEPTH = 10       # root move + depth-1 search

RESULT_TO_CP_SCALE = 400.0  # how much game-result head influences eval

ENGINE = None   # set at import time


# ============================================================
# BOARD → PLANES (must match training)
# ============================================================

PIECE_PLANES = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Same encoding as your training: (18, 8, 8).
    """
    P = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        p_idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        P[p_idx, r, f] = 1.0

    # side to move
    if board.turn == chess.WHITE:
        P[12, :, :] = 1.0
    else:
        P[12, :, :] = 0.0

    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        P[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        P[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        P[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        P[16, :, :] = 1.0

    # en-passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        P[17, :, file] = 1.0

    return P


def move_to_index(move: chess.Move) -> int:
    """
    Same policy index scheme as training:
      (from * 64 + to) * 5 + promo_idx
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion if move.promotion is not None else None
    promo_idx = PROMO_PIECES.index(promo)  # 0..4
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx


# ============================================================
# MODEL DEFINITION (must match train.py)
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h + self.short(x)
        return self.relu(h)


class ResNet20_PolicyDelta(nn.Module):
    def __init__(self, planes, policy_dim, cp_scale):
        super().__init__()
        self.cp_scale = cp_scale

        def make_layer(in_c, out_c, blocks):
            layers = [BasicBlock(in_c, out_c)]
            for _ in range(blocks - 1):
                layers.append(BasicBlock(out_c, out_c))
            return nn.Sequential(*layers)

        self.l1 = make_layer(planes, 128, 3)
        self.l2 = make_layer(128, 256, 3)
        self.l3 = make_layer(256, 256, 3)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)

        # Heads: must match trainer
        self.policy_head = nn.Linear(256, policy_dim)
        self.delta_head = nn.Linear(256, 1)
        self.cp_head = nn.Linear(256, 1)
        self.result_head = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc(x))

        policy_logits = self.policy_head(x)

        delta_scaled = self.delta_head(x).squeeze(-1)
        cp_scaled = self.cp_head(x).squeeze(-1)

        delta_real = delta_scaled * self.cp_scale
        cp_real = cp_scaled * self.cp_scale

        result_raw = self.result_head(x).squeeze(-1)
        result = torch.tanh(result_raw)  # [-1,1]

        return policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result


# ============================================================
# NN-GUIDED SEARCH ENGINE
# ============================================================

class NNEvalEngine:
    """
    Alpha-beta search guided by:
      - cp_head + result_head for evaluation
      - policy logits for move ordering
    At the root, we blend:
      - NN policy distribution (over legal moves)
      - shallow search evals over legal moves
    """

    def __init__(self):
        print(f"[ENGINE] Initializing NN engine from {MODEL_PATH!r}")
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet20_PolicyDelta(
            planes=NUM_PLANES,
            policy_dim=POLICY_DIM,
            cp_scale=CP_SCALE,
        ).to(self.device)

        state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        self.time_limit = TIME_LIMIT_SECONDS
        self.max_depth = MAX_SEARCH_DEPTH
        self.time_up = False
        self.tt = {}
        self.nn_cache = {}

        print(f"[ENGINE] NN engine ready on {self.device}")

    @torch.no_grad()
    def _policy_and_value(self, board: chess.Board):
        """
        Returns:
          - policy_logits: (POLICY_DIM,)
          - eval_cp: float (combined cp + result)
        Uses a small cache keyed by FEN.
        """
        fen = board.fen()
        cached = self.nn_cache.get((fen, board.turn))
        if cached is not None:
            return cached

        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        (
            policy_logits,
            _delta_real,
            _delta_scaled,
            cp_real,
            _cp_scaled,
            result,
        ) = self.model(x)

        policy_logits = policy_logits[0].detach().cpu()
        cp_eval = float(cp_real[0].item())      # absolute cp_before
        res_eval = float(result[0].item())      # in [-1,1]

        eval_cp = cp_eval + res_eval * RESULT_TO_CP_SCALE
        out = (policy_logits, eval_cp)
        self.nn_cache[(fen, board.turn)] = out
        return out

    def _evaluate(self, board: chess.Board) -> float:
        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE
            return 0.0
        _, eval_cp = self._policy_and_value(board)
        return max(-MATE_SCORE, min(MATE_SCORE, eval_cp))

    def _policy_scores_for_moves(self, board: chess.Board, legal_moves):
        policy_logits, _ = self._policy_and_value(board)
        scores = []
        for mv in legal_moves:
            idx = move_to_index(mv)
            s = float(policy_logits[idx].item()) if 0 <= idx < POLICY_DIM else 0.0
            scores.append(s)
        return scores

    def _time_exceeded(self, start_time: float) -> bool:
        if self.time_limit <= 0:
            return False
        if time.perf_counter() - start_time >= self.time_limit:
            self.time_up = True
            return True
        return False

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, start_time: float) -> float:
        if self._time_exceeded(start_time):
            return 0.0

        key = (board.fen(), depth, board.turn)
        if key in self.tt:
            return self.tt[key]

        if depth == 0 or board.is_game_over():
            val = self._evaluate(board)
            self.tt[key] = val
            return val

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            val = self._evaluate(board)
            self.tt[key] = val
            return val

        scores = self._policy_scores_for_moves(board, legal_moves)
        ordered_moves = [
            mv for _, mv in sorted(
                zip(scores, legal_moves),
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        best_val = -float("inf")

        for mv in ordered_moves:
            if self.time_up:
                break
            board.push(mv)
            val = -self._search(board, depth - 1, -beta, -alpha, start_time)
            board.pop()

            if self.time_up:
                break

            if val > best_val:
                best_val = val
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        self.tt[key] = best_val
        return best_val

    def select_move(self, board: chess.Board):
        """
        Root move selection:
          - Get NN policy over legal moves.
          - For each legal move, run shallow search (depth = max_depth - 1).
          - Convert both to distributions and blend them.
          - Choose argmax of blended distribution (deterministic).
        """
        start_time = time.perf_counter()
        self.time_up = False
        self.tt.clear()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        # --------------------------------------
        # 1) NN policy over legal moves (prior)
        # --------------------------------------
        policy_logits, _ = self._policy_and_value(board)

        policy_scores = []
        for mv in legal_moves:
            idx = move_to_index(mv)
            if 0 <= idx < POLICY_DIM:
                policy_scores.append(float(policy_logits[idx].item()))
            else:
                policy_scores.append(-1e9)

        max_p = max(policy_scores)
        exp_p = [np.exp(s - max_p) for s in policy_scores]
        Zp = sum(exp_p)
        if Zp <= 0:
            prior_probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            prior_probs = [x / Zp for x in exp_p]

        # --------------------------------------
        # 2) Shallow search evals for each move
        # --------------------------------------
        search_vals = []
        search_depth = max(1, self.max_depth - 1)

        for mv in legal_moves:
            if self._time_exceeded(start_time):
                break
            board.push(mv)
            val = -self._search(board, depth=search_depth, alpha=-float("inf"), beta=float("inf"), start_time=start_time)
            board.pop()
            search_vals.append(val)

        # If we ran out of time before finishing all moves,
        # just fall back to pure policy argmax + prior_probs.
        if len(search_vals) < len(legal_moves):
            best_idx = int(np.argmax(policy_scores))
            best_move = legal_moves[best_idx]
            probs_dict = {mv: prior_probs[i] for i, mv in enumerate(legal_moves)}
            return best_move, probs_dict

        # --------------------------------------
        # 3) Convert search evals to distribution
        # --------------------------------------
        max_s = max(search_vals)
        cp_temp = 120.0  # centipawn temperature (tune)
        exp_s = [np.exp((v - max_s) / cp_temp) for v in search_vals]
        Zs = sum(exp_s)
        if Zs <= 0:
            search_probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            search_probs = [x / Zs for x in exp_s]

        # --------------------------------------
        # 4) Blend policy + search distributions
        # --------------------------------------
        Wp = 0.3   # weight for policy
        Ws = 0.7   # weight for search

        combined = [
            Wp * prior_probs[i] + Ws * search_probs[i]
            for i in range(len(legal_moves))
        ]

        Zc = sum(combined)
        if Zc <= 0:
            final_probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            final_probs = [x / Zc for x in combined]

        # --------------------------------------
        # 5) Final move = argmax of blended probs
        # --------------------------------------
        best_idx = int(np.argmax(final_probs))
        best_move = legal_moves[best_idx]

        probs_dict = {mv: final_probs[i] for i, mv in enumerate(legal_moves)}
        return best_move, probs_dict

    def reset(self):
        self.tt.clear()
        self.nn_cache.clear()
        self.time_up = False


# ============================================================
# ENGINE INIT
# ============================================================

try:
    ENGINE = NNEvalEngine()
except Exception as e:
    ENGINE = None
    print(f"[ENGINE] Failed to initialize NN engine, falling back to random. Error: {e}")


# ============================================================
# Submission platform hooks
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.01)

    board = ctx.board
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Fallback: random policy if ENGINE is None
    if ENGINE is None:
        print("NO ENGINE, using random.")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    best_move, probs = ENGINE.select_move(board)
    print("HELP")
    if best_move is None:

        print("ENGINE returned no move, using random fallback.")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # Ensure probabilities are defined over exactly current legal moves
    probs_filtered = {mv: probs.get(mv, 0.0) for mv in legal_moves}
    Z = sum(probs_filtered.values())
    if Z > 0:
        probs_filtered = {mv: p / Z for mv, p in probs_filtered.items()}
    else:
        p = 1.0 / len(legal_moves)
        probs_filtered = {mv: p for mv in legal_moves}

    ctx.logProbabilities(probs_filtered)
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    if ENGINE is not None:
        ENGINE.reset()
