import tkinter as tk
from tkinter import messagebox
import random
import numpy as np
import time
import math
import copy
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#################################################################################
# BLOCK 1 - Global variables and constants

# Initialize empty 8x8 grid (0 = empty, 1 = Black/Human, 2 = White/AI)
Grid = np.zeros((8, 8), dtype=int)

# Positional weight map for the 8x8 board.
# Corners are the highest-value cells; X-squares (diagonally adjacent to corners)
# are the most dangerous to occupy since they hand corners to the opponent.
# Used by EvaluateGrid (Block 3) - defined here as a shared constant.
POSITION_MAP = np.array([
    [ 100, -20,  10,   5,   5,  10, -20,  100],
    [ -20, -40,  -5,  -5,  -5,  -5, -40,  -20],
    [  10,  -5,   3,   1,   1,   3,  -5,   10],
    [   5,  -5,   1,   0,   0,   1,  -5,    5],
    [   5,  -5,   1,   0,   0,   1,  -5,    5],
    [  10,  -5,   3,   1,   1,   3,  -5,   10],
    [ -20, -40,  -5,  -5,  -5,  -5, -40,  -20],
    [ 100, -20,  10,   5,   5,  10, -20,  100],
], dtype=int)

# Scores and game state
score_h  = 0        # Human (Black) score
score_ia = 0        # AI    (White) score

grid_color = "dark green"   # Default board colour
StartMatch = False          # True while waiting for click to start new match

# current_turn: 1 = Human (Black), 2 = AI (White)
current_turn = 1

# PLAYER TYPES: "1" (Random) "2" (Heuristic) "3" (Minimax) "4" (Minimax AB)
#               "5" (MCTS)   "6" (Descent)
# CNN variants: "1c" "2c" "3c" "4c" "5c" "6c"  (same strategy, CNN eval)
player2_type = "1"  # Default - replaced at runtime by __main__

# Minimax search depth (full exhaustive tree, no alpha-beta pruning)
MINIMAX_DEPTH = 4


#################################################################################
# BLOCK 2 - Game logic

# All 8 directions to scan (row_delta, col_delta)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]


def InitBoard():
    """Reset the grid and place the four starting discs in the centre."""
    global Grid
    Grid = np.zeros((8, 8), dtype=int)
    Grid[3][3] = 2  # White
    Grid[3][4] = 1  # Black
    Grid[4][3] = 1  # Black
    Grid[4][4] = 2  # White


def GetFlippedDiscs(row, col, player):
    """
    Return the list of (r, c) discs that would be flipped if `player`
    places a disc at (row, col).  Returns an empty list if the move is
    illegal (cell occupied or no disc would be flipped).
    """
    if Grid[row][col] != 0:
        return []

    opponent = 3 - player  # 1 <-> 2
    flipped  = []

    for dr, dc in DIRECTIONS:
        r, c    = row + dr, col + dc
        line    = []

        # Walk in this direction while we see opponent discs
        while 0 <= r < 8 and 0 <= c < 8 and Grid[r][c] == opponent:
            line.append((r, c))
            r += dr
            c += dc

        # The line is valid only if it ends on one of the player's own discs
        if line and 0 <= r < 8 and 0 <= c < 8 and Grid[r][c] == player:
            flipped.extend(line)

    return flipped


def IsValidMove(row, col, player):
    """Return True if placing `player`'s disc at (row, col) is a legal move."""
    return len(GetFlippedDiscs(row, col, player)) > 0


def FlipDiscs(row, col, player, flipped):
    """
    Place `player`'s disc at (row, col) and flip all discs in `flipped`.
    `flipped` must be the list returned by GetFlippedDiscs for the same move.
    """
    Grid[row][col] = player
    for r, c in flipped:
        Grid[r][c] = player


def GetPossibleMoves_fast(player):
    """
    Return legal moves for `player`, scanning only empty cells that have
    at least one adjacent opponent disc (early-exit filter before the full
    direction walk).  Equivalent result to GetPossibleMoves.
    """
    opponent = 3 - player
    moves    = []
    for row, col in np.argwhere(Grid == 0):
        row = int(row); col = int(col)
        # Quick adjacency pre-filter: skip cells with no opponent neighbour
        has_opponent_neighbour = False
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and Grid[nr][nc] == opponent:
                has_opponent_neighbour = True
                break
        if not has_opponent_neighbour:
            continue
        # Full validity check: at least one direction must bracket opponent discs
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            line_len = 0
            while 0 <= r < 8 and 0 <= c < 8 and Grid[r][c] == opponent:
                line_len += 1
                r += dr; c += dc
            if line_len > 0 and 0 <= r < 8 and 0 <= c < 8 and Grid[r][c] == player:
                moves.append((row, col))
                break
    return moves


def WinnerCheck():
    """
    Determine the winner once the game has ended.
    Returns 1 (Black), 2 (White), or 4 (draw).
    """
    black = int(np.sum(Grid == 1))
    white = int(np.sum(Grid == 2))
    if   black > white: return 1
    elif white > black: return 2
    else:               return 4


def CheckGameState():
    """
    Check the current state of the game.
    Returns:
        0  - game in progress (at least one player can move)
        1  - Black wins
        2  - White wins
        4  - draw
    Standard Othello rule: if the current player has no moves, the turn passes
    to the opponent.  The game ends only when *neither* player can move.
    """
    if GetPossibleMoves_fast(1) or GetPossibleMoves_fast(2):
        return 0    # At least one side can still play
    return WinnerCheck()


def EndMatch(winner):
    """Update scores and signal that a click should start a new match."""
    global score_h, score_ia, grid_color, StartMatch

    if winner == 1:
        score_h  += 1
        grid_color = "dark green"
    elif winner == 2:
        score_ia += 1
        grid_color = "dark green"
    elif winner == 4:
        grid_color = "dark green"

    StartMatch = True


#################################################################################
# BLOCK 3a - CNN evaluation backend
#
# This block handles:
#   - Definition of OthelloCNN (must match the architecture in OthelloCNN_Train.py
#     exactly so that saved weights can be loaded without re-training).
#   - Loading and caching the model from a .pth file (load once, reuse forever).
#   - EvaluateGridCNN(): drop-in replacement for EvaluateGrid() that uses
#     the trained network to score any board position.
#
# The CNN predicts the final disc margin from Black's perspective (positive =
# Black leads).  We convert that to a signed advantage score for p_id the same
# way EvaluateGrid does: positive = good for p_id.
#
# All other AI logic (Minimax, Alpha-Beta, MCTS, Descent) is unchanged - they
# simply call _eval_fn(p_id, opp_id) which is dispatched to either
# EvaluateGrid or EvaluateGridCNN depending on the user's choice.
# ---------------------------------------------------------------------------

# Path to the trained weights file - user should set this before running.
CNN_MODEL_PATH = os.path.join(ROOT, "models", "othello_cnn_v2.pth")  # Place next to this script, or set full path

# These must match OthelloCNN_Train.py exactly
_CNN_CHANNELS = 64
_CNN_N_BLOCKS = 4

# Module-level cache: loaded on first use, then reused
_cnn_model  = None      # Loaded torch.nn.Module (eval mode)
_cnn_device = None      # "cpu" or "cuda"
_cnn_available = None   # True / False / None (not yet tested)


def _try_import_torch():
    """Return True if PyTorch is importable, False otherwise."""
    try:
        import torch
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# OthelloCNN definition (copy of the training architecture)
# Imported here so Othello_v7.py is self-contained for inference.
# ---------------------------------------------------------------------------

def _build_cnn_model():
    """Build and return an un-initialised OthelloCNN instance."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F_nn

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(channels)
        def forward(self, x):
            residual = x
            out = F_nn.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F_nn.relu(out + residual)

    class OthelloCNN(nn.Module):
        def __init__(self, channels=_CNN_CHANNELS, n_blocks=_CNN_N_BLOCKS):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
            # Head must match OthelloCNN_Train.py exactly (no Dropout there).
            # State-dict keys: head.2 = Linear(ch,32), head.3 = ReLU, head.4 = Linear(32,1)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),   # head.0
                nn.Flatten(),                    # head.1
                nn.Linear(channels, 32),         # head.2
                nn.ReLU(inplace=True),           # head.3
                nn.Linear(32, 1),                # head.4
            )
        def forward(self, x):
            x = self.stem(x)
            x = self.body(x)
            x = self.head(x)
            return x.squeeze(1)

    return OthelloCNN()


def LoadCNN(model_path=CNN_MODEL_PATH):
    """
    Load the trained CNN weights from `model_path`.
    Caches the model in _cnn_model so subsequent calls are instant.
    Returns True on success, False if PyTorch is missing or weights not found.
    """
    global _cnn_model, _cnn_device, _cnn_available

    if _cnn_available is True:
        return True     # Already loaded
    if _cnn_available is False:
        return False    # Previously failed; don't retry

    if not _try_import_torch():
        print("[CNN] PyTorch not installed — CNN evaluation unavailable.")
        _cnn_available = False
        return False

    import torch

    if not os.path.isfile(model_path):
        print(f"[CNN] Weights not found at '{model_path}'. "
              "Download trained cnn othello_cnn_v2.pth")
        _cnn_available = False
        return False

    try:
        _cnn_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _build_cnn_model()
        state = torch.load(model_path, map_location=_cnn_device)
        model.load_state_dict(state)
        model.to(_cnn_device)
        model.eval()
        _cnn_model     = model
        _cnn_available = True
        print(f"[CNN] Model loaded from '{model_path}' (device: {_cnn_device})")
        return True
    except Exception as exc:
        print(f"[CNN] Failed to load model: {exc}")
        _cnn_available = False
        return False


def _board_to_tensor_local(p_id):
    """
    Encode the current global Grid as a (1, 3, 8, 8) float32 torch.Tensor
    for inference.  Always encodes from Black's (player 1) perspective:
      channel 0 = Black discs
      channel 1 = White discs
      channel 2 = 1.0 everywhere if it is Black's (p_id=1) turn; else 0.0
    """
    import torch
    t = np.zeros((3, 8, 8), dtype=np.float32)
    t[0] = (Grid == 1).astype(np.float32)
    t[1] = (Grid == 2).astype(np.float32)
    if p_id == 1:
        t[2, :, :] = 1.0
    return torch.tensor(t, dtype=torch.float32).unsqueeze(0)   # (1,3,8,8)


def EvaluateGridCNN(p_id, opp_id):
    """
    CNN-based board evaluation function - drop-in replacement for EvaluateGrid.

    The network predicts the final disc margin from Black's perspective.
    We convert that to a score from p_id's perspective so the sign convention
    matches EvaluateGrid: positive means p_id is winning.

    Terminal states are still handled by EvaluateGrid (exact, fast).
    Falls back to EvaluateGrid if the CNN is not available.
    """
    # Terminal states: use exact heuristic (no network call needed)
    state = CheckGameState()
    if state != 0:
        return EvaluateGrid_v2(p_id, opp_id)

    if not _cnn_available:
        # Graceful fallback so the game does not crash
        return EvaluateGrid_v2(p_id, opp_id)

    import torch
    with torch.no_grad():
        x      = _board_to_tensor_local(p_id).to(_cnn_device)
        margin = _cnn_model(x).item()   # Predicted Black - White disc count

    # margin is from Black's perspective.
    # If p_id == 1 (Black): score = margin
    # If p_id == 2 (White): score = -margin  (flip sign)
    return margin if p_id == 1 else -margin


# Dispatcher: returns the correct evaluation function based on use_cnn flag
def _get_eval_fn(use_cnn):
    """Return EvaluateGridCNN if use_cnn and model is loaded, else EvaluateGrid."""
    if use_cnn and LoadCNN():
        return EvaluateGridCNN
    return EvaluateGrid_v2


#################################################################################
# BLOCK 3b - AI strategies
#
# Every strategy now accepts a `use_cnn=False` keyword argument.
# When use_cnn=True the internal evaluation calls are routed through
# EvaluateGridCNN instead of EvaluateGrid.  All other logic is unchanged.
# ---------------------------------------------------------------------------

# Component weights for EvaluateGrid (heuristic path)
_EG_W_POSITION  = 3
_EG_W_MOBILITY  = 5
_EG_W_FRONTIER  = 2
_EG_W_DISCS     = 1
_EG_LATE_THRESH = 48

# ---------------------------------------------------------------------------
# Weights for GetMoveScore (Heuristic AI)
# ---------------------------------------------------------------------------
_W_POSITION = 2
_W_FLIPS    = 3
_W_MOBILITY = 2


def EvaluateGrid_v2(p_id, opp_id):
    """
    Fast heuristic evaluation — identical 4-component formula to EvaluateGrid
    but avoids the expensive Python frontier loop and skips CheckGameState
    except when the board is actually full or a player is eliminated.
 
    Components (weights unchanged):
      1. Positional score  (POSITION_MAP; boolean-mask indexing)
      2. Mobility          (legal move count difference)
      3. Frontier penalty  (vectorised 8-neighbour dilation via np.roll)
      4. Disc difference   (late-game scaling at >= 48 total discs)
    """
    # Quick disc counts (needed for terminal check and disc-diff component)
    disc_p   = int(np.sum(Grid == p_id))
    disc_opp = int(np.sum(Grid == opp_id))
    total    = disc_p + disc_opp
 
    # Terminal detection: only call CheckGameState when strictly necessary
    # (board full, or a player has been wiped out)
    if total == 64 or disc_p == 0 or disc_opp == 0:
        state = CheckGameState()
        if state == p_id:   return  500
        if state == opp_id: return -500
        if state == 4:      return    0
 
    # 1. Positional score – boolean mask avoids element-wise multiply
    positional = (int(np.sum(POSITION_MAP[Grid == p_id]))
                  - int(np.sum(POSITION_MAP[Grid == opp_id])))
 
    # 2. Mobility
    mob_p   = len(GetPossibleMoves_fast(p_id))
    mob_opp = len(GetPossibleMoves_fast(opp_id))
    mobility = mob_p - mob_opp
 
    # 3. Frontier – vectorised 8-neighbour dilation
    #    adj_empty[r,c] is True if any neighbour of (r,c) is an empty cell.
    #    np.roll wraps at the board boundary; on an 8×8 board the resulting
    #    1-cell artefacts are negligible for a heuristic evaluation.
    empty     = (Grid == 0)
    adj_empty = (
        np.roll(empty,  1, axis=0) | np.roll(empty, -1, axis=0) |
        np.roll(empty,  1, axis=1) | np.roll(empty, -1, axis=1) |
        np.roll(np.roll(empty,  1, axis=0),  1, axis=1) |
        np.roll(np.roll(empty,  1, axis=0), -1, axis=1) |
        np.roll(np.roll(empty, -1, axis=0),  1, axis=1) |
        np.roll(np.roll(empty, -1, axis=0), -1, axis=1)
    )
    frontier_p   = int(np.sum((Grid == p_id)   & adj_empty))
    frontier_opp = int(np.sum((Grid == opp_id) & adj_empty))
    frontier = frontier_opp - frontier_p     # fewer frontier discs = better
 
    # 4. Disc difference (up-weighted late game)
    disc_weight = _EG_W_DISCS * (3 if total >= _EG_LATE_THRESH else 1)
    disc_diff   = disc_p - disc_opp
 
    return (
        _EG_W_POSITION * positional
        + _EG_W_MOBILITY * mobility
        + _EG_W_FRONTIER * frontier
        + disc_weight    * disc_diff
    )


# ---------------------------------------------------------------------------
# 1. RANDOM AI
# ---------------------------------------------------------------------------

def PlayAIRandom(player_id, use_cnn=False):
    """Pick a random legal move for `player_id`. use_cnn is ignored (no eval needed)."""
    moves = GetPossibleMoves_fast(player_id)
    if not moves:
        return
    row, col = random.choice(moves)
    flipped  = GetFlippedDiscs(row, col, player_id)
    FlipDiscs(row, col, player_id, flipped)


# ---------------------------------------------------------------------------
# 2. HEURISTIC AI
# ---------------------------------------------------------------------------

def GetMoveScore(row, col, player):
    """
    Return a raw heuristic score for placing `player`'s disc at (row, col).
    Combines positional value, flip count, and opponent mobility.
    """
    flipped = GetFlippedDiscs(row, col, player)
    if not flipped:
        return 0

    opponent  = 3 - player
    pos_score = int(POSITION_MAP[row][col])
    flip_score = len(flipped)

    FlipDiscs(row, col, player, flipped)
    opp_mobility = len(GetPossibleMoves_fast(opponent))
    Grid[row][col] = 0
    for r, c in flipped:
        Grid[r][c] = opponent

    return _W_POSITION * pos_score + _W_FLIPS * flip_score - _W_MOBILITY * opp_mobility


def GetScoreForPosition(row, col, player):
    """
    Strategic priority score for a move (100 win / 50 block / 30 corner / general).
    Used by PlayAIHeuristic.
    """
    opponent = 3 - player
    flipped  = GetFlippedDiscs(row, col, player)
    if not flipped:
        return 0

    FlipDiscs(row, col, player, flipped)
    state_after = CheckGameState()
    Grid[row][col] = 0
    for r, c in flipped:
        Grid[r][c] = opponent

    if state_after == player:
        return 100

    opp_flipped = GetFlippedDiscs(row, col, opponent)
    if opp_flipped:
        FlipDiscs(row, col, opponent, opp_flipped)
        opp_state = CheckGameState()
        Grid[row][col] = 0
        for r, c in opp_flipped:
            Grid[r][c] = player
        if opp_state == opponent:
            return 50

    if POSITION_MAP[row][col] == 100:
        return 30

    return max(0, GetMoveScore(row, col, player))


def PlayAIHeuristic(player_id, use_cnn=False):
    """
    Heuristic AI: scores every legal move and plays the best one.
    When use_cnn=True the CNN evaluation is used to score the resulting
    board state instead of the static GetScoreForPosition ladder.
    """
    moves = GetPossibleMoves_fast(player_id)
    if not moves:
        return

    opp_id  = 3 - player_id
    eval_fn = _get_eval_fn(use_cnn)

    if use_cnn and LoadCNN():
        # CNN path: apply each move, score the resulting board, undo
        best_score = -float('inf')
        best_moves = []
        for row, col in moves:
            flipped = GetFlippedDiscs(row, col, player_id)
            FlipDiscs(row, col, player_id, flipped)
            score = eval_fn(player_id, opp_id)
            Grid[row][col] = 0
            for r, c in flipped:
                Grid[r][c] = opp_id
            if score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))
    else:
        # Heuristic path: use GetScoreForPosition (unchanged from v6)
        best_score = -float('inf')
        best_moves = []
        for row, col in moves:
            score = GetScoreForPosition(row, col, player_id)
            if score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))

    row, col = random.choice(best_moves)
    flipped  = GetFlippedDiscs(row, col, player_id)
    FlipDiscs(row, col, player_id, flipped)


# ---------------------------------------------------------------------------
# 3. MINIMAX AI
# ---------------------------------------------------------------------------

# Use POSITION_MAP values to prioritise corners and high-value cells.
# Explored first → tighter alpha/beta window → more pruning.
_MOVE_ORDER = POSITION_MAP   # same array; alias for clarity
 
def _sort_moves(moves):
    """
    Sort legal moves by positional value, highest first.
    Better moves explored first increases alpha-beta cutoff frequency.
    """
    return sorted(moves, key=lambda m: _MOVE_ORDER[m[0]][m[1]], reverse=True)

def Minimax(depth, is_maximizing, p_id, opp_id, eval_fn):
    """
    Recursive minimax search (no pruning).
    eval_fn is either EvaluateGrid or EvaluateGridCNN — injected by the caller.
    """
    state = CheckGameState()
    if depth == 0 or state != 0:
        return eval_fn(p_id, opp_id)

    current_player  = p_id  if is_maximizing else opp_id
    opponent_player = opp_id if is_maximizing else p_id
    possible_moves  = GetPossibleMoves_fast(current_player)

    if not possible_moves:
        if not GetPossibleMoves_fast(opponent_player):
            return eval_fn(p_id, opp_id)
        return Minimax(depth, not is_maximizing, p_id, opp_id, eval_fn)

    if is_maximizing:
        best_val = -float('inf')
        for move in possible_moves:
            row, col = move
            flipped  = GetFlippedDiscs(row, col, current_player)
            FlipDiscs(row, col, current_player, flipped)
            value    = Minimax(depth - 1, False, p_id, opp_id, eval_fn)
            Grid[row][col] = 0
            for r, c in flipped:
                Grid[r][c] = opponent_player
            best_val = max(best_val, value)
        return best_val
    else:
        best_val = float('inf')
        for move in possible_moves:
            row, col = move
            flipped  = GetFlippedDiscs(row, col, current_player)
            FlipDiscs(row, col, current_player, flipped)
            value    = Minimax(depth - 1, True, p_id, opp_id, eval_fn)
            Grid[row][col] = 0
            for r, c in flipped:
                Grid[r][c] = opponent_player
            best_val = min(best_val, value)
        return best_val


def PlayAIMinimax(player_id, depth=MINIMAX_DEPTH, use_cnn=False):
    """
    Entry point for Minimax AI.
    use_cnn=True routes leaf evaluation through EvaluateGridCNN.
    """
    opp_id         = 3 - player_id
    possible_moves = GetPossibleMoves_fast(player_id)
    if not possible_moves:
        return

    eval_fn    = _get_eval_fn(use_cnn)
    best_score = -float('inf')
    best_moves = []

    for move in possible_moves:
        row, col = move
        flipped  = GetFlippedDiscs(row, col, player_id)
        FlipDiscs(row, col, player_id, flipped)
        score = Minimax(depth - 1, False, player_id, opp_id, eval_fn)
        Grid[row][col] = 0
        for r, c in flipped:
            Grid[r][c] = opp_id
        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    chosen_row, chosen_col = random.choice(best_moves)
    flipped = GetFlippedDiscs(chosen_row, chosen_col, player_id)
    FlipDiscs(chosen_row, chosen_col, player_id, flipped)


# ---------------------------------------------------------------------------
# 4. MINIMAX AI WITH ALPHA-BETA PRUNING
# ---------------------------------------------------------------------------

def MinimaxAB_v2(depth, is_maximizing, p_id, opp_id, alpha, beta, eval_fn):
    """
    Alpha-beta minimax with move ordering.
 
    Identical correctness contract as MinimaxAB; the only differences are:
      • GetPossibleMoves_fast replaces GetPossibleMoves
      • _sort_moves is applied before iterating, so high-value moves are
        tried first, producing tighter alpha/beta windows and more cutoffs.
 
    eval_fn is injected by PlayAIMinimaxAB_v2 (EvaluateGrid_v2 or
    EvaluateGridCNN for the CNN variant).
    """
    state = CheckGameState()
    if depth == 0 or state != 0:
        return eval_fn(p_id, opp_id)
 
    current_player  = p_id  if is_maximizing else opp_id
    opponent_player = opp_id if is_maximizing else p_id
    possible_moves  = _sort_moves(GetPossibleMoves_fast(current_player))
 
    if not possible_moves:
        if not GetPossibleMoves_fast(opponent_player):
            return eval_fn(p_id, opp_id)
        return MinimaxAB_v2(depth, not is_maximizing, p_id, opp_id,
                            alpha, beta, eval_fn)
 
    if is_maximizing:
        best_val = -float('inf')
        for move in possible_moves:
            row, col = move
            flipped  = GetFlippedDiscs(row, col, current_player)
            FlipDiscs(row, col, current_player, flipped)
            value    = MinimaxAB_v2(depth - 1, False, p_id, opp_id,
                                    alpha, beta, eval_fn)
            Grid[row][col] = 0
            for r, c in flipped:
                Grid[r][c] = opponent_player
            best_val = max(best_val, value)
            alpha    = max(alpha, best_val)
            if alpha >= beta:
                break                        # beta cutoff
        return best_val
    else:
        best_val = float('inf')
        for move in possible_moves:
            row, col = move
            flipped  = GetFlippedDiscs(row, col, current_player)
            FlipDiscs(row, col, current_player, flipped)
            value    = MinimaxAB_v2(depth - 1, True, p_id, opp_id,
                                    alpha, beta, eval_fn)
            Grid[row][col] = 0
            for r, c in flipped:
                Grid[r][c] = opponent_player
            best_val = min(best_val, value)
            beta     = min(beta, best_val)
            if alpha >= beta:
                break                        # alpha cutoff
        return best_val


def PlayAIMinimaxAB_v2(player_id, depth=MINIMAX_DEPTH, use_cnn=False):
    """
    Entry point for the optimised Minimax AI with alpha-beta pruning.
    Wires MinimaxAB_v2 + EvaluateGrid_v2 (or EvaluateGridCNN when use_cnn).
    Interface identical to PlayAIMinimaxAB.
    """
    opp_id         = 3 - player_id
    possible_moves = _sort_moves(GetPossibleMoves_fast(player_id))
    if not possible_moves:
        return
 
    eval_fn    = _get_eval_fn(use_cnn)
    best_score = -float('inf')
    best_moves = []
 
    for move in possible_moves:
        row, col = move
        flipped  = GetFlippedDiscs(row, col, player_id)
        FlipDiscs(row, col, player_id, flipped)
        score = MinimaxAB_v2(depth - 1, False, player_id, opp_id,
                             -float('inf'), float('inf'), eval_fn)
        Grid[row][col] = 0
        for r, c in flipped:
            Grid[r][c] = opp_id
        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)
 
    chosen_row, chosen_col = random.choice(best_moves)
    flipped = GetFlippedDiscs(chosen_row, chosen_col, player_id)
    FlipDiscs(chosen_row, chosen_col, player_id, flipped)


# ---------------------------------------------------------------------------
# 5. MONTE CARLO TREE SEARCH AI (UCT variant)
#
# When use_cnn=True the random rollout is replaced by a CNN leaf evaluation
# (one forward pass instead of a full random playout).  This turns MCTS into
# a neural-network-guided tree search, similar in spirit to AlphaZero's
# evaluation head replacing the random rollout.
#
# The rollout budget (MCTS_TIME_BUDGET) is unchanged: the CNN evaluation
# is typically much faster than a full random playout, so more iterations
# are completed in the same wall-clock time.
# ---------------------------------------------------------------------------

MCTS_TIME_BUDGET   = 1.0
MCTS_C             = math.sqrt(2)
MCTS_ROLLOUT_DEPTH = 60


class MCTSNode:
    """Single node in the MCTS tree (board snapshot, UCB1 statistics)."""

    __slots__ = ("state", "turn", "parent", "move",
                 "children", "untried", "W", "N")

    def __init__(self, state, turn, parent=None, move=None):
        self.state    = state
        self.turn     = turn
        self.parent   = parent
        self.move     = move
        self.children = []
        self.W        = 0.0
        self.N        = 0
        self.untried  = _mcts_legal_moves(state, turn)

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def is_terminal(self):
        return (not _mcts_legal_moves(self.state, 1) and
                not _mcts_legal_moves(self.state, 2))

    def ucb1(self, c):
        if self.N == 0:
            return float('inf')
        return self.W / self.N + c * math.sqrt(math.log(self.parent.N) / self.N)

    def best_child(self, c):
        return max(self.children, key=lambda ch: ch.ucb1(c))


def _mcts_legal_moves(state, player):
    """Legal moves for `player` on a board snapshot (not the global Grid)."""
    moves    = []
    opponent = 3 - player
    for r in range(8):
        for c in range(8):
            if state[r][c] != 0:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                line   = []
                while 0 <= nr < 8 and 0 <= nc < 8 and state[nr][nc] == opponent:
                    line.append((nr, nc))
                    nr += dr
                    nc += dc
                if line and 0 <= nr < 8 and 0 <= nc < 8 and state[nr][nc] == player:
                    moves.append((r, c))
                    break
    return moves


def _mcts_apply_move(state, row, col, player):
    """Apply `player`'s move to a snapshot, returning a new snapshot."""
    new_state = state.copy()
    opponent  = 3 - player
    new_state[row][col] = player
    for dr, dc in DIRECTIONS:
        nr, nc = row + dr, col + dc
        line   = []
        while 0 <= nr < 8 and 0 <= nc < 8 and new_state[nr][nc] == opponent:
            line.append((nr, nc))
            nr += dr
            nc += dc
        if line and 0 <= nr < 8 and 0 <= nc < 8 and new_state[nr][nc] == player:
            for fr, fc in line:
                new_state[fr][fc] = player
    return new_state


def _mcts_rollout_result(state, p_id, opp_id):
    """Random playout; returns +1 / -1 / 0 from p_id's perspective."""
    current_state      = state.copy()
    turn               = p_id
    consecutive_passes = 0
    for _ in range(MCTS_ROLLOUT_DEPTH):
        moves = _mcts_legal_moves(current_state, turn)
        if not moves:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            turn = 3 - turn
            continue
        consecutive_passes = 0
        row, col      = random.choice(moves)
        current_state = _mcts_apply_move(current_state, row, col, turn)
        turn          = 3 - turn
    p_discs   = int(np.sum(current_state == p_id))
    opp_discs = int(np.sum(current_state == opp_id))
    if   p_discs > opp_discs: return  1.0
    elif p_discs < opp_discs: return -1.0
    else:                     return  0.0


def _mcts_cnn_result(state, p_id, opp_id):
    """
    CNN leaf evaluation replacing the random rollout.
    Temporarily writes the snapshot to the global Grid, calls EvaluateGridCNN,
    then restores the original Grid.
    Returns a value in the same sign convention as _mcts_rollout_result:
    positive = p_id leading.
    """
    global Grid
    original = Grid.copy()
    Grid[:] = state
    score = EvaluateGridCNN(p_id, opp_id)
    Grid[:] = original
    # Normalise to [-1, +1] (network output is in disc-count range ~[-64, 64])
    return float(np.clip(score / 64.0, -1.0, 1.0))


def _mcts_select(node):
    while not node.is_terminal() and node.is_fully_expanded():
        node = node.best_child(MCTS_C)
    return node


def _mcts_expand(node):
    if not node.untried:
        return node
    move      = node.untried.pop()
    row, col  = move
    new_state = _mcts_apply_move(node.state, row, col, node.turn)
    next_turn = 3 - node.turn
    if not _mcts_legal_moves(new_state, next_turn):
        next_turn = node.turn
    child = MCTSNode(state=new_state, turn=next_turn, parent=node, move=move)
    node.children.append(child)
    return child


def _mcts_backpropagate(node, result):
    while node is not None:
        node.N += 1
        node.W += result
        node   = node.parent


def _mcts_run(root_state, root_turn, p_id, opp_id, use_cnn=False):
    """
    Full MCTS loop for MCTS_TIME_BUDGET seconds.
    When use_cnn=True, CNN evaluation replaces the random rollout.
    """
    root = MCTSNode(state=root_state, turn=root_turn)
    t0   = time.time()
    while time.time() - t0 < MCTS_TIME_BUDGET:
        leaf   = _mcts_select(root)
        if not leaf.is_terminal():
            leaf = _mcts_expand(leaf)
        if use_cnn and _cnn_available:
            result = _mcts_cnn_result(leaf.state, p_id, opp_id)
        else:
            result = _mcts_rollout_result(leaf.state, p_id, opp_id)
        _mcts_backpropagate(leaf, result)
    return root


def PlayAIMCTS(player_id, use_cnn=False):
    """
    Entry point for MCTS AI.
    use_cnn=True replaces random rollouts with CNN leaf evaluations.
    """
    opp_id         = 3 - player_id
    possible_moves = GetPossibleMoves_fast(player_id)
    if not possible_moves:
        return

    if use_cnn:
        LoadCNN()   # Ensure model is loaded before entering the time loop

    root_state = Grid.copy()
    root       = _mcts_run(root_state, player_id, player_id, opp_id, use_cnn)

    if not root.children:
        row, col = random.choice(possible_moves)
        flipped  = GetFlippedDiscs(row, col, player_id)
        FlipDiscs(row, col, player_id, flipped)
        return

    best_child = max(root.children, key=lambda ch: ch.N)
    row, col   = best_child.move
    flipped = GetFlippedDiscs(row, col, player_id)
    FlipDiscs(row, col, player_id, flipped)


# ---------------------------------------------------------------------------
# 6. Descent Minimax AI (Cohen-Solal & Cazenave, AAMAS 2023)
# ---------------------------------------------------------------------------

def _board_key(turn):
    """Hashable snapshot of the current board state + whose turn it is."""
    return (tuple(Grid.flatten()), turn)


def _heuristic_eval(p_id, opp_id, eval_fn):
    """Thin wrapper: routes to either EvaluateGrid or EvaluateGridCNN."""
    return eval_fn(p_id, opp_id)


def _apply_move(move, player):
    """Place disc, flip, return flipped list. Returns None if illegal."""
    row, col = move
    flipped  = GetFlippedDiscs(row, col, player)
    if not flipped:
        return None
    FlipDiscs(row, col, player, flipped)
    return flipped


def _undo_move(move, player, flipped):
    """Undo a move by clearing the cell and restoring flipped discs."""
    row, col = move
    opponent = 3 - player
    Grid[row][col] = 0
    for r, c in flipped:
        Grid[r][c] = opponent


def descent_iter(turn, S, p_id, opp_id, eval_fn):
    """
    One recursive iteration of Descent Minimax.
    eval_fn is passed through so terminal and non-terminal evaluations
    use the same function (heuristic or CNN).
    """
    key   = _board_key(turn)
    state = CheckGameState()

    if state != 0:
        val    = _heuristic_eval(p_id, opp_id, eval_fn)
        S[key] = {"value": val, "turn": turn}
        return val

    if key not in S:
        next_turn      = opp_id if turn == p_id else p_id
        possible_moves = GetPossibleMoves_fast(turn)

        if not possible_moves:
            return descent_iter(next_turn, S, p_id, opp_id, eval_fn)

        S[key] = {"value": None, "turn": turn, "edge": {}}
        for move in possible_moves:
            flipped = _apply_move(move, turn)
            if flipped is None:
                continue
            child_key   = _board_key(next_turn)
            child_state = CheckGameState()
            if child_state != 0:
                child_val         = _heuristic_eval(p_id, opp_id, eval_fn)
                S[child_key]      = {"value": child_val, "turn": next_turn}
                S[key]["edge"][move] = child_val
            else:
                S[key]["edge"][move] = _heuristic_eval(p_id, opp_id, eval_fn)
            _undo_move(move, turn, flipped)

    edges = S[key]["edge"]
    if not edges:
        return _heuristic_eval(p_id, opp_id, eval_fn)

    best_move = (max if turn == p_id else min)(edges, key=lambda m: edges[m])

    next_turn = opp_id if turn == p_id else p_id
    flipped   = _apply_move(best_move, turn)
    if flipped is not None:
        updated_val = descent_iter(next_turn, S, p_id, opp_id, eval_fn)
        _undo_move(best_move, turn, flipped)
        S[key]["edge"][best_move] = updated_val
        best_move = (max if turn == p_id else min)(edges, key=lambda m: edges[m])

    S[key]["value"] = edges[best_move]
    return S[key]["value"]


def descent(start_turn, S, p_id, opp_id, tau, eval_fn):
    """Time-bounded loop calling descent_iter for `tau` seconds."""
    t0 = time.time()
    while time.time() - t0 < tau:
        descent_iter(start_turn, S, p_id, opp_id, eval_fn)
    return S


def PlayAIDescent(player_id, tau=1.0, use_cnn=False):
    """
    Entry point for Descent Minimax AI.
    use_cnn=True routes the internal heuristic evaluation through EvaluateGridCNN.
    """
    opp_id         = 3 - player_id
    possible_moves = GetPossibleMoves_fast(player_id)
    if not possible_moves:
        return

    eval_fn = _get_eval_fn(use_cnn)
    S       = {}
    descent(player_id, S, player_id, opp_id, tau, eval_fn)

    root_key = _board_key(player_id)
    if root_key in S and S[root_key].get("edge"):
        edges      = S[root_key]["edge"]
        best_score = max(edges.values())
        best_moves = [m for m, v in edges.items() if v == best_score]
        chosen     = random.choice(best_moves)
    else:
        chosen = random.choice(possible_moves)

    flipped = _apply_move(chosen, player_id)
    if flipped is None:
        row, col = random.choice(possible_moves)
        flipped  = GetFlippedDiscs(row, col, player_id)
        FlipDiscs(row, col, player_id, flipped)


# ---------------------------------------------------------------------------
# AI dispatcher
#
# Keys "1"-"6"  : standard strategies with heuristic evaluation
# Keys "1c"-"6c": same strategies with CNN evaluation (use_cnn=True)
#
# Each entry is a lambda that accepts player_id as its only required argument,
# forwarding use_cnn to the underlying function.
# ---------------------------------------------------------------------------

import os   # needed by LoadCNN (os.path.isfile)

AI_FUNCTIONS = {
    # --- Heuristic evaluation (original) ---
    "1":  lambda player_id: PlayAIRandom(player_id,     use_cnn=False),
    "2":  lambda player_id: PlayAIHeuristic(player_id,  use_cnn=False),
    "3":  lambda player_id: PlayAIMinimax(player_id,    use_cnn=False),
    "4":  lambda player_id: PlayAIMinimaxAB_v2(player_id,  use_cnn=False),
    "5":  lambda player_id: PlayAIMCTS(player_id,       use_cnn=False),
    "6":  lambda player_id: PlayAIDescent(player_id,    use_cnn=False),
    # --- CNN evaluation ---
    "1c": lambda player_id: PlayAIRandom(player_id,     use_cnn=True),
    "2c": lambda player_id: PlayAIHeuristic(player_id,  use_cnn=True),
    "3c": lambda player_id: PlayAIMinimax(player_id,    use_cnn=True),
    "4c": lambda player_id: PlayAIMinimaxAB_v2(player_id,  use_cnn=True),
    "5c": lambda player_id: PlayAIMCTS(player_id,       use_cnn=True),
    "6c": lambda player_id: PlayAIDescent(player_id,    use_cnn=True),
}

VALID_TYPES     = list(AI_FUNCTIONS.keys())
VALID_PLAY_MODE = ["1","2","3","4","5","6","1c","2c","3c","4c","5c","6c"]
VALID_SIM_MODE  = ["1","2","3","4","5","6","1c","2c","3c","4c","5c","6c"]


#################################################################################
# BLOCK 4 - Display and interface (tkinter)

CELL = 80
LARG = 8 * CELL
HAUT = 8 * CELL + 40


def HighlightMoves(player):
    """Draw a small hint circle on every legal move cell for `player`."""
    for row, col in GetPossibleMoves_fast(player):
        xc = col * CELL
        yc = row * CELL + 40
        canvas.create_oval(
            xc + CELL//2 - 8, yc + CELL//2 - 8,
            xc + CELL//2 + 8, yc + CELL//2 + 8,
            fill="lime green", outline="lime green"
        )


def Dessine(show_hints=True):
    """Redraw the entire board: header, grid, discs, optional hints."""
    canvas.delete("all")

    black_count = int(np.sum(Grid == 1))
    white_count = int(np.sum(Grid == 2))

    # CNN indicator in the header
    cnn_tag = " [CNN]" if player2_type.endswith("c") else ""
    score_text = (
        f"Human (Black): {score_h} pts  |  "
        f"AI{cnn_tag} (White): {score_ia} pts  |  "
        f"Discs — Black: {black_count}  White: {white_count}"
    )
    canvas.create_rectangle(0, 0, LARG, 40, fill="black", outline="black")
    canvas.create_text(LARG // 2, 20, text=score_text,
                       fill="white", font=('Arial', 10, 'bold'))

    canvas.create_rectangle(0, 40, LARG, HAUT, fill=grid_color, outline=grid_color)

    for i in range(9):
        canvas.create_line(i * CELL, 40, i * CELL, HAUT, fill="black", width=1)
    for j in range(9):
        canvas.create_line(0, j * CELL + 40, LARG, j * CELL + 40, fill="black", width=1)

    for row in range(8):
        for col in range(8):
            xc = col * CELL
            yc = row * CELL + 40
            if Grid[row][col] == 1:
                canvas.create_oval(xc+6, yc+6, xc+CELL-6, yc+CELL-6,
                                   fill="black", outline="white")
            elif Grid[row][col] == 2:
                canvas.create_oval(xc+6, yc+6, xc+CELL-6, yc+CELL-6,
                                   fill="white", outline="black")

    if show_hints and current_turn == 1:
        HighlightMoves(1)


def MouseClick(event):
    """Handle left-click: reset after match end, or process human move."""
    global StartMatch, current_turn

    if StartMatch:
        InitBoard()
        StartMatch   = False
        current_turn = 1
        Dessine()
        return

    if current_turn != 1:
        return

    col = event.x // CELL
    row = (event.y - 40) // CELL

    if not (0 <= row < 8 and 0 <= col < 8):
        return

    flipped = GetFlippedDiscs(row, col, 1)
    if not flipped:
        return

    FlipDiscs(row, col, 1, flipped)
    Dessine(show_hints=False)

    state = CheckGameState()
    if state != 0:
        _show_result(state)
        EndMatch(state)
        Dessine(show_hints=False)
        return

    current_turn = 2
    _run_ai_turn()


def _run_ai_turn():
    """Execute the AI move (or pass), then hand control back to the human."""
    global current_turn

    ai_moves = GetPossibleMoves_fast(2)

    if ai_moves:
        Window.update()
        time.sleep(0.3)
        AI_FUNCTIONS[player2_type](player_id=2)
        Dessine(show_hints=False)

        state = CheckGameState()
        if state != 0:
            _show_result(state)
            EndMatch(state)
            Dessine(show_hints=False)
            return
    else:
        canvas.create_text(LARG // 2, HAUT // 2,
                           text="AI has no moves — your turn again",
                           fill="yellow", font=('Arial', 14, 'bold'),
                           tags="pass_msg")
        Window.update()
        time.sleep(1.2)

    current_turn = 1

    if not GetPossibleMoves_fast(1):
        state = CheckGameState()
        _show_result(state)
        EndMatch(state)
        Dessine(show_hints=False)
        return

    Dessine(show_hints=True)


def _show_result(state):
    """Display a messagebox with the match outcome."""
    black_count = int(np.sum(Grid == 1))
    white_count = int(np.sum(Grid == 2))
    detail = f"Black: {black_count}  |  White: {white_count}\n\nClick to play again."
    if state == 1:
        title, msg = "Black wins!", f"You win!\n{detail}"
    elif state == 2:
        title, msg = "White wins!", f"AI wins!\n{detail}"
    else:
        title, msg = "Draw!", f"It's a draw.\n{detail}"
    messagebox.showinfo(title, msg)


#################################################################################
# BLOCK 5 - Simulation and execution control

def RunBackgroundGame(p1_type, p2_type):
    """
    Run a complete headless game between two AIs.
    Returns 1 (Black wins), 2 (White wins), or 4 (draw).
    """
    global Grid
    InitBoard()

    turn               = 1
    consecutive_passes = 0

    while True:
        ai_fn = AI_FUNCTIONS[p1_type] if turn == 1 else AI_FUNCTIONS[p2_type]
        moves = GetPossibleMoves_fast(turn)

        if moves:
            consecutive_passes = 0
            ai_fn(player_id=turn)
        else:
            consecutive_passes += 1

        state = CheckGameState()
        if state != 0:
            return state

        if consecutive_passes >= 2:
            return WinnerCheck()

        turn = 3 - turn


def StartSimulation(num_games, p1_type, p2_type):
    """Run `num_games` headless matches and print aggregated results."""
    cnn_note = ""
    if p1_type.endswith("c") or p2_type.endswith("c"):
        if LoadCNN():
            cnn_note = " (CNN loaded)"
        else:
            cnn_note = " (CNN unavailable — falling back to heuristic)"
    print(f"\n--- SIMULATION: AI {p1_type} (Black) vs AI {p2_type} (White){cnn_note} ---")
    results = {1: 0, 2: 0, 4: 0}

    for _ in range(num_games):
        winner = RunBackgroundGame(p1_type, p2_type)
        results[winner] += 1

    print(f"RESULTS: Black wins: {results[1]} | White wins: {results[2]} | Draws: {results[4]}")
    print(f"Win rate (White / AI): {(results[2] / num_games) * 100:.2f}%")



# ---------------------------------------------------------------------------
# COMPLETE TEST
# ---------------------------------------------------------------------------

# SECTION A - St<tistical utilities
def _wilson_ci(wins, n, z=1.96):
    """
    Wilson score confidence interval for a proportion.

    Args:
        wins (int): successes observed
        n    (int): total trials
        z  (float): z-score (1.96 for 95%, 2.576 for 99%)

    Returns:
        (p_hat, lower, upper) — all in [0, 1]
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat  = wins / n
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)


def _significance_label(ci_low, ci_high):
    """Return a short verdict string based on the Wilson CI versus 50%."""
    if ci_low > 0.50:
        return "P1 SIGNIFICANTLY BETTER (p<0.05)"
    if ci_high < 0.50:
        return "P2 SIGNIFICANTLY BETTER (p<0.05)"
    return "No significant difference (CI crosses 50%)"


# SECTION B - Single-phase benchmark engine
def _time_one_game(p1, p2):
    """
    Run a single background game and return elapsed seconds.
    Used for speed calibration before committing to a full batch.
    """
    t0 = time.time()
    RunBackgroundGame(p1, p2)
    return time.time() - t0


def _choose_n(avg_sec_per_game, budget_min=20, n_min=100, n_max=400):
    """
    Decide how many games to run given the per-game timing constraint.

    Rules:
    - If even n_min games exceed the budget, run n_min anyway (mandatory floor).
    - Otherwise run as many as possible up to n_max within the budget.
    - n must be even to guarantee symmetric colour split.

    Returns:
        (n, estimated_total_seconds)
    """
    budget_sec = budget_min * 60.0

    # Maximum games that fit in the budget (symmetric: step by 2)
    if avg_sec_per_game > 0:
        n_feasible = int(budget_sec / avg_sec_per_game)
        n_feasible = (n_feasible // 2) * 2   # round down to nearest even
    else:
        n_feasible = n_max

    n = max(n_min, min(n_max, n_feasible))
    estimated = avg_sec_per_game * n
    return n, estimated


def _run_phase(p1, p2, label, log_lines, cnn_needed=False):
    """
    Execute one benchmark phase between p1 and p2.

    Protocol:
    - Calibrate timing with one sample game.
    - Decide n (100-400) based on a 20-minute budget.
    - Run n/2 games with p1=Black, then n/2 games with p1=White.
    - Print progress every 10 games.
    - Compute and print Wilson CI statistics.
    - Append all data to log_lines (for the external report file).

    Args:
        p1, p2    (str): AI type keys (e.g. "1", "2c")
        label     (str): human-readable phase description
        log_lines (list): mutable list; results are appended here
        cnn_needed (bool): if True, attempt LoadCNN() before timing

    Returns:
        dict with all result fields (for the final summary table)
    """
    SEP = "=" * 66

    # --- Header ---
    header = f"\n{SEP}\n  PHASE: {label}\n  P1={p1}  P2={p2}\n{SEP}"
    print(header)
    log_lines.append(header)

    # --- CNN pre-load (if required) ---
    if cnn_needed:
        cnn_ok = LoadCNN()
        cnn_msg = "  CNN loaded successfully." if cnn_ok \
                  else "  CNN unavailable — falling back to heuristic evaluation."
        print(cnn_msg)
        log_lines.append(cnn_msg)

    # --- Speed calibration ---
    print("  [Calibration] Running 1 sample game to measure speed...")
    log_lines.append("  [Calibration] 1 sample game")
    avg_sec = _time_one_game(p1, p2)
    calib_msg = (f"  Average time per game : {avg_sec:.3f} s")
    print(calib_msg)
    log_lines.append(calib_msg)

    n, est_sec = _choose_n(avg_sec)
    est_min    = est_sec / 60.0
    plan_msg   = (
        f"  Games planned         : {n}  "
        f"(min=100, max=400, budget=20 min)\n"
        f"  Estimated total time  : {est_min:.1f} min  ({est_sec:.0f} s)\n"
        f"  Expected 95% CI width : ±{100 * 1.96 * 0.5 / math.sqrt(n):.1f}%"
    )
    print(plan_msg)
    log_lines.append(plan_msg)

    half = n // 2   # games per colour block

    # Storage: individual game records
    game_records  = []   # list of dicts: {game_id, p1_colour, winner, duration_s}
    p1_wins       = 0
    total_time    = 0.0

    print(f"\n  --- BLOCK A: P1 ({p1}) as Black ({half} games) ---")
    log_lines.append(f"\n  --- BLOCK A: P1 ({p1}) as Black ({half} games) ---")

    # --- Block A: p1 = Black (player 1, first mover) ---
    for i in range(half):
        t0     = time.time()
        winner = RunBackgroundGame(p1, p2)
        dur    = time.time() - t0
        total_time += dur

        p1_won = (winner == 1)   # Black wins => p1 wins
        if p1_won:
            p1_wins += 1

        game_records.append({
            "game_id":   i + 1,
            "p1_colour": "Black",
            "winner_code": winner,
            "p1_won":    p1_won,
            "duration_s": dur,
        })

        # Progress every 10 games
        if (i + 1) % 10 == 0:
            prog = (
                f"  [A] Game {i+1:>3}/{half}  "
                f"P1 wins: {p1_wins:>3}  "
                f"Elapsed: {total_time:.1f}s"
            )
            print(prog)
            log_lines.append(prog)

    print(f"\n  --- BLOCK B: P1 ({p1}) as White ({half} games) ---")
    log_lines.append(f"\n  --- BLOCK B: P1 ({p1}) as White ({half} games) ---")

    # --- Block B: p1 = White (player 2, second mover) ---
    for i in range(half):
        t0     = time.time()
        winner = RunBackgroundGame(p2, p1)   # roles inverted at function call level
        dur    = time.time() - t0
        total_time += dur

        p1_won = (winner == 2)   # White wins => p1 wins (p1 is now White)
        if p1_won:
            p1_wins += 1

        game_records.append({
            "game_id":   half + i + 1,
            "p1_colour": "White",
            "winner_code": winner,
            "p1_won":    p1_won,
            "duration_s": dur,
        })

        if (i + 1) % 10 == 0:
            prog = (
                f"  [B] Game {i+1:>3}/{half}  "
                f"P1 wins (total): {p1_wins:>3}  "
                f"Elapsed: {total_time:.1f}s"
            )
            print(prog)
            log_lines.append(prog)

    # --- Statistics ---
    p1_rate, ci_low, ci_high = _wilson_ci(p1_wins, n)
    verdict = _significance_label(ci_low, ci_high)

    draws    = sum(1 for r in game_records if r["winner_code"] == 4)
    p2_wins  = n - p1_wins - draws

    stats_block = (
        f"\n  {'─'*60}\n"
        f"  RESULTS  ({n} games total, {total_time:.1f} s elapsed)\n"
        f"  {'─'*60}\n"
        f"  P1 ({p1}) wins : {p1_wins:>4}  ({p1_rate*100:.1f}%)\n"
        f"  P2 ({p2}) wins : {p2_wins:>4}  ({(1-p1_rate - draws/n)*100:.1f}%)\n"
        f"  Draws          : {draws:>4}  ({draws/n*100:.1f}%)\n"
        f"  95% Wilson CI  : [{ci_low*100:.1f}%, {ci_high*100:.1f}%]\n"
        f"  Verdict        : {verdict}\n"
        f"  {'─'*60}"
    )
    print(stats_block)
    log_lines.append(stats_block)

    # Individual game CSV block in log
    log_lines.append("\n  INDIVIDUAL GAME LOG (game_id, p1_colour, winner_code, p1_won, duration_s)")
    for r in game_records:
        log_lines.append(
            f"  {r['game_id']}, {r['p1_colour']}, {r['winner_code']}, "
            f"{int(r['p1_won'])}, {r['duration_s']:.4f}"
        )

    return {
        "label":      label,
        "p1":         p1,
        "p2":         p2,
        "n":          n,
        "p1_wins":    p1_wins,
        "p2_wins":    p2_wins,
        "draws":      draws,
        "p1_rate":    p1_rate,
        "ci_low":     ci_low,
        "ci_high":    ci_high,
        "verdict":    verdict,
        "elapsed_s":  total_time,
    }


# SECTION C - Report file writeR
def _write_report(log_lines, summary_rows, filepath):
    """
    Write the full benchmark report to a plain-text file.

    Args:
        log_lines    (list of str): per-phase detailed lines accumulated during the run
        summary_rows (list of dict): one dict per phase (from _run_phase)
        filepath     (str): destination file path
    """
    with open(filepath, "w", encoding="utf-8") as f:
        # File header
        f.write("OTHELLO AI BENCHMARK REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 66 + "\n\n")

        # Full per-phase detail
        f.write("DETAILED PHASE LOGS\n")
        f.write("=" * 66 + "\n")
        for line in log_lines:
            f.write(line + "\n")

        # Summary table
        f.write("\n\n")
        f.write("=" * 66 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 66 + "\n")
        col_w = [36, 5, 6, 6, 6, 20, 8]
        hdr = (
            f"{'Phase':<{col_w[0]}} {'N':>{col_w[1]}} "
            f"{'P1W%':>{col_w[2]}} {'P2W%':>{col_w[3]}} "
            f"{'D%':>{col_w[4]}} {'95% CI':^{col_w[5]}} "
            f"{'Time(s)':>{col_w[6]}}"
        )
        f.write(hdr + "\n")
        f.write("-" * 66 + "\n")
        for r in summary_rows:
            p1w_pct = r["p1_rate"] * 100
            p2w_pct = (r["p2_wins"] / r["n"]) * 100
            d_pct   = (r["draws"]   / r["n"]) * 100
            ci_str  = f"[{r['ci_low']*100:.1f}%, {r['ci_high']*100:.1f}%]"
            row_str = (
                f"{r['label']:<{col_w[0]}} {r['n']:>{col_w[1]}} "
                f"{p1w_pct:>{col_w[2]}.1f} {p2w_pct:>{col_w[3]}.1f} "
                f"{d_pct:>{col_w[4]}.1f} {ci_str:^{col_w[5]}} "
                f"{r['elapsed_s']:>{col_w[6]}.1f}"
            )
            f.write(row_str + "\n")
        f.write("=" * 66 + "\n")

        # Verdict column
        f.write("\nVERDICTS\n")
        for r in summary_rows:
            f.write(f"  {r['label']:<36}  {r['verdict']}\n")


# SECTION D - Main entry point: CompleteTest()
def CompleteTest():
    """
    Run the full ESIEE Othello benchmark suite.

    Phase plan:
    -- Baseline / non-CNN --
    1.  Random      vs Random          (sanity check: expected ~50%)
    2.  Heuristic   vs Random          (expected >80%)
    3.  Minimax AB  vs Heuristic       (expected >60%)
    4.  MCTS        vs Heuristic       (expected >55%)
    5.  Descent     vs Heuristic       (expected >65%)
    6.  Descent     vs Minimax         (expected >55%)

    -- CNN enhancement --
    8.  Heuristic CNN   vs Heuristic   (expected >50% if CNN improves eval)
    9.  Minimax AB CNN  vs Minimax AB  (expected >50%)
    10. MCTS CNN        vs MCTS        (expected >50%)
    11. Descent CNN     vs Descent     (expected >50%)

    Each phase:
    - Times one sample game.
    - Decides n in [100, 400] within a 20-minute budget.
    - Runs n/2 games with P1=Black, then n/2 with P1=White.
    - Prints progress every 10 games.
    - Computes Wilson 95% CI and prints verdict.

    All data are saved to 'results/benchmark/othello_benchmark_results_cnn_v2.txt'
    """
    REPORT_FILE = os.path.join(ROOT, "results", "benchmark", "othello_benchmark_results_cnn_v2.txt")

    # These are the matchups: (p1_key, p2_key, label, needs_cnn)
    PHASES = [
        # -- Non-CNN phases --
        ("1",  "1",  "1. Random vs Random (sanity check)",        False),
        ("2",  "1",  "2. Heuristic vs Random",                    False),
        ("4",  "2",  "3. Minimax AB vs Heuristic",                False),
        ("5",  "2",  "4. MCTS vs Heuristic",                      False),
        ("6",  "2",  "5. Descent vs Heuristic",                   False),
        ("6",  "4",  "6. Descent vs Minimax AB",                  False),
        # -- CNN phases --
        ("2c", "2",  "8.  Heuristic CNN vs Heuristic",            True),
        ("4c", "4",  "9.  Minimax AB CNN vs Minimax AB",          True),
        ("5c", "5",  "10. MCTS CNN vs MCTS",                      True),
        ("6c", "6",  "11. Descent CNN vs Descent",                True),
    ]

    # Accumulated data
    log_lines    = []
    summary_rows = []

    # Suite header
    suite_header = (
        "\n" + "#" * 66 + "\n"
        "#  OTHELLO COMPLETE TEST SUITE\n"
        f"#  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"#  Phases: {len(PHASES)}  |  Budget per phase: 20 min  |  n: 100–400\n"
        + "#" * 66
    )
    print(suite_header)
    log_lines.append(suite_header)

    suite_t0 = time.time()

    # Run each phase
    for p1, p2, label, cnn_needed in PHASES:
        result = _run_phase(p1, p2, label, log_lines, cnn_needed=cnn_needed)
        summary_rows.append(result)

    # Global summary
    total_elapsed = time.time() - suite_t0
    suite_footer = (
        "\n" + "#" * 66 + "\n"
        "  COMPLETE TEST SUITE — FINAL SUMMARY\n"
        + "#" * 66
    )
    print(suite_footer)
    log_lines.append(suite_footer)

    # Print summary table to console
    print(f"\n  {'Phase':<36} {'N':>4}  {'P1%':>5}  {'95% CI':^20}  Verdict")
    print(f"  {'─'*36} {'─'*4}  {'─'*5}  {'─'*20}  {'─'*36}")
    for r in summary_rows:
        ci_str = f"[{r['ci_low']*100:.1f}%, {r['ci_high']*100:.1f}%]"
        print(
            f"  {r['label']:<36} {r['n']:>4}  "
            f"{r['p1_rate']*100:>4.1f}%  {ci_str:^20}  {r['verdict']}"
        )

    elapsed_msg = f"\n  Total suite time: {total_elapsed/60:.1f} min ({total_elapsed:.0f} s)"
    print(elapsed_msg)
    log_lines.append(elapsed_msg)

    # Write report file
    _write_report(log_lines, summary_rows, REPORT_FILE)
    abs_path = os.path.abspath(REPORT_FILE)
    report_msg = f"\n  Report saved to: {abs_path}"
    print(report_msg)


#################################################################################
# Entry point

_PLAY_PROMPT = (
    "Choose your opponent:\n"
    "  1:  Random           1c: Random + CNN\n"
    "  2:  Heuristic        2c: Heuristic + CNN\n"
    "  3:  Minimax          3c: Minimax + CNN\n"
    "  4:  Minimax AB       4c: Minimax AB + CNN\n"
    "  5:  MCTS             5c: MCTS + CNN\n"
    "  6:  Descent          6c: Descent + CNN\n"
)

_SIM_PROMPT = (
    "Available AI types: 1 2 3 4 5 6  (heuristic)\n"
    "                    1c 2c 3c 4c 5c 6c  (CNN evaluation)\n"
)


if __name__ == "__main__":
    print("WELCOME TO OTHELLO")
    print("1: Play vs AI | 2: Simulation | 3: Complete Test")

    mode = input("Select Mode: ").strip()
    if mode not in ['1', '2', '3']:
        print("Input not accepted. Exit")
        exit()

    if mode == '1':
        print()
        print(_PLAY_PROMPT)
        player2_type = input("Choice: ").strip()
        if player2_type not in VALID_PLAY_MODE:
            print("Input not accepted. Exit")
            exit()

        # Pre-load CNN if a CNN variant was chosen (avoids first-move delay)
        if player2_type.endswith("c"):
            if not LoadCNN():
                print("[WARNING] CNN not available — falling back to heuristic evaluation.")

        Window = tk.Tk()
        Window.geometry(f"{LARG}x{HAUT}")
        Window.title("ESIEE - Othello")
        Window.resizable(False, False)

        F = tk.Frame(Window)
        F.pack(side="top", fill="both", expand=True)
        F.grid_rowconfigure(0, weight=1)
        F.grid_columnconfigure(0, weight=1)

        ListePages = {}

        def CreerUnePage(page_id):
            frame = tk.Frame(F)
            ListePages[page_id] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            return frame

        def AfficherPage(page_id):
            ListePages[page_id].tkraise()

        Frame0 = CreerUnePage(0)
        canvas = tk.Canvas(Frame0, width=LARG, height=HAUT, bg="black")
        canvas.place(x=0, y=0)
        canvas.bind('<ButtonPress-1>', MouseClick)

        AfficherPage(0)
        InitBoard()
        Dessine()
        Window.mainloop()

    elif mode == '2':
        print()
        print(_SIM_PROMPT)
        p1 = input("Select AI 1 (Black): ").strip()
        if p1 not in VALID_SIM_MODE:
            print("Input not accepted. Exit")
            exit()
        p2 = input("Select AI 2 (White): ").strip()
        if p2 not in VALID_SIM_MODE:
            print("Input not accepted. Exit")
            exit()
        games = int(input("Number of games to simulate: "))
        StartSimulation(games, p1, p2)
        exit()
    
    elif mode == '3':
        CompleteTest()
        exit()