# =============================================================================
# OthelloCNN_Train.py
# Colab training script for the Othello board-value CNN.
#
# Pipeline:
#   1. Mount Google Drive and copy the .wtb archive into the VM.
#   2. Parse all WTB files (French Federation format) into (board, value) pairs.
#   3. Train a lightweight residual CNN to predict the final disc-margin
#      from any mid-game board position (regression target in [-64, +64]).
#   4. Save the trained model weights (.pth) back to Drive.
#
# Drive layout expected:
#   MyDrive/thesis/Othello/CNN/
#       wtb_files.zip       <- compressed folder of *.wtb game files
#       othello_cnn.pth     <- written here after training
#
# WTB format reference:
#   Each game record is 68 bytes:
#     Bytes  0- 1 : year (uint16-LE)
#     Byte       2 : tournament / source tag (ignored)
#     Byte       3 : number of moves in the game (n_moves)
#     Bytes  4-67  : move sequence, one byte per move (max 60 moves + padding)
#                    Each byte encodes a square as row*8+col (0-based).
#                    0x00 = pass.
#                    Final disc count stored at byte 4 + n_moves (after moves):
#                    First two remaining bytes = black_count, white_count.
#
# Architecture choice - why NOT TinyCNN_Sport's ConvBlock design:
#   The sport classifier reduces a 64x64 image to a class label using 3x max-
#   pooling stages (64->8 spatial). An 8x8 Othello board is already tiny;
#   pooling would destroy spatial information in one step. Instead we use a
#   stack of residual blocks with NO max-pooling: spatial resolution is kept at
#   8x8 throughout the feature extractor so every cell retains positional
#   identity. A single global-average-pool at the very end collapses the
#   feature maps to a scalar value. The result is a net with ~300k parameters,
#   fast enough to call thousands of times per second on a CPU at inference time.
# =============================================================================

# ---------------------------------------------------------------------------
# 0. Drive setup (mirrors TinyCNN_Sport header)
# ---------------------------------------------------------------------------

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

# Copy the WTB archive from Drive into the Colab VM's fast local storage
import os, subprocess, shutil

wtb_source = '/content/drive/MyDrive/thesis/Othello/Dataset/dataset.zip'
wtb_dest   = '/content/wtb_files.zip'
shutil.copy(wtb_source, wtb_dest)

# Unzip into /content/wtb_files/
os.makedirs('/content/wtb_files', exist_ok=True)
subprocess.run(['unzip', '-q', '-o', wtb_dest, '-d', '/content/wtb_files'], check=True)
print("WTB files:")
print(os.listdir('/content/wtb_files'))

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

import os
import struct
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# 2. WTB parser
#
# The WTHOR / WTB format (used by the French Othello Federation and Cassio)
# stores games in fixed-size records.  Corrected layout:
#
# FILE HEADER (16 bytes):
#   Bytes 00-03 : year/century tags (ignored)
#   Bytes 04-07 : n_games in this file (uint32 little-endian)  <-- KEY
#   Bytes 08-15 : n_records, board size, reserved (ignored)
#
# GAME RECORD (68 bytes):
#   Bytes 00-01 : tournament label index (uint16 LE, ignored)
#   Bytes 02-03 : black player label index (uint16 LE, ignored)
#   Bytes 04-05 : white player label index (uint16 LE, ignored)
#   Byte  06    : black disc count at game end
#   Byte  07    : theoretical score (ignored)
#   Bytes 08-67 : 60 move bytes; 0x00 = pass or unused padding
#                 Encoding: (col+1)*10 + (row+1), both 1-based.
#
# Reference: https://www.potvincles.be/othello/wthor.htm
# ---------------------------------------------------------------------------

RECORD_SIZE  = 68
HEADER_SIZE  = 16
MOVES_OFFSET = 8    # moves start at byte 8 of each record, NOT byte 4
SCORE_OFFSET = 6    # black disc count is at byte 6, NOT 4+n_moves
N_MOVE_SLOTS = 60

_START_BLACK = {(3, 4), (4, 3)}
_START_WHITE = {(3, 3), (4, 4)}

DIRECTIONS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def _init_board():
    """Return (black_set, white_set) for the standard Othello starting position."""
    return set(_START_BLACK), set(_START_WHITE)


def _get_flipped(row, col, player, black_set, white_set):
    """
    Return the list of (r,c) cells that would be flipped if `player` places
    at (row, col).  player=1 -> Black, player=2 -> White.
    Works on explicit sets rather than numpy arrays for parsing speed.
    """
    own   = black_set if player == 1 else white_set
    opp   = white_set if player == 1 else black_set
    flipped = []

    for dr, dc in DIRECTIONS_8:
        r, c  = row + dr, col + dc
        line  = []
        while 0 <= r < 8 and 0 <= c < 8 and (r, c) in opp:
            line.append((r, c))
            r += dr
            c += dc
        if line and 0 <= r < 8 and 0 <= c < 8 and (r, c) in own:
            flipped.extend(line)

    return flipped


def _apply_move(row, col, player, black_set, white_set):
    """Place disc and flip; mutates black_set / white_set in place."""
    own = black_set if player == 1 else white_set
    opp = white_set if player == 1 else black_set
    flipped = _get_flipped(row, col, player, black_set, white_set)
    own.add((row, col))
    for cell in flipped:
        opp.discard(cell)
        own.add(cell)


def _board_to_tensor(black_set, white_set, turn):
    """
    Encode the board as a (3, 8, 8) float32 tensor:
      channel 0: 1.0 where Black has a disc
      channel 1: 1.0 where White has a disc
      channel 2: 1.0 everywhere if it is Black's turn, 0.0 otherwise
    The perspective is always Black=player1; margin target is black-white.
    """
    t = np.zeros((3, 8, 8), dtype=np.float32)
    for r, c in black_set:
        t[0, r, c] = 1.0
    for r, c in white_set:
        t[1, r, c] = 1.0
    if turn == 1:   # Black to move
        t[2, :, :] = 1.0
    return t


def _legal_moves(player, black_set, white_set):
    """Return list of (row, col) legal moves for `player`."""
    own = black_set if player == 1 else white_set
    opp = white_set if player == 1 else black_set
    moves = []
    occupied = own | opp
    for r in range(8):
        for c in range(8):
            if (r, c) in occupied:
                continue
            if _get_flipped(r, c, player, black_set, white_set):
                moves.append((r, c))
    return moves


def parse_wtb_file(filepath):
    """
    Parse a single .wtb file (WTHOR format) into a list of GAMES.

    Each element of the returned list is one game, represented as a list of
    (tensor, margin) pairs: one pair per mid-game position of that game, all
    sharing the same final margin.  Keeping positions grouped by game allows
    the train/validation split to be performed at game level, so that
    positions from the same game never end up on both sides of the split.

    Reads n_games from the file header (bytes 4-7, uint32 LE).
    Moves start at byte 8 of each 68-byte record.
    Move encoding: (col+1)*10 + (row+1), both 1-based decimal digits.

    Opening positions (fewer than 10 moves played) are skipped.
    Games that yield no position (e.g. corrupt or very short records) are dropped.
    """
    games = []

    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        return games

    # Read authoritative game count from header bytes 4-7 (uint32 LE)
    n_games = struct.unpack_from('<I', data, 4)[0]
    max_possible = (len(data) - HEADER_SIZE) // RECORD_SIZE
    if n_games == 0 or n_games > max_possible:
        n_games = max_possible  # fall back if header is corrupt

    for game_idx in range(n_games):
        offset = HEADER_SIZE + game_idx * RECORD_SIZE
        if offset + RECORD_SIZE > len(data):
            break

        record = data[offset: offset + RECORD_SIZE]

        black_set, white_set = _init_board()
        turn               = 1   # Black moves first
        positions          = []  # board tensors collected during replay
        n_played           = 0   # non-pass moves applied
        consecutive_passes = 0

        for slot in range(N_MOVE_SLOTS):
            byte = record[MOVES_OFFSET + slot]

            if byte == 0x00:
                # 0x00 = pass or end-of-record padding.
                # If the player still has legal moves, the remaining slots
                # are just zero-padding — stop replaying.
                if _legal_moves(turn, black_set, white_set):
                    break
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break   # Double pass: game over
                turn = 3 - turn
                continue

            consecutive_passes = 0

            # Decode: tens digit = col (1-based), units digit = row (1-based)
            col = (byte // 10) - 1
            row = (byte  % 10) - 1

            if not (0 <= row < 8 and 0 <= col < 8):
                continue  # corrupt byte, skip

            if n_played >= 10:  # skip opening phase
                positions.append(_board_to_tensor(black_set, white_set, turn))

            _apply_move(row, col, turn, black_set, white_set)
            n_played += 1
            turn = 3 - turn

        # Compute margin from the replayed final board state
        margin = float(len(black_set) - len(white_set))

        if positions:
            games.append([(tensor, margin) for tensor in positions])

    return games


def parse_all_wtb(folder):
    """
    Walk `folder` recursively, parse every *.wtb file found, and return a
    combined list of games (each game = list of (tensor, margin) pairs).

    Files are processed in sorted order so that, together with the fixed
    seed used for the split, the game-level train/validation partition is
    reproducible across runs.
    """
    all_games = []
    files = sorted(glob.glob(os.path.join(folder, '**', '*.wtb'), recursive=True))
    print(f"Found {len(files)} WTB file(s) in {folder}")

    for i, fp in enumerate(files):
        all_games.extend(parse_wtb_file(fp))
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            n_pos = sum(len(g) for g in all_games)
            print(f"  Parsed {i+1}/{len(files)} files — "
                  f"{len(all_games)} games, {n_pos} positions so far")

    n_pos = sum(len(g) for g in all_games)
    print(f"\nTotal: {len(all_games)} games, {n_pos} positions")
    return all_games

# ---------------------------------------------------------------------------
# 3. PyTorch Dataset
# ---------------------------------------------------------------------------

class OthelloDataset(Dataset):
    """
    Wraps a list of (np.ndarray board, float margin) pairs as a PyTorch Dataset.

    Optionally mirrors boards horizontally/vertically/diagonally (8-fold
    Othello symmetry) to multiply data without additional game files.
    """

    def __init__(self, samples, augment=True):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board, margin = self.samples[idx]   # board: (3,8,8) ndarray

        if self.augment:
            # Apply one of the 8 dihedral symmetries uniformly at random.
            # Because Othello is symmetric under these transformations,
            # the margin is unchanged.
            k        = random.randint(0, 3)
            board    = np.rot90(board, k=k, axes=(1, 2)).copy()
            if random.random() < 0.5:
                board = np.flip(board, axis=2).copy()   # Horizontal mirror

        x = torch.tensor(board,  dtype=torch.float32)
        y = torch.tensor(margin, dtype=torch.float32)
        return x, y

# ---------------------------------------------------------------------------
# 4. CNN Architecture
#
# OthelloCNN differs from TinyCNN_Sport in three key ways:
#
#  a) NO max-pooling after each block: the board is already 8x8, so pooling
#     would destroy the positional structure that makes Othello strategy
#     possible (corner vs edge vs centre matters enormously).
#
#  b) Residual (skip) connections: the network needs to pass "who owns the
#     corners" information from the first layer all the way to the output
#     without it washing out.  Residual connections make this reliable.
#
#  c) Regression output (single scalar): we predict the final disc margin
#     in [-64, +64] rather than a class probability.  The head is therefore
#     Linear(channels, 1) with no softmax.
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    One residual block: Conv -> BN -> ReLU -> Conv -> BN, plus skip connection.
    Spatial size (8x8) is preserved throughout (padding=1 on 3x3 kernels).
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out      = F.relu(self.bn1(self.conv1(x)))
        out      = self.bn2(self.conv2(out))
        return F.relu(out + residual)   # Skip connection


class OthelloCNN(nn.Module):
    """
    Lightweight residual CNN for Othello board-value regression.

    Input  : (batch, 3, 8, 8) float32 tensor
               channel 0 = Black disc mask
               channel 1 = White disc mask
               channel 2 = Turn plane (all-ones if Black to move)
    Output : (batch,) float32 — predicted final margin (Black - White discs)

    Architecture summary:
      Stem   : Conv 3->64, BN, ReLU             — extracts local patterns
      Body   : 4 x ResBlock(64)                 — deepens without pooling
      Head   : GlobalAvgPool -> FC(64,32) -> FC(32,1)
    Total parameters: ~54 000 (fast on CPU at inference time)
    """

    def __init__(self, channels=64, n_blocks=4):
        super().__init__()

        # Stem: map 3 input channels to `channels` feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Body: stack of residual blocks, all keeping 8x8 spatial size
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        # Head: global-average-pool then two FC layers
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # (batch, channels, 1, 1)
            nn.Flatten(),                   # (batch, channels)
            nn.Linear(channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),              # Scalar margin prediction
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x.squeeze(1)    # (batch,) not (batch, 1)

# ---------------------------------------------------------------------------
# 5. Training scenario (mirrors TinyCNN_Sport's createScenario / train pattern)
# ---------------------------------------------------------------------------

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

class Scenario: pass

SEED = 42

def set_seed(seed=SEED):
    """Seed every random generator used by the script (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def seed_worker(worker_id):
    """
    DataLoader worker initialisation: derive each worker's Python and NumPy
    seeds from the (already seeded) PyTorch worker seed, so that the random
    dihedral augmentation is reproducible and different across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def createScenario(lr, train_ds, val_ds):
    """Initialise all training components and return a Scenario object."""
    S = Scenario()

    # Local (VM) and Drive output locations
    S.drive_folder = '/content/drive/MyDrive/thesis/Othello'
    S.log_folder   = '/content/logs'
    os.makedirs(S.log_folder, exist_ok=True)
    os.makedirs(S.drive_folder, exist_ok=True)

    S.CSVname     = os.path.join(S.log_folder,   'training_log.csv')
    S.csv_drive   = os.path.join(S.drive_folder, 'training_log.csv')
    S.preds_local = os.path.join(S.log_folder,   'val_predictions.csv')
    S.preds_drive = os.path.join(S.drive_folder, 'val_predictions.csv')

    S.epochs      = 60
    S.batch_size  = 256   # Large batch is fine for tiny 3x8x8 tensors

    # Seeded generator -> reproducible shuffling order of the training set
    g = torch.Generator()
    g.manual_seed(SEED)

    # num_workers=4 keeps the GPU fed with 52k+ steps/epoch;
    # with num_workers=2 the DataLoader becomes the bottleneck on 4M samples.
    S.train_batch = DataLoader(train_ds, batch_size=S.batch_size,
                               shuffle=True,  num_workers=4, pin_memory=True,
                               worker_init_fn=seed_worker, generator=g)
    S.valid_batch = DataLoader(val_ds,   batch_size=S.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True,
                               worker_init_fn=seed_worker)

    S.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {S.device}")

    S.model     = OthelloCNN(channels=64, n_blocks=4).to(S.device)
    S.loss_fn   = nn.MSELoss()
    S.lr        = lr
    S.optimizer = optim.SGD(S.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # LR drops at epoch 30 and 50: 0.01 -> 0.001 -> 0.0001
    S.scheduler = MultiStepLR(S.optimizer, milestones=[30, 50], gamma=0.1)

    # Gradient clipping (max global L2 norm): prevents the huge early updates,
    # caused by unnormalised targets in [-64, +64], that can push every ReLU
    # of the head into the negative region ("dead ReLU" collapse).
    S.max_grad_norm = 1.0

    # Collapse detector: if the std of the validation predictions falls
    # below this value (in discs), the network is predicting ~a constant.
    S.collapse_std = 1.0

    # Checkpoint paths (Drive + local VM copy). New file name, so that the
    # original benchmark model (othello_cnn.pth) is never overwritten.
    S.ckpt_local = '/content/othello_cnn_v2.pth'
    S.ckpt_drive = os.path.join(S.drive_folder, 'othello_cnn_v2.pth')
    S.ckpt_every = 5   # Save model + CSV log every N epochs

    return S


def compute_val_metrics(preds, targets):
    """
    Correlation metrics on the full validation set.
      pearson  : linear correlation between predicted and true margin
      spearman : rank correlation (monotonic agreement)
      r2       : coefficient of determination, 1 - SS_res / SS_tot
      sign_acc : share of decisive positions (true margin != 0) whose
                 predicted sign matches the actual winner
      pred_std : std of predictions (near 0 means the network has collapsed
                 to a constant output)
    """
    p = preds.astype(np.float64)
    y = targets.astype(np.float64)

    pred_std = float(p.std())
    if pred_std > 0:
        pearson     = float(np.corrcoef(p, y)[0, 1])
        spearman, _ = spearmanr(p, y)
        spearman    = float(spearman)
    else:                                   # constant output: undefined
        pearson = spearman = float('nan')

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot

    decisive = y != 0
    sign_acc = float(np.mean(np.sign(p[decisive]) == np.sign(y[decisive])))

    return pearson, spearman, r2, sign_acc, pred_std


def train(S):
    """
    Training loop: mirrors TinyCNN_Sport's train().

    Per epoch it logs: learning rate, MSE loss and MAE (train/valid) and the
    validation correlation metrics of compute_val_metrics().  Every
    S.ckpt_every epochs (and at the last epoch) both the model weights and
    the CSV log are saved locally and copied to Drive, so a Colab
    disconnection loses at most S.ckpt_every epochs of work and of logs.

    Gradients are clipped to S.max_grad_norm to prevent dead-ReLU collapse;
    a warning is printed if the validation predictions become ~constant.
    The validation predictions of the last epoch are kept in
    S.last_val_preds / S.last_val_targets for the final dump.
    """
    PRINT_EVERY = 500   # Print a mid-epoch line every N training steps

    columns = ["epoch", "lr",
               "loss/train", "loss/valid", "mae/train", "mae/valid",
               "pearson/valid", "spearman/valid", "r2/valid",
               "sign_acc/valid", "pred_std/valid"]
    info = []

    for epoch in range(S.epochs):

        current_lr = S.optimizer.param_groups[0]['lr']   # LR used this epoch

        # --- Training phase ---
        S.model.train()
        train_loss = train_mae = 0.0
        step = 0
        for x, y in S.train_batch:
            x, y = x.to(S.device), y.to(S.device)
            S.optimizer.zero_grad()
            preds = S.model(x)
            loss  = S.loss_fn(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(S.model.parameters(), S.max_grad_norm)
            S.optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_mae  += (preds - y).abs().sum().item()
            step += 1
            # Mid-epoch progress: running loss so far divided by samples seen
            if step % PRINT_EVERY == 0:
                samples_so_far = step * S.batch_size
                print(f"  ep {epoch+1:>2} step {step:>5}  "
                      f"running tLoss {train_loss/samples_so_far:7.2f}  "
                      f"tMAE {train_mae/samples_so_far:5.2f}", flush=True)

        # --- Validation phase (predictions kept for correlation metrics) ---
        S.model.eval()
        val_loss = val_mae = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in S.valid_batch:
                x, y  = x.to(S.device), y.to(S.device)
                preds = S.model(x)
                loss  = S.loss_fn(preds, y)
                val_loss += loss.item() * x.size(0)
                val_mae  += (preds - y).abs().sum().item()
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        S.scheduler.step()

        n_tr = len(S.train_batch.dataset)
        n_va = len(S.valid_batch.dataset)
        train_loss /= n_tr;  train_mae /= n_tr
        val_loss   /= n_va;  val_mae   /= n_va

        p_val = torch.cat(all_preds).numpy()
        y_val = torch.cat(all_targets).numpy()
        pearson, spearman, r2, sign_acc, pred_std = compute_val_metrics(p_val, y_val)
        S.last_val_preds, S.last_val_targets = p_val, y_val

        print(f"{epoch+1:>3}/{S.epochs}  lr {current_lr:.0e}  "
              f"tLoss {train_loss:7.2f}  vLoss {val_loss:7.2f}  "
              f"tMAE {train_mae:5.2f}  vMAE {val_mae:5.2f}  "
              f"r {pearson:.3f}  rho {spearman:.3f}  R2 {r2:.3f}  "
              f"signAcc {sign_acc:.3f}  predStd {pred_std:5.2f}", flush=True)

        if pred_std < S.collapse_std:
            print(f"  [WARNING] validation predictions nearly constant "
                  f"(std = {pred_std:.3f} discs): possible dead-ReLU collapse",
                  flush=True)

        info.append([epoch+1, current_lr, train_loss, val_loss, train_mae, val_mae,
                     pearson, spearman, r2, sign_acc, pred_std])

        # --- Periodic checkpoint: model weights + CSV log, local and Drive ---
        if (epoch + 1) % S.ckpt_every == 0 or (epoch + 1) == S.epochs:
            torch.save(S.model.state_dict(), S.ckpt_local)
            shutil.copy(S.ckpt_local, S.ckpt_drive)
            pd.DataFrame(info, columns=columns).to_csv(S.CSVname, index=False)
            shutil.copy(S.CSVname, S.csv_drive)
            print(f"  [checkpoint] epoch {epoch+1}: model and log saved to Drive",
                  flush=True)

    print(f"\nTraining log saved to {S.csv_drive}")


def dump_val_predictions(S, val_meta):
    """
    Save the final model's prediction for every validation position:
      game_id        : index of the game within the validation set
      n_moves_played : moves already played in the position (>= 10)
      y_true, y_pred : true and predicted final margin (Black - White)
    Row order matches val_data (the validation DataLoader does not shuffle).
    """
    game_ids, n_moves = zip(*val_meta)
    df = pd.DataFrame({
        "game_id":        game_ids,
        "n_moves_played": n_moves,
        "y_true":         S.last_val_targets,
        "y_pred":         np.round(S.last_val_preds, 3),
    })
    df.to_csv(S.preds_local, index=False)
    shutil.copy(S.preds_local, S.preds_drive)
    print(f"Validation predictions ({len(df)} rows) saved to {S.preds_drive}")

# ---------------------------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Seed everything (split, weight init, shuffling, augmentation) ---
    set_seed(SEED)

    # --- Parse WTB files (grouped by game) ---
    wtb_folder = '/content/wtb_files'
    all_games  = parse_all_wtb(wtb_folder)

    if len(all_games) == 0:
        raise RuntimeError("No games extracted — check WTB files.")

    # --- Game-level train/validation split (80/20) ---
    # The split is performed on GAMES, before they are expanded into
    # positions: all positions of a given game fall entirely in the
    # training set or entirely in the validation set (no leakage).
    random.shuffle(all_games)
    split       = int(0.8 * len(all_games))
    train_games = all_games[:split]
    val_games   = all_games[split:]

    # --- Expand games into flat lists of (tensor, margin) positions ---
    train_data = [pos for game in train_games for pos in game]
    val_data   = [pos for game in val_games   for pos in game]

    # Metadata of each validation position, in the same order as val_data:
    # (game index in the validation set, moves already played).
    # Positions are recorded from the 11th move on, one per non-pass move,
    # so the i-th position of a game has 10 + i moves already played.
    val_meta = [(gi, 10 + mi)
                for gi, game in enumerate(val_games)
                for mi in range(len(game))]

    print(f"Train: {len(train_games)} games -> {len(train_data)} positions | "
          f"Val: {len(val_games)} games -> {len(val_data)} positions")

    train_ds = OthelloDataset(train_data, augment=True)
    val_ds   = OthelloDataset(val_data,   augment=False)

    # --- Train (model + CSV log saved to Drive every S.ckpt_every epochs) ---
    S = createScenario(lr=1e-2, train_ds=train_ds, val_ds=val_ds)
    train(S)
    print(f"Model weights saved to Drive: {S.ckpt_drive}")

    # --- Final dump of the validation predictions (for the thesis figures) ---
    dump_val_predictions(S, val_meta)

    # --- Quick sanity check: 5 random validation positions ---
    for i in random.sample(range(len(val_data)), 5):
        print(f"  Predicted margin: {S.last_val_preds[i]:+.1f}  |  "
              f"True margin: {S.last_val_targets[i]:+.0f}")