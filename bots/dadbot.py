"""
DadBot v4 -- Parallel Cython-accelerated Monte Carlo 2-ply evaluation
with risk heuristics and performance tiers.

Uses the crossplay engine's compiled C extension (gaddag_accel.pyd) for
blazing-fast opponent simulation with multiprocessing parallelism.

Architecture:
  - Persistent worker pool initialized on first pick_move()
  - Each worker loads GADDAG + dictionary once (30MB, cached in process)
  - Per-move: serialize grid + blank set, fan out candidates to workers
  - Each worker: reconstruct board, place candidate, create BoardContext,
    run K MC sims with early stopping, return avg_opp
  - Main process: aggregate results, add SuperLeaves + positional adj,
    pick best

Evaluation modes:
  - Mid-game (bag > 8): Parallel MC 2-ply + SuperLeaves + positional adj
    + blank correction + bingo probability + exchange screening
  - Near-endgame (bag 1-8): Hybrid -- exhaustive 3-ply for bag-emptying
    moves, parity-adjusted 1-ply for non-emptying moves
  - Endgame (bag=0): Deterministic minimax (opponent rack known exactly)

Performance tiers (BOT_TIER env var):
  - blitz:    ~1s/move   (N=7,  K=150,  SE=1.5, min_sims=20)
  - fast:     ~3s/move   (N=15, K=400,  SE=1.2, min_sims=50)  [default]
  - standard: ~10s/move  (N=30, K=1500, SE=0.8, min_sims=80)
  - deep:     ~30s/move  (N=35, K=2000, SE=0.5, min_sims=100)

Falls back to sequential Python if Cython extension is unavailable.
"""

import os
import sys
import random
import pickle
import time
from math import comb
from collections import Counter
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

from bots.base_engine import BaseEngine, get_legal_moves

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CROSSPLAY_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'crossplay',
))
_TOURNAMENT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..',
))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
sys.path.insert(0, _TOURNAMENT_DIR)
from engine.config import (
    TILE_VALUES, TILE_DISTRIBUTION, BONUS_SQUARES,
    VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE, BOARD_SIZE,
)

# ---------------------------------------------------------------------------
# Performance tier system
# ---------------------------------------------------------------------------
# Calibrated for real throughput ~1600 sims/sec (8 workers under tournament load)
TIERS = {
    'blitz': {
        'N_CANDIDATES': 7,
        'K_SIMS': 150,
        'ES_SE_THRESHOLD': 1.5,
        'ES_MIN_SIMS': 20,
        'NEAR_ENDGAME_TIME': 3.0,
        'EXCHANGE_EVAL': False,
        'MC_SKIP_MARGIN': 10.0,
    },
    'fast': {
        'N_CANDIDATES': 15,
        'K_SIMS': 400,
        'ES_SE_THRESHOLD': 1.2,
        'ES_MIN_SIMS': 50,
        'NEAR_ENDGAME_TIME': 5.0,
        'EXCHANGE_EVAL': True,
        'MC_SKIP_MARGIN': 8.0,
    },
    'standard': {
        'N_CANDIDATES': 30,
        'K_SIMS': 1500,
        'ES_SE_THRESHOLD': 0.8,
        'ES_MIN_SIMS': 80,
        'NEAR_ENDGAME_TIME': 15.0,
        'EXCHANGE_EVAL': True,
    },
    'deep': {
        'N_CANDIDATES': 35,
        'K_SIMS': 2000,
        'ES_SE_THRESHOLD': 0.5,
        'ES_MIN_SIMS': 100,
        'NEAR_ENDGAME_TIME': 15.0,
        'EXCHANGE_EVAL': True,
    },
}

# Fixed MC parameters
ES_CHECK_EVERY = 10         # Check convergence every N sims

# Dynamic worker count: cpu_threads - 3 (reserve for OS + opponent bot + main).
# 9 workers on 12-thread machine tested ~7% faster than 7 workers with no
# degradation on opponent throughput. Override with MC_WORKERS env var.
_mc_workers_env = os.environ.get('MC_WORKERS')
if _mc_workers_env:
    MC_WORKERS = int(_mc_workers_env)
else:
    MC_WORKERS = max(1, os.cpu_count() - 3)  # e.g. 12 threads -> 9 workers

# Exchange parameters
EXCHANGE_EQUITY_THRESHOLD = 35.0  # consider exchange if best 1-ply < this
EXCHANGE_TOP_CANDIDATES = 5       # exchange options to evaluate
EXCHANGE_QUICK_MC = 200           # quick MC sims per exchange option

# Positional evaluation constants
RISK_PENALTIES = {'3W': 8.0, '2W': 3.0, '3L': 1.5, '2L': 0.5}
MC_POSITIONAL_DAMPEN = 0.5

# Multiplier when a bonus square is attackable from both H and V directions
DUAL_DIRECTION_MULT = 1.5

# Bag parity penalty table (from crossplay engine)
_PARITY_P_OPP_EMPTIES = {
    1: 0.97, 2: 0.94, 3: 0.88, 4: 0.78,
    5: 0.62, 6: 0.40, 7: 0.18,
}
_PARITY_STRUCTURAL_ADV = 10.0

# Pre-computed tables
_TV = [0] * 26
for _ch, _val in TILE_VALUES.items():
    if _ch != '?':
        _TV[ord(_ch) - 65] = _val

_BONUS = [[(1, 1)] * 15 for _ in range(15)]
for (_r1, _c1), _btype in BONUS_SQUARES.items():
    _r0, _c0 = _r1 - 1, _c1 - 1
    if _btype == '2L':
        _BONUS[_r0][_c0] = (2, 1)
    elif _btype == '3L':
        _BONUS[_r0][_c0] = (3, 1)
    elif _btype == '2W':
        _BONUS[_r0][_c0] = (1, 2)
    elif _btype == '3W':
        _BONUS[_r0][_c0] = (1, 3)

# Pre-computed bonus square sets (1-indexed)
DLS_POSITIONS = frozenset((r, c) for (r, c), v in BONUS_SQUARES.items() if v == '2L')

# Hookability: how many 2-letter words can each letter form?
HOOKABILITY = {
    'A': 28, 'B': 6, 'C': 0, 'D': 7, 'E': 24, 'F': 5, 'G': 3, 'H': 10,
    'I': 18, 'J': 1, 'K': 3, 'L': 5, 'M': 12, 'N': 9, 'O': 27, 'P': 6,
    'Q': 1, 'R': 4, 'S': 8, 'T': 8, 'U': 9, 'V': 0, 'W': 5, 'X': 5,
    'Y': 7, 'Z': 1,
}

# High-value tiles and their 2-letter words (for DLS exposure)
HIGH_VALUE_2LETTER = {
    'J': ['JO'],
    'Q': ['QI'],
    'Z': ['ZA'],
    'X': ['AX', 'EX', 'OX', 'XI', 'XU'],
    'K': ['KA', 'KI', 'OK'],
}


# ---------------------------------------------------------------------------
# SuperLeaves table (crossplay engine's trained leave values)
# ---------------------------------------------------------------------------
_LEAVES_PATH = os.path.normpath(os.path.join(
    _TOURNAMENT_DIR, 'engine', 'data', 'deployed_leaves.pkl',
))
_leaves_table = None

# Bingo probability database (crossplay engine's precomputed data)
BINGO_WEIGHT = 0.5
EXPECTED_BINGO_SCORE = 77.0
_BINGO_DB_PATH = os.path.normpath(os.path.join(
    _TOURNAMENT_DIR, 'engine', 'data', 'leave_bingo_prod.pkl',
))
_bingo_db = None


def _load_leaves():
    global _leaves_table
    if _leaves_table is not None:
        return _leaves_table
    try:
        with open(_LEAVES_PATH, 'rb') as f:
            _leaves_table = pickle.load(f)
    except Exception:
        _leaves_table = {}
    return _leaves_table


def _load_bingo_db():
    global _bingo_db
    if _bingo_db is not None:
        return _bingo_db
    try:
        with open(_BINGO_DB_PATH, 'rb') as f:
            _bingo_db = pickle.load(f)
    except Exception:
        _bingo_db = {}
    return _bingo_db


def _leave_value(leave_str, bag_empty=False):
    """Evaluate leave quality using SuperLeaves (mid-game) or formula (endgame)."""
    if not leave_str or leave_str == '-':
        return 0.0

    leave_str = leave_str.upper()

    if not bag_empty:
        table = _load_leaves()
        key = tuple(sorted(leave_str))
        val = table.get(key)
        if val is not None:
            return val

    base = _formula_leave(leave_str, bag_empty)

    # Add bingo probability bonus (only when bag has tiles and leave < 7)
    if not bag_empty and len(leave_str) < 7:
        bingo_db = _load_bingo_db()
        leave_key = tuple(sorted(leave_str))
        bingo_prob = bingo_db.get(leave_key, 0.0)
        base += BINGO_WEIGHT * bingo_prob * EXPECTED_BINGO_SCORE

    return base


def _formula_leave(leave_str, bag_empty=False):
    """Simple leave formula when SuperLeaves lookup misses."""
    if not leave_str:
        return 0.0

    value = 0.0
    tiles = list(leave_str.upper())
    n = len(tiles)

    vowels = sum(1 for t in tiles if t in 'AEIOU')

    if n >= 2:
        ratio = vowels / n if n > 0 else 0.5
        if ratio < 0.2 or ratio > 0.7:
            value -= 5.0
        elif ratio < 0.3 or ratio > 0.6:
            value -= 2.0

    counts = Counter(tiles)
    for letter, cnt in counts.items():
        if letter != '?' and cnt >= 2:
            value -= (cnt - 1) * 3.0

    blanks = counts.get('?', 0)
    value += blanks * 15.0

    s_count = counts.get('S', 0)
    value += min(s_count, 2) * 8.0

    for t in tiles:
        if t in ('Q', 'Z', 'X', 'J'):
            value -= 5.0
        elif t in ('V', 'W', 'K'):
            value -= 2.0

    return value


# ---------------------------------------------------------------------------
# Main-process resources
# ---------------------------------------------------------------------------
_gdata_bytes = None
_word_set = None


def _ensure_resources():
    global _gdata_bytes, _word_set
    if _gdata_bytes is not None:
        return
    from engine.gaddag import get_gaddag
    _gdata_bytes = bytes(get_gaddag()._data)
    from engine.dictionary import get_dictionary
    _word_set = get_dictionary()._words


def _get_accel():
    if _CROSSPLAY_DIR not in sys.path:
        sys.path.insert(0, _CROSSPLAY_DIR)
    try:
        import gaddag_accel
        return gaddag_accel
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Unseen tile pool
# ---------------------------------------------------------------------------
def _compute_unseen(grid, my_rack, blanks_on_board):
    """Compute unseen tiles from grid (0-indexed) + rack + blanks."""
    counts = dict(TILE_DISTRIBUTION)
    bb_1idx = {(r, c) for r, c, _ in (blanks_on_board or [])}

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            tile = grid[r][c]
            if tile is not None:
                if (r + 1, c + 1) in bb_1idx:
                    counts['?'] = counts.get('?', 0) - 1
                else:
                    counts[tile] = counts.get(tile, 0) - 1

    for ch in my_rack.upper():
        counts[ch] = counts.get(ch, 0) - 1

    pool = []
    for letter, cnt in counts.items():
        for _ in range(max(0, cnt)):
            pool.append(letter)
    return pool


# ---------------------------------------------------------------------------
# 1-ply equity ranking (score + leave value)
# ---------------------------------------------------------------------------
def _rank_by_equity(moves, bag_tiles):
    """Sort moves by 1-ply equity = score + leave_value. Returns sorted list."""
    bag_empty = bag_tiles <= RACK_SIZE
    ranked = []
    for m in moves:
        leave = m.get('leave', '')
        lv = _leave_value(leave, bag_empty=bag_empty) if bag_tiles > 0 else 0.0
        ranked.append((m, m['score'] + lv, lv))
    ranked.sort(key=lambda x: -x[1])
    return ranked


# ===================================================================
# POSITIONAL ADJUSTMENT FUNCTIONS (main process, <25ms total)
# ===================================================================

def _get_new_positions(grid, move):
    """Get positions of newly placed tiles (1-indexed). Returns list of (r, c, letter)."""
    word = move['word']
    row, col = move['row'], move['col']
    horizontal = move['direction'] == 'H'
    positions = []
    for i, letter in enumerate(word):
        if horizontal:
            r, c = row, col + i
        else:
            r, c = row + i, col
        if 1 <= r <= 15 and 1 <= c <= 15 and grid[r - 1][c - 1] is None:
            positions.append((r, c, letter))
    return positions


def _get_word_positions(move):
    """Get all positions the word occupies (1-indexed). Returns set of (r, c)."""
    word = move['word']
    row, col = move['row'], move['col']
    horizontal = move['direction'] == 'H'
    positions = set()
    for i in range(len(word)):
        if horizontal:
            positions.add((row, col + i))
        else:
            positions.add((row + i, col))
    return positions


def _was_already_reachable(grid, pr, pc, word_positions):
    """Check if a square was already adjacent to an existing tile before our move."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ar, ac = pr + dr, pc + dc
        if 1 <= ar <= 15 and 1 <= ac <= 15:
            if (ar, ac) not in word_positions and grid[ar - 1][ac - 1] is not None:
                return True
    return False


def _direction_count(grid, br, bc, word_positions):
    """Count how many axes (H, V) a bonus square can be exploited from.

    A bonus square is exploitable on an axis if there's an adjacent tile
    along that axis (provides a hook for word formation).
    Returns 1 or 2.
    """
    def _has_adj(r, c):
        if 1 <= r <= 15 and 1 <= c <= 15:
            if (r, c) in word_positions or grid[r - 1][c - 1] is not None:
                return True
        return False

    h_ok = _has_adj(br, bc - 1) or _has_adj(br, bc + 1)
    v_ok = _has_adj(br - 1, bc) or _has_adj(br + 1, bc)
    return (1 if h_ok else 0) + (1 if v_ok else 0) or 1


def _compute_risk(grid, move):
    """Compute risk penalty for newly opened bonus squares.

    Scans perpendicular neighbors of each tile in the word for bonus squares
    that become newly reachable. Returns risk_penalty (positive float).
    """
    word_positions = _get_word_positions(move)
    word = move['word']
    row, col = move['row'], move['col']
    horizontal = move['direction'] == 'H'
    word_len = len(word)

    risk_penalty = 0.0
    seen_opened = set()

    # Perpendicular adjacency for each tile in word
    for i in range(word_len):
        if horizontal:
            r, c = row, col + i
            adj = [(r - 1, c), (r + 1, c)]
        else:
            r, c = row + i, col
            adj = [(r, c - 1), (r, c + 1)]

        for nr, nc in adj:
            if not (1 <= nr <= 15 and 1 <= nc <= 15):
                continue
            if (nr, nc) in word_positions or (nr, nc) in seen_opened:
                continue
            if grid[nr - 1][nc - 1] is not None:
                continue
            bonus_type = BONUS_SQUARES.get((nr, nc))
            if bonus_type and not _was_already_reachable(grid, nr, nc, word_positions):
                seen_opened.add((nr, nc))
                dirs = _direction_count(grid, nr, nc, word_positions)
                mult = DUAL_DIRECTION_MULT if dirs >= 2 else 1.0
                risk_penalty += RISK_PENALTIES.get(bonus_type, 0) * mult

    return risk_penalty


def _compute_dls_exposure(grid, move, unseen_pool=None):
    """Penalty for tiles adjacent to open DLS when opponent might have HVTs."""
    new_positions = _get_new_positions(grid, move)
    horizontal = move['direction'] == 'H'

    total_penalty = 0.0
    unseen_counts = Counter(unseen_pool) if unseen_pool else None
    total_unseen = len(unseen_pool) if unseen_pool else 0

    for r, c, letter in new_positions:
        adj_positions = []
        if horizontal:
            for dist in [1, 2]:
                adj_positions.append((r - dist, c))
                adj_positions.append((r + dist, c))
            adj_positions.append((r, c - 1))
            adj_positions.append((r, c + 1))
        else:
            for dist in [1, 2]:
                adj_positions.append((r, c - dist))
                adj_positions.append((r, c + dist))
            adj_positions.append((r - 1, c))
            adj_positions.append((r + 1, c))

        for ar, ac in adj_positions:
            if not (1 <= ar <= 15 and 1 <= ac <= 15):
                continue
            if (ar, ac) not in DLS_POSITIONS or grid[ar - 1][ac - 1] is not None:
                continue

            max_damage = 0
            worst_tile = None
            for hv_tile, hv_words in HIGH_VALUE_2LETTER.items():
                tile_val = TILE_VALUES.get(hv_tile, 0)
                for w in hv_words:
                    if letter in w and len(w) == 2:
                        damage = tile_val * 2 + TILE_VALUES.get(letter, 0)
                        if damage > max_damage:
                            max_damage = damage
                            worst_tile = hv_tile

            if max_damage > 0:
                prob = 0.15
                if unseen_counts and total_unseen > 0:
                    tile_count = unseen_counts.get(worst_tile, 0)
                    if tile_count > 0:
                        prob = 1 - ((total_unseen - tile_count) / total_unseen) ** 7

                dist = abs(ar - r) + abs(ac - c)
                dist_factor = 1.0 if dist == 1 else 0.5
                total_penalty += max_damage * prob * dist_factor
            elif HOOKABILITY.get(letter, 0) > 10:
                total_penalty += HOOKABILITY[letter] * 0.1

    return -total_penalty


def _compute_positional_adj(grid, move, unseen_pool, bag_size):
    """Positional adjustment: risk penalty + DLS exposure."""
    risk = _compute_risk(grid, move)
    dls = _compute_dls_exposure(grid, move, unseen_pool)
    return -risk + dls


# ===================================================================
# WORKER PROCESS CODE (runs in separate processes)
# ===================================================================

_w_gdata_bytes = None
_w_word_set = None
_w_accel = None
_w_tv = None
_w_bonus = None


def _worker_init(crossplay_dir, tournament_dir):
    """Initialize worker: load GADDAG + dictionary + Cython extension."""
    global _w_gdata_bytes, _w_word_set, _w_accel, _w_tv, _w_bonus

    if tournament_dir not in sys.path:
        sys.path.insert(0, tournament_dir)
    if crossplay_dir not in sys.path:
        sys.path.insert(0, crossplay_dir)

    from engine.gaddag import get_gaddag
    _w_gdata_bytes = bytes(get_gaddag()._data)

    from engine.dictionary import get_dictionary
    _w_word_set = get_dictionary()._words

    try:
        import gaddag_accel
        _w_accel = gaddag_accel
    except ImportError:
        _w_accel = None

    from engine.config import TILE_VALUES as TV, BONUS_SQUARES as BS
    _w_tv = [0] * 26
    for ch, val in TV.items():
        if ch != '?':
            _w_tv[ord(ch) - 65] = val

    _w_bonus = [[(1, 1)] * 15 for _ in range(15)]
    for (r1, c1), btype in BS.items():
        r0, c0 = r1 - 1, c1 - 1
        if btype == '2L':
            _w_bonus[r0][c0] = (2, 1)
        elif btype == '3L':
            _w_bonus[r0][c0] = (3, 1)
        elif btype == '2W':
            _w_bonus[r0][c0] = (1, 2)
        elif btype == '3W':
            _w_bonus[r0][c0] = (1, 3)


def _worker_eval_candidate(args):
    """Worker function: evaluate one candidate with K sims.

    Args tuple: (grid, bb_set_list, move, unseen_pool, k_sims, seed,
                  es_min_sims, es_check_every, es_se_threshold)
    """
    (grid, bb_set_list, move, unseen_pool, k_sims, seed,
     es_min_sims, es_check_every, es_se_threshold) = args

    random.seed(seed)

    from engine.config import VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE

    bb_set = set(bb_set_list)

    from engine.board import Board
    board = Board()
    for r in range(15):
        for c in range(15):
            if grid[r][c] is not None:
                board._grid[r][c] = grid[r][c]

    horizontal = move['direction'] == 'H'
    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    blanks_used = move.get('blanks_used', [])
    if blanks_used:
        for bi in blanks_used:
            if horizontal:
                br, bc = move['row'] - 1, move['col'] - 1 + bi
            else:
                br, bc = move['row'] - 1 + bi, move['col'] - 1
            bb_set.add((br, bc))

    post_grid = board._grid
    pool_size = len(unseen_pool)
    rack_draw = min(RACK_SIZE, pool_size)

    use_cython = (_w_accel is not None and
                  hasattr(_w_accel, 'prepare_board_context') and
                  _w_gdata_bytes is not None)

    ctx = None
    if use_cython:
        ctx = _w_accel.prepare_board_context(
            post_grid, _w_gdata_bytes, bb_set,
            _w_word_set, VALID_TWO_LETTER,
            _w_tv, _w_bonus, BINGO_BONUS, RACK_SIZE,
        )

    running_sum = 0.0
    running_sum_sq = 0.0
    n_sims = 0

    for sim_i in range(k_sims):
        if n_sims >= es_min_sims and n_sims % es_check_every == 0:
            variance = (running_sum_sq / n_sims) - (running_sum / n_sims) ** 2
            if variance > 0:
                se = (variance / n_sims) ** 0.5
                if se < es_se_threshold:
                    break

        opp_rack = ''.join(random.sample(unseen_pool, rack_draw))

        if use_cython:
            opp_score, _, _, _, _ = _w_accel.find_best_score_c(ctx, opp_rack)
        else:
            blanks_1idx = [(r + 1, c + 1, '') for r, c in bb_set]
            opp_moves = get_legal_moves(board, opp_rack, blanks_1idx)
            opp_score = opp_moves[0]['score'] if opp_moves else 0

        running_sum += opp_score
        running_sum_sq += opp_score * opp_score
        n_sims += 1

    board.undo_move(placed)

    avg_opp = running_sum / n_sims if n_sims > 0 else 0.0
    return {
        'word': move['word'],
        'row': move['row'],
        'col': move['col'],
        'direction': move['direction'],
        'avg_opp': avg_opp,
        'n_sims': n_sims,
        'sum': running_sum,
        'sum_sq': running_sum_sq,
    }


# ===================================================================
# Endgame minimax worker (bag=0, parallel)
# ===================================================================

def _worker_eval_endgame(args):
    """Worker: deterministic minimax for one move when bag=0.

    Opponent rack is known exactly (unseen tiles = their rack).
    Evaluates: our_score - opponent_best_response.

    Args tuple: (grid, bb_set_list, move, opp_rack)
    """
    grid, bb_set_list, move, opp_rack = args

    from engine.config import VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE
    from engine.board import Board

    bb_set = set(bb_set_list)

    board = Board()
    for r in range(15):
        for c in range(15):
            if grid[r][c] is not None:
                board._grid[r][c] = grid[r][c]

    horizontal = move['direction'] == 'H'
    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    # Update blank set with blanks from this move
    move_bb = set(bb_set)
    for bi in move.get('blanks_used', []):
        if horizontal:
            move_bb.add((move['row'] - 1, move['col'] - 1 + bi))
        else:
            move_bb.add((move['row'] - 1 + bi, move['col'] - 1))

    # Find opponent's best response
    use_cython = (_w_accel is not None and
                  hasattr(_w_accel, 'prepare_board_context') and
                  _w_gdata_bytes is not None)

    if use_cython:
        ctx = _w_accel.prepare_board_context(
            board._grid, _w_gdata_bytes, move_bb,
            _w_word_set, VALID_TWO_LETTER,
            _w_tv, _w_bonus, BINGO_BONUS, RACK_SIZE,
        )
        opp_score, _, _, _, _ = _w_accel.find_best_score_c(ctx, opp_rack)
    else:
        blanks_1idx = [(r + 1, c + 1, '') for r, c in move_bb]
        opp_moves = get_legal_moves(board, opp_rack, blanks_1idx)
        opp_score = opp_moves[0]['score'] if opp_moves else 0

    board.undo_move(placed)
    equity = move['score'] - opp_score

    return {
        'word': move['word'],
        'row': move['row'],
        'col': move['col'],
        'direction': move['direction'],
        'score': move['score'],
        'equity': equity,
    }


# ===================================================================
# Near-endgame worker (bag 1-8, parallel exhaustive 3-ply)
# ===================================================================

def _worker_eval_near_endgame(args):
    """Worker: exhaustive 3-ply for one bag-emptying move.

    Iterates over all C(unseen, rack_size) opponent rack combinations.
    For each: our_score - opp_best_response + our_follow_up.

    Args tuple: (grid, bb_set_list, move, unseen_pool, rack)
    """
    grid, bb_set_list, move, unseen_pool, rack = args

    from engine.config import VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE
    from engine.board import Board

    bb_set = set(bb_set_list)

    board = Board()
    for r in range(15):
        for c in range(15):
            if grid[r][c] is not None:
                board._grid[r][c] = grid[r][c]

    # Place our move (ply 1)
    horizontal = move['direction'] == 'H'
    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    move_bb = set(bb_set)
    for bi in move.get('blanks_used', []):
        if horizontal:
            move_bb.add((move['row'] - 1, move['col'] - 1 + bi))
        else:
            move_bb.add((move['row'] - 1 + bi, move['col'] - 1))

    # Compute our leave (tiles remaining after playing this move)
    tiles_used = move.get('tiles_used', list(move['word']))
    rack_list = list(rack.upper())
    for t in tiles_used:
        if t in rack_list:
            rack_list.remove(t)
        elif '?' in rack_list:
            rack_list.remove('?')
    your_leave = ''.join(rack_list)

    use_cython = (_w_accel is not None and
                  hasattr(_w_accel, 'prepare_board_context') and
                  _w_gdata_bytes is not None)

    # Prepare board context for opponent response (ply 2)
    ctx_ply2 = None
    if use_cython:
        ctx_ply2 = _w_accel.prepare_board_context(
            board._grid, _w_gdata_bytes, move_bb,
            _w_word_set, VALID_TWO_LETTER,
            _w_tv, _w_bonus, BINGO_BONUS, RACK_SIZE,
        )

    unseen_str_list = list(unseen_pool)
    unseen_count = len(unseen_pool)
    opp_rack_size = min(RACK_SIZE, unseen_count)
    net_scores = []

    for combo_indices in combinations(range(unseen_count), opp_rack_size):
        opp_rack = ''.join(unseen_str_list[i] for i in combo_indices)
        drawn_indices = set(range(unseen_count)) - set(combo_indices)
        drawn_tiles = ''.join(unseen_str_list[i] for i in drawn_indices)
        your_full_rack = your_leave + drawn_tiles

        # Ply 2: opponent's best response
        if use_cython:
            opp_score, opp_word, opp_r, opp_c, opp_d = _w_accel.find_best_score_c(
                ctx_ply2, opp_rack)
        else:
            opp_score = 0
            opp_ms = get_legal_moves(board, opp_rack,
                                     [(r + 1, c + 1, '') for r, c in move_bb])
            if opp_ms:
                opp_score = opp_ms[0]['score']
                opp_word = opp_ms[0]['word']
                opp_r = opp_ms[0]['row']
                opp_c = opp_ms[0]['col']
                opp_d = opp_ms[0]['direction']

        # Ply 3: our follow-up after opponent's move
        your_resp_score = 0
        if opp_score > 0:
            opp_horiz = opp_d == 'H' if isinstance(opp_d, str) else opp_d
            placed_2 = board.place_move(opp_word, opp_r, opp_c, opp_horiz)

            if use_cython:
                ctx_ply3 = _w_accel.prepare_board_context(
                    board._grid, _w_gdata_bytes, move_bb,
                    _w_word_set, VALID_TWO_LETTER,
                    _w_tv, _w_bonus, BINGO_BONUS, RACK_SIZE,
                )
                your_resp_score, _, _, _, _ = _w_accel.find_best_score_c(
                    ctx_ply3, your_full_rack)
            else:
                resp_ms = get_legal_moves(board, your_full_rack,
                                          [(r + 1, c + 1, '') for r, c in move_bb])
                if resp_ms:
                    your_resp_score = resp_ms[0]['score']

            board.undo_move(placed_2)
        else:
            # Opponent passed -- use ply2 board state
            if use_cython:
                your_resp_score, _, _, _, _ = _w_accel.find_best_score_c(
                    ctx_ply2, your_full_rack)
            else:
                resp_ms = get_legal_moves(board, your_full_rack,
                                          [(r + 1, c + 1, '') for r, c in move_bb])
                if resp_ms:
                    your_resp_score = resp_ms[0]['score']

        net = move['score'] - opp_score + your_resp_score
        net_scores.append(net)

    board.undo_move(placed)

    avg_net = sum(net_scores) / len(net_scores) if net_scores else float(move['score'])

    return {
        'word': move['word'],
        'row': move['row'],
        'col': move['col'],
        'direction': move['direction'],
        'score': move['score'],
        'avg_equity': avg_net,
    }


# ===================================================================
# Near-endgame hybrid evaluator (bag 1-8)
# ===================================================================

def _evaluate_near_endgame(board, rack, moves, unseen_pool, blanks_on_board,
                           time_budget=15.0):
    """Hybrid evaluation for bag 1-8. Parallel.

    Bag-emptying moves: parallel exhaustive 3-ply via worker pool.
    Non-emptying moves: parity-adjusted 1-ply equity (instant, main process).

    Returns best move.
    """
    unseen_count = len(unseen_pool)
    bag_size = max(0, unseen_count - RACK_SIZE)

    results = []

    ranked = _rank_by_equity(moves, bag_size)
    candidates = ranked[:25]

    bb_set_list = [(r - 1, c - 1) for r, c, _ in (blanks_on_board or [])]
    grid = [row[:] for row in board._grid]

    # PASS 1: non-emptying moves (instant -- parity-adjusted 1-ply)
    exhaust_cands = []
    for move, equity_1ply, leave_val in candidates:
        tiles_used = move.get('tiles_used', [])
        n_used = len(tiles_used)

        if n_used >= bag_size:
            exhaust_cands.append((move, equity_1ply, leave_val))
        else:
            n_draw = min(n_used, bag_size)
            bag_after = bag_size - n_draw
            parity_penalty = 0.0
            if 1 <= bag_after <= 7:
                p_opp = _PARITY_P_OPP_EMPTIES.get(bag_after, 0.0)
                parity_penalty = -p_opp * _PARITY_STRUCTURAL_ADV

            results.append((move, equity_1ply + parity_penalty))

    # PASS 2: bag-emptying moves -- parallel exhaustive 3-ply via workers
    if exhaust_cands:
        work = []
        for move, equity_1ply, leave_val in exhaust_cands:
            move_data = {
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'blanks_used': move.get('blanks_used', []),
                'tiles_used': move.get('tiles_used', list(move['word'])),
            }
            work.append((grid, bb_set_list, move_data, unseen_pool, rack))

        pool = _get_pool()
        futures = [pool.submit(_worker_eval_near_endgame, w) for w in work]

        t_start = time.perf_counter()
        for i, future in enumerate(futures):
            remaining = time_budget - (time.perf_counter() - t_start)
            if remaining <= 0:
                # Time's up -- use 1-ply fallback for remaining candidates
                move, equity_1ply, _ = exhaust_cands[i]
                results.append((move, equity_1ply))
                continue
            try:
                result = future.result(timeout=max(1, remaining))
                # Find the original move object for this result
                orig_move = exhaust_cands[i][0]
                results.append((orig_move, result['avg_equity']))
            except Exception:
                move, equity_1ply, _ = exhaust_cands[i]
                results.append((move, equity_1ply))

    if not results:
        return moves[0] if moves else None

    results.sort(key=lambda x: -x[1])
    return results[0][0]


# ===================================================================
# Blank correction factor (for MC opponent score adjustment)
# ===================================================================

def _blank_correction_factor(total_unseen, blanks_unseen):
    """Correction multiplier for MC opponent scores when blanks are capped at 2.

    Only needed when 3 blanks are unseen (Crossplay has 3 blanks).
    Returns multiplier to apply to avg_opp (1.0 = no correction).
    """
    if blanks_unseen <= 2 or total_unseen < 7:
        return 1.0

    RATIO_0v2 = 0.470
    RATIO_1v2 = 0.687
    RATIO_3v2 = 1.036

    draw = min(7, total_unseen)

    def p_draw_k(k):
        non = total_unseen - blanks_unseen
        if k > blanks_unseen or k > draw or (draw - k) > non:
            return 0.0
        return comb(blanks_unseen, k) * comb(non, draw - k) / comb(total_unseen, draw)

    p0 = p_draw_k(0)
    p1 = p_draw_k(1)
    p2 = p_draw_k(2)
    p3 = p_draw_k(3) if blanks_unseen >= 3 else 0.0

    e_true = p0 * RATIO_0v2 + p1 * RATIO_1v2 + p2 * 1.0 + p3 * RATIO_3v2

    blanks_removed = blanks_unseen - 2
    cap_total = total_unseen - blanks_removed
    cap_blanks = 2
    cap_draw = min(7, cap_total)

    if cap_total < 1 or cap_draw < 1:
        return 1.0

    cap_non = cap_total - cap_blanks

    def p_draw_k_capped(k):
        if k > cap_blanks or k > cap_draw or (cap_draw - k) > cap_non:
            return 0.0
        return comb(cap_blanks, k) * comb(cap_non, cap_draw - k) / comb(cap_total, cap_draw)

    cp0 = p_draw_k_capped(0)
    cp1 = p_draw_k_capped(1)
    cp2 = p_draw_k_capped(2)

    e_capped = cp0 * RATIO_0v2 + cp1 * RATIO_1v2 + cp2 * 1.0

    if e_capped <= 0:
        return 1.0

    return e_true / e_capped


# ===================================================================
# Exchange evaluation
# ===================================================================

def _generate_exchange_candidates(rack, unseen_pool):
    """Generate top exchange options sorted by expected new rack leave."""
    if len(unseen_pool) < RACK_SIZE:
        return []

    rack_list = list(rack.upper())
    options = []

    # Full exchange
    total = 0.0
    for _ in range(EXCHANGE_QUICK_MC):
        drawn = random.sample(unseen_pool, min(RACK_SIZE, len(unseen_pool)))
        total += _leave_value(''.join(drawn))
    options.append({
        'keep': '', 'dump': rack,
        'expected_leave': total / EXCHANGE_QUICK_MC,
    })

    # Partial exchanges: keep 1-4 tiles
    for keep_n in range(1, min(5, len(rack_list))):
        seen_keeps = set()
        for keep_combo in combinations(range(len(rack_list)), keep_n):
            keep = tuple(sorted(rack_list[i] for i in keep_combo))
            if keep in seen_keeps:
                continue
            seen_keeps.add(keep)

            keep_str = ''.join(keep)
            keep_lv = _leave_value(keep_str)
            if keep_lv < -5 and keep_n >= 3:
                continue

            draw_n = RACK_SIZE - keep_n
            total = 0.0
            for _ in range(EXCHANGE_QUICK_MC):
                drawn = random.sample(unseen_pool, min(draw_n, len(unseen_pool)))
                new_rack = list(keep) + drawn
                total += _leave_value(''.join(new_rack))
            avg_leave = total / EXCHANGE_QUICK_MC

            remaining = list(rack_list)
            for t in keep:
                remaining.remove(t)

            options.append({
                'keep': keep_str, 'dump': ''.join(remaining),
                'expected_leave': avg_leave,
            })

    options.sort(key=lambda x: -x['expected_leave'])
    return options[:EXCHANGE_TOP_CANDIDATES]


# ===================================================================
# DadBot class
# ===================================================================

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        print(f"  [DadBot] MC pool: {MC_WORKERS} workers "
              f"({os.cpu_count()} threads - 3 reserved)")
        _pool = ProcessPoolExecutor(
            max_workers=MC_WORKERS,
            initializer=_worker_init,
            initargs=(_CROSSPLAY_DIR, _TOURNAMENT_DIR),
        )
    return _pool


class DadBot(BaseEngine):

    def __init__(self):
        super().__init__()
        tier = os.environ.get('BOT_TIER', 'fast')
        self.config = TIERS.get(tier, TIERS['fast'])
        self.tier = tier

    @property
    def name(self):
        return "DadBot"

    def game_over(self, result, game_info):
        """Called when game ends. Pool kept alive for multi-game matches."""
        pass  # Workers persist -- GADDAG loaded once per pool lifetime

    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None

        _ensure_resources()
        _DIAG = os.environ.get('DADBOT_TIMING', '')

        bag_tiles = game_info.get('tiles_in_bag', 1)
        blanks_on_board = game_info.get('blanks_on_board', [])
        grid = [row[:] for row in board._grid]
        cfg = self.config

        # ---------------------------------------------------------------
        # Endgame: bag=0, deterministic minimax
        # ---------------------------------------------------------------
        if bag_tiles == 0:
            return self._endgame_pick(board, rack, moves, blanks_on_board, grid)

        # ---------------------------------------------------------------
        # Near-endgame: bag 1-8, hybrid evaluation
        # ---------------------------------------------------------------
        if 1 <= bag_tiles <= 8:
            unseen_pool = _compute_unseen(grid, rack, blanks_on_board)
            return _evaluate_near_endgame(
                board, rack, moves, unseen_pool, blanks_on_board,
                time_budget=cfg.get('NEAR_ENDGAME_TIME', 15.0))

        # ---------------------------------------------------------------
        # Mid-game: parallel MC 2-ply with SuperLeaves + positional adj
        # ---------------------------------------------------------------
        t_move_start = time.perf_counter()
        unseen_pool = _compute_unseen(grid, rack, blanks_on_board)

        # Rank candidates by 1-ply equity (score + leave)
        t0 = time.perf_counter()
        ranked = _rank_by_equity(moves, bag_tiles)
        n_cands = cfg['N_CANDIDATES']
        candidates = [(m, lv) for m, eq, lv in ranked[:n_cands]]
        t_rank = time.perf_counter() - t0

        # Compute positional adjustment for each candidate (main process, <25ms)
        t0 = time.perf_counter()
        pos_adjs = []
        for move, lv in candidates:
            pos_adj = _compute_positional_adj(grid, move, unseen_pool, bag_tiles)
            pos_adjs.append(pos_adj)
        t_posadj = time.perf_counter() - t0

        # MC skip: if top 1-ply candidate leads by a wide margin, skip MC
        mc_skip_margin = cfg.get('MC_SKIP_MARGIN', 0)
        if mc_skip_margin > 0 and len(candidates) >= 2:
            # Combine 1-ply equity + positional adj for skip comparison
            top_eq = ranked[0][1] + pos_adjs[0] * MC_POSITIONAL_DAMPEN
            second_eq = ranked[1][1] + pos_adjs[1] * MC_POSITIONAL_DAMPEN
            if top_eq - second_eq >= mc_skip_margin:
                return candidates[0][0]

        # Check if exchange should be considered
        t0 = time.perf_counter()
        best_1ply_equity = ranked[0][1] if ranked else 0
        exch_opts = None
        n_exch_combos = 0
        if (cfg.get('EXCHANGE_EVAL', True)
                and best_1ply_equity < EXCHANGE_EQUITY_THRESHOLD
                and bag_tiles >= RACK_SIZE):
            exch_opts = _generate_exchange_candidates(rack, unseen_pool)
            n_exch_combos = len(exch_opts) if exch_opts else 0
        t_exch = time.perf_counter() - t0

        # Build work items for MC (with tier-specific ES params)
        bb_set_list = [(r - 1, c - 1) for r, c, _ in (blanks_on_board or [])]
        k_sims = cfg['K_SIMS']
        es_se = cfg['ES_SE_THRESHOLD']
        es_min = cfg.get('ES_MIN_SIMS', 30)

        work = []
        for i, (move, lv) in enumerate(candidates):
            move_data = {
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'blanks_used': move.get('blanks_used', []),
                'tiles_used': move.get('tiles_used', list(move['word'])),
            }
            seed = random.randint(0, 2**31)
            work.append((grid, bb_set_list, move_data, unseen_pool,
                         k_sims, seed, es_min, ES_CHECK_EVERY, es_se))

        # Fan out to worker pool
        t0 = time.perf_counter()
        pool = _get_pool()
        futures = [pool.submit(_worker_eval_candidate, w) for w in work]

        # Compute blank correction factor
        blanks_in_unseen = sum(1 for t in unseen_pool if t == '?')
        blank_corr = _blank_correction_factor(len(unseen_pool), blanks_in_unseen)

        # Collect regular move results
        best_move = None
        best_total = float('-inf')
        bag_empty_flag = bag_tiles <= RACK_SIZE
        total_sims = 0

        for i, future in enumerate(futures):
            result = future.result(timeout=60)
            avg_opp = result['avg_opp'] * blank_corr  # Apply blank correction
            total_sims += result.get('n_sims', 0)
            move, leave_val = candidates[i]
            mc_equity = move['score'] - avg_opp

            leave = move.get('leave', '')
            lv = _leave_value(leave, bag_empty=bag_empty_flag) if bag_tiles > 0 else 0.0

            # Combine: MC equity + leave + damped positional adjustment
            total = mc_equity + lv + pos_adjs[i] * MC_POSITIONAL_DAMPEN

            if total > best_total:
                best_total = total
                best_move = move

        t_mc = time.perf_counter() - t0
        t_total = time.perf_counter() - t_move_start

        # Timing diagnostics
        if _DIAG:
            sims_per_sec = total_sims / t_mc if t_mc > 0 else 0
            print(f"  [TIMING] rank={t_rank*1000:.0f}ms "
                  f"posadj={t_posadj*1000:.0f}ms "
                  f"exch={t_exch*1000:.0f}ms({n_exch_combos}opts) "
                  f"MC={t_mc*1000:.0f}ms({total_sims}sims,{sims_per_sec:.0f}/s) "
                  f"total={t_total*1000:.0f}ms "
                  f"bag={bag_tiles} cands={len(candidates)} "
                  f"moves={len(moves)}")

        # Simple exchange threshold check (non-MC path)
        if exch_opts:
            exch_leave = exch_opts[0]['expected_leave']
            if exch_leave > best_1ply_equity and best_total < 0:
                return None

        return best_move

    def _endgame_pick(self, board, rack, moves, blanks_on_board, grid):
        """Deterministic endgame: opponent rack known exactly. Parallel.

        Exhaustive minimax over ALL legal moves -- no pruning. Each worker
        evaluates one move: our_score - opponent_best_response. Global time
        budget (180s) ensures we don't hang; if time runs out, best result
        so far is returned.
        """
        unseen = _compute_unseen(grid, rack, blanks_on_board)
        opp_rack = ''.join(unseen)

        bb_set_list = [(r - 1, c - 1) for r, c, _ in (blanks_on_board or [])]

        # Evaluate ALL legal moves (exhaustive minimax)
        work = []
        for move in moves:
            move_data = {
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'blanks_used': move.get('blanks_used', []),
            }
            work.append((grid, bb_set_list, move_data, opp_rack))

        # Fan out to worker pool
        pool = _get_pool()
        futures = [pool.submit(_worker_eval_endgame, w) for w in work]

        best_move = None
        best_equity = float('-inf')
        completed = 0
        timed_out = 0

        # Global time budget: 180s for all moves
        # (dense board: ~400 moves / 7 workers * ~2s each = ~114s typical)
        t_start = time.perf_counter()
        ENDGAME_BUDGET = 180.0

        for i, future in enumerate(futures):
            remaining = ENDGAME_BUDGET - (time.perf_counter() - t_start)
            if remaining <= 0:
                timed_out += len(futures) - i
                break
            try:
                result = future.result(timeout=max(2.0, remaining))
                completed += 1
            except Exception:
                timed_out += 1
                continue
            if result['equity'] > best_equity:
                best_equity = result['equity']
                best_move = moves[i]

        if timed_out:
            print(f"  [DadBot] Endgame: {completed}/{len(futures)} evaluated, "
                  f"{timed_out} timed out ({time.perf_counter() - t_start:.1f}s)")

        # Fallback: if nothing completed, play highest-scoring move
        if best_move is None:
            print("  [DadBot] Endgame: all workers failed, "
                  "falling back to top score")
            best_move = moves[0]

        return best_move
