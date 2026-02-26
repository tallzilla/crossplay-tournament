"""
DadBot -- Parallel Cython-accelerated Monte Carlo 2-ply evaluation.

Uses the crossplay engine's compiled C extension (gaddag_accel.pyd) for
blazing-fast opponent simulation with multiprocessing parallelism.

Architecture:
  - Persistent worker pool (8 workers) initialized on first pick_move()
  - Each worker loads GADDAG + dictionary once (30MB, cached in process)
  - Per-move: serialize grid + blank set, fan out candidates to workers
  - Each worker: reconstruct board, place candidate, create BoardContext,
    run K/num_workers MC sims with early stopping, return avg_opp
  - Main process: aggregate results, add SuperLeaves, pick best

Endgame (bag=0): deterministic minimax in main process -- opponent rack
is known exactly, no sampling needed.

Falls back to sequential Python if Cython extension is unavailable.
"""

import os
import sys
import random
import pickle
import time
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
# Configuration (imported in main process for endgame + leave eval)
# ---------------------------------------------------------------------------
sys.path.insert(0, _TOURNAMENT_DIR)
from engine.config import (
    TILE_VALUES, TILE_DISTRIBUTION, BONUS_SQUARES,
    VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE, BOARD_SIZE,
)

# MC parameters
N_CANDIDATES = 15       # Top N moves to evaluate with MC
K_SIMS = 300            # Total MC iterations per candidate (split across workers)
ES_MIN_SIMS = 30        # Min sims per worker before early stopping
ES_CHECK_EVERY = 10     # Check convergence every N sims
ES_SE_THRESHOLD = 1.5   # Stop when SE < this
MC_WORKERS = 8          # Parallel worker count
ENDGAME_FULL_SEARCH = True

# Pre-computed tables (main process, for endgame)
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

# ---------------------------------------------------------------------------
# SuperLeaves table
# ---------------------------------------------------------------------------
_LEAVES_PATH = os.path.normpath(os.path.join(
    _CROSSPLAY_DIR, 'superleaves', 'deployed_leaves.pkl',
))
_leaves_table = None


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


def _leave_value(leave_str):
    table = _load_leaves()
    key = tuple(sorted(leave_str.upper()))
    return table.get(key, 0.0)


# ---------------------------------------------------------------------------
# Main-process resources (for endgame only)
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


# ===================================================================
# WORKER PROCESS CODE (runs in separate processes)
# ===================================================================

# Worker-level globals (loaded once per process via initializer)
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

    # Pre-compute tile values and bonus grid (same as main process)
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

    Args (tuple):
        grid: 15x15 list of lists (None or uppercase letter)
        bb_set_list: list of (r0, c0) tuples for blanks on board
        move: dict with word, row, col, direction, score, blanks_used
        unseen_pool: list of tile characters
        k_sims: number of MC iterations for this worker
        seed: random seed

    Returns:
        dict with move identifier + avg_opp score + sims_run
    """
    grid, bb_set_list, move, unseen_pool, k_sims, seed = args

    random.seed(seed)

    from engine.config import VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE

    bb_set = set(bb_set_list)

    # Reconstruct board from grid
    from engine.board import Board
    board = Board()
    for r in range(15):
        for c in range(15):
            if grid[r][c] is not None:
                board._grid[r][c] = grid[r][c]

    # Place candidate move
    horizontal = move['direction'] == 'H'
    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    # Update blank set for candidate's blanks
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

    # Create Cython BoardContext
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
        # Early stopping
        if n_sims >= ES_MIN_SIMS and n_sims % ES_CHECK_EVERY == 0:
            variance = (running_sum_sq / n_sims) - (running_sum / n_sims) ** 2
            if variance > 0:
                se = (variance / n_sims) ** 0.5
                if se < ES_SE_THRESHOLD:
                    break

        opp_rack = ''.join(random.sample(unseen_pool, rack_draw))

        if use_cython:
            opp_score, _, _, _, _ = _w_accel.find_best_score_c(ctx, opp_rack)
        else:
            # Slow fallback
            from bots.base_engine import get_legal_moves
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
# DadBot class
# ===================================================================

# Persistent worker pool (shared across all DadBot instances)
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=MC_WORKERS,
            initializer=_worker_init,
            initargs=(_CROSSPLAY_DIR, _TOURNAMENT_DIR),
        )
    return _pool


class DadBot(BaseEngine):

    @property
    def name(self):
        return "DadBot"

    def game_over(self, result, game_info):
        """Shut down worker pool when game ends."""
        global _pool
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None

    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None

        _ensure_resources()

        bag_tiles = game_info.get('tiles_in_bag', 1)
        blanks_on_board = game_info.get('blanks_on_board', [])
        grid = [row[:] for row in board._grid]  # snapshot

        # ---------------------------------------------------------------
        # Endgame: bag empty, deterministic minimax (main process)
        # ---------------------------------------------------------------
        if bag_tiles == 0 and ENDGAME_FULL_SEARCH:
            unseen = _compute_unseen(grid, rack, blanks_on_board)
            opp_rack = ''.join(unseen)
            accel = _get_accel()

            best_move = None
            best_equity = float('-inf')

            bb_set = {(r - 1, c - 1) for r, c, _ in (blanks_on_board or [])}

            for move in moves:
                horizontal = move['direction'] == 'H'
                placed = board.place_move(
                    move['word'], move['row'], move['col'], horizontal)

                move_bb = set(bb_set)
                for bi in move.get('blanks_used', []):
                    if horizontal:
                        move_bb.add((move['row'] - 1, move['col'] - 1 + bi))
                    else:
                        move_bb.add((move['row'] - 1 + bi, move['col'] - 1))

                if accel and hasattr(accel, 'prepare_board_context'):
                    ctx = accel.prepare_board_context(
                        board._grid, _gdata_bytes, move_bb,
                        _word_set, VALID_TWO_LETTER,
                        _TV, _BONUS, BINGO_BONUS, RACK_SIZE,
                    )
                    opp_score, _, _, _, _ = accel.find_best_score_c(
                        ctx, opp_rack)
                else:
                    opp_score = 0
                    opp_moves = get_legal_moves(
                        board, opp_rack, blanks_on_board)
                    if opp_moves:
                        opp_score = opp_moves[0]['score']

                board.undo_move(placed)
                equity = move['score'] - opp_score
                if equity > best_equity:
                    best_equity = equity
                    best_move = move

            return best_move

        # ---------------------------------------------------------------
        # Mid-game: parallel MC 2-ply with SuperLeaves
        # ---------------------------------------------------------------
        unseen_pool = _compute_unseen(grid, rack, blanks_on_board)
        candidates = moves[:N_CANDIDATES]

        # Blank set as list of tuples (serializable)
        bb_set_list = [(r - 1, c - 1) for r, c, _ in (blanks_on_board or [])]

        # Build work items: one per candidate, each gets full K_SIMS
        # Workers do their own early stopping so actual sims may be less
        work = []
        for i, move in enumerate(candidates):
            # Serialize only what workers need
            move_data = {
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'blanks_used': move.get('blanks_used', []),
            }
            seed = random.randint(0, 2**31)
            work.append((grid, bb_set_list, move_data, unseen_pool,
                         K_SIMS, seed))

        # Fan out to worker pool
        pool = _get_pool()
        futures = [pool.submit(_worker_eval_candidate, w) for w in work]

        # Collect results
        best_move = None
        best_total = float('-inf')

        for i, future in enumerate(futures):
            result = future.result(timeout=30)
            avg_opp = result['avg_opp']
            mc_equity = candidates[i]['score'] - avg_opp

            leave = candidates[i].get('leave', '')
            lv = _leave_value(leave) if bag_tiles > RACK_SIZE else 0.0

            total = mc_equity + lv

            if total > best_total:
                best_total = total
                best_move = candidates[i]

        return best_move
