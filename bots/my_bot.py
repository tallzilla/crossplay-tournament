"""
MyBot v12 -- Cython-accelerated Monte Carlo (sequential, main process).

v12: Replaces v11's Python `get_legal_moves` simulations with
`find_best_score_c` from the engine's Cython extension.

Key design choices:
  - No worker processes: eliminates fork overhead and CPU contention
  - Cython in main process: ~1ms/sim vs ~10ms Python = ~10x faster
  - Adequate K_SIMS: reliable signal without full convergence
  - Raw score MC (same as DadBot): measures max opponent scoring opportunity
  - All v11 heuristics preserved (leave values, risk/HVT, endgame minimax)
  - Falls back to Python if Cython unavailable

v12 vs v11 sim budget (same wall time targets):
  blitz:    K=8 Python   → K=60  Cython  (~60ms MC, was ~80ms)
  fast:     K=50 Python  → K=200 Cython  (~200ms MC, was ~500ms)
  standard: K=75 Python  → K=500 Cython  (~500ms MC, was ~750ms)
  deep:     K=150 Python → K=1000 Cython (~1s MC, was ~1.5s)

Previous v12 failure (6 sims) was too noisy to overcome raw-score bias.
100+ sims gives reliable ranking; raw score compresses equity range which
keeps risk/HVT penalties at comparable relative influence.
"""
import os
import sys
import random
from collections import Counter
from bots.base_engine import BaseEngine, get_legal_moves
from engine.config import (
    TILE_DISTRIBUTION, BONUS_SQUARES, VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE,
)

# ---------------------------------------------------------------------------
# Tier-aware parameters
# ---------------------------------------------------------------------------
_tier = os.environ.get('BOT_TIER', 'fast')
if _tier == 'blitz':
    N_CANDIDATES = 6
    K_SIMS       = 60    # ~60ms Cython; target <0.5s/move
elif _tier == 'standard':
    N_CANDIDATES = 12
    K_SIMS       = 300   # target ~7s/move
elif _tier == 'deep':
    N_CANDIDATES = 15
    K_SIMS       = 700   # target ~18s/move
else:  # fast (default)
    N_CANDIDATES = 8
    K_SIMS       = 120   # ~120ms Cython; target ~2.5s/move

# High-value tiles for HVT exposure penalty
_HVT = {'J': 10, 'Q': 10, 'Z': 10, 'X': 8, 'K': 5}

# Tiles that contribute to bingo probability
_BINGO_TILES = set('SATINRELD?')

# ---------------------------------------------------------------------------
# Crossplay-calibrated single-tile leave values (unchanged from v11)
# ---------------------------------------------------------------------------
CROSSPLAY_TILE_VALUES = {
    '?': 20.0,   # Blank
    'S':  7.0,
    'Z':  5.12,
    'X':  3.31,
    'R':  1.10,
    'H':  0.60,
    'C':  0.85,
    'M':  0.58,
    'D':  0.45,
    'E':  0.35,
    'N':  0.22,
    'T': -0.10,
    'L':  0.20,
    'P': -0.46,
    'K': -0.20,
    'Y': -0.63,
    'A': -0.63,
    'J': -0.80,
    'B': -1.50,
    'I': -2.07,
    'F': -2.21,
    'O': -2.50,
    'G': -1.80,
    'W': -4.50,
    'U': -4.00,
    'V': -6.50,
    'Q': -6.79,
}


def _leave_decay(tiles_in_bag):
    """Scale down leave weight as bag empties."""
    if tiles_in_bag >= 30:
        return 1.0
    elif tiles_in_bag >= 15:
        return 0.70
    elif tiles_in_bag >= 7:
        return 0.40
    else:
        return 0.10


def crossplay_leave_value(leave, tiles_in_bag=100, unseen=None):
    """Return equity of holding this leave."""
    value = sum(CROSSPLAY_TILE_VALUES.get(t, -1.0) for t in leave)

    if len(leave) >= 2:
        vowels = sum(1 for t in leave if t in 'AEIOU')
        consonants = sum(1 for t in leave if t.isalpha() and t not in 'AEIOU' and t != '?')
        if vowels == 1 and consonants >= 1:
            value += 2.0
        elif vowels >= 2 and consonants == 0:
            value -= 5.0

        for tile, count in Counter(t.upper() for t in leave).items():
            if count >= 2:
                tv = CROSSPLAY_TILE_VALUES.get(tile, -1.0)
                if tile == '?':     penalty = 8.0
                elif tv >= 5.0:     penalty = 5.0
                elif tv >= 2.0:     penalty = 4.0
                elif tv >= 0.5:     penalty = 3.0
                else:               penalty = 1.5
                value -= penalty * (count - 1)

        if len(leave) <= 4:
            bingo_count = sum(1 for t in leave if t.upper() in _BINGO_TILES)
            if bingo_count >= 2:
                value += 2.0

    if 'Q' in leave and unseen is not None and unseen.get('U', 0) == 0:
        value -= 8.0
    value *= _leave_decay(tiles_in_bag)
    return value


# ---------------------------------------------------------------------------
# Feature 1: Bonus square risk penalty (unchanged from v11)
# ---------------------------------------------------------------------------
def _risk_penalty(move, board):
    """Penalty for opening bonus squares to the opponent."""
    word = move['word']
    row, col = move['row'], move['col']
    horizontal = move['direction'] == 'H'
    penalty = 0.0
    checked = set()

    for i in range(len(word)):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if not board.is_empty(r, c):
            continue

        perp = [(r - 1, c), (r + 1, c)] if horizontal else [(r, c - 1), (r, c + 1)]
        for nr, nc in perp:
            if (nr, nc) in checked:
                continue
            checked.add((nr, nc))
            if 1 <= nr <= 15 and 1 <= nc <= 15 and board.is_empty(nr, nc):
                b = BONUS_SQUARES.get((nr, nc), '')
                if b == '3W':     penalty += 8.0
                elif b == '2W':   penalty += 3.0
                elif b == '3L':   penalty += 1.5
                elif b == '2L':   penalty += 0.5

    return penalty


# ---------------------------------------------------------------------------
# Feature 2: HVT exposure penalty (unchanged from v11)
# ---------------------------------------------------------------------------
def _hvt_exposure(move, board, unseen):
    """Penalty for exposing DLS squares to high-value tiles."""
    total_unseen = sum(unseen.values())
    if total_unseen < 7:
        return 0.0

    word = move['word']
    row, col = move['row'], move['col']
    horizontal = move['direction'] == 'H'
    penalty = 0.0
    checked = set()

    for i in range(len(word)):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if not board.is_empty(r, c):
            continue

        perp = [(r - 1, c), (r + 1, c)] if horizontal else [(r, c - 1), (r, c + 1)]
        for nr, nc in perp:
            if (nr, nc) in checked:
                continue
            checked.add((nr, nc))
            if (1 <= nr <= 15 and 1 <= nc <= 15
                    and board.is_empty(nr, nc)
                    and BONUS_SQUARES.get((nr, nc)) == '2L'):
                for tile, tv in _HVT.items():
                    cnt = unseen.get(tile, 0)
                    if cnt > 0:
                        p_draw = 1.0 - ((total_unseen - cnt) / total_unseen) ** 7
                        penalty += p_draw * tv

    return penalty


# ---------------------------------------------------------------------------
# Unseen tile tracking (unchanged from v11)
# ---------------------------------------------------------------------------
def unseen_tiles(board, rack, game_info):
    """Return dict of {tile: count} for tiles not on board and not in our rack."""
    remaining = dict(TILE_DISTRIBUTION)
    blanks_on_board = {(r, c) for r, c, _ in game_info.get('blanks_on_board', [])}
    for row in range(1, 16):
        for col in range(1, 16):
            tile = board.get_tile(row, col)
            if tile is not None:
                key = '?' if (row, col) in blanks_on_board else tile.upper()
                remaining[key] = max(0, remaining.get(key, 0) - 1)
    for tile in rack:
        t = tile.upper()
        remaining[t] = max(0, remaining.get(t, 0) - 1)
    return {k: v for k, v in remaining.items() if v > 0}


# ---------------------------------------------------------------------------
# Cython resource initialization (lazy, first call only)
# ---------------------------------------------------------------------------
_resources_loaded = False
_accel = None
_gdata_bytes = None
_word_set = None
_tv = None
_bonus = None


def _load_resources():
    global _resources_loaded, _accel, _gdata_bytes, _word_set, _tv, _bonus
    if _resources_loaded:
        return

    # Add engine dir to path for gaddag_accel.pyd
    _engine_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine'))
    if _engine_dir not in sys.path:
        sys.path.insert(0, _engine_dir)

    try:
        import gaddag_accel
        _accel = gaddag_accel
    except ImportError:
        _accel = None

    from engine.gaddag import get_gaddag
    _gdata_bytes = bytes(get_gaddag()._data)

    from engine.dictionary import get_dictionary
    _word_set = get_dictionary()._words

    from engine.config import TILE_VALUES, BONUS_SQUARES as BS
    _tv = [0] * 26
    for ch, val in TILE_VALUES.items():
        if ch != '?':
            _tv[ord(ch) - 65] = val

    _bonus = [[(1, 1)] * 15 for _ in range(15)]
    for (r1, c1), btype in BS.items():
        r0, c0 = r1 - 1, c1 - 1
        if btype == '2L':   _bonus[r0][c0] = (2, 1)
        elif btype == '3L': _bonus[r0][c0] = (3, 1)
        elif btype == '2W': _bonus[r0][c0] = (1, 2)
        elif btype == '3W': _bonus[r0][c0] = (1, 3)

    _resources_loaded = True


# ---------------------------------------------------------------------------
# Cython simulation
# ---------------------------------------------------------------------------
def _simulate_c(board, move, unseen_list, game_info, k_sims):
    """Place candidate, run k_sims opponent simulations, return avg raw score.

    Uses find_best_score_c (Cython) when available, falls back to Python
    get_legal_moves otherwise.
    """
    blanks_on_board = game_info.get('blanks_on_board', [])
    horizontal = move['direction'] == 'H'

    # Build 0-indexed blank set for Cython (existing + new from this move)
    bb_set = {(r - 1, c - 1) for r, c, _ in blanks_on_board}
    for bi in move.get('blanks_used', []):
        if horizontal:
            bb_set.add((move['row'] - 1, move['col'] - 1 + bi))
        else:
            bb_set.add((move['row'] - 1 + bi, move['col'] - 1))

    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    pool_size = len(unseen_list)
    rack_draw = min(RACK_SIZE, pool_size)
    running_sum = 0.0

    if _accel is not None:
        ctx = _accel.prepare_board_context(
            board._grid, _gdata_bytes, bb_set,
            _word_set, VALID_TWO_LETTER, _tv, _bonus, BINGO_BONUS, RACK_SIZE,
        )
        for _ in range(k_sims):
            opp_rack = ''.join(random.sample(unseen_list, rack_draw))
            opp_score, _, _, _, _ = _accel.find_best_score_c(ctx, opp_rack)
            running_sum += opp_score
    else:
        # Python fallback: reconstruct 1-indexed blanks with correct letters
        py_blanks = list(blanks_on_board)
        for bi in move.get('blanks_used', []):
            if horizontal:
                py_blanks.append((move['row'], move['col'] + bi, move['word'][bi]))
            else:
                py_blanks.append((move['row'] + bi, move['col'], move['word'][bi]))
        for _ in range(k_sims):
            opp_rack = ''.join(random.sample(unseen_list, rack_draw))
            opp_moves = get_legal_moves(board, opp_rack, py_blanks)
            if opp_moves:
                running_sum += opp_moves[0]['score']

    board.undo_move(placed)
    return running_sum / k_sims if k_sims > 0 else 0.0


# ---------------------------------------------------------------------------
# Feature 6: Endgame minimax (bag = 0, opponent rack known exactly)
# ---------------------------------------------------------------------------
def _endgame_minimax(board, moves, unseen, game_info):
    """When bag is empty, opponent rack is known. Maximize our_score - opp_best."""
    blanks_on_board = game_info.get('blanks_on_board', [])
    opp_rack = ''.join(t for t, cnt in unseen.items() for _ in range(cnt))
    if not opp_rack:
        return moves[0]

    candidates = moves[:30]
    best_move = None
    best_equity = float('-inf')

    for move in candidates:
        horizontal = move['direction'] == 'H'
        placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

        new_blanks = list(blanks_on_board)
        for bi in move.get('blanks_used', []):
            if horizontal:
                new_blanks.append((move['row'], move['col'] + bi, move['word'][bi]))
            else:
                new_blanks.append((move['row'] + bi, move['col'], move['word'][bi]))

        opp_moves = get_legal_moves(board, opp_rack, new_blanks)
        opp_score = max((m['score'] for m in opp_moves), default=0)
        board.undo_move(placed)

        equity = move['score'] - opp_score
        if equity > best_equity:
            best_equity = equity
            best_move = move

    return best_move or moves[0]


# ---------------------------------------------------------------------------
# Main bot
# ---------------------------------------------------------------------------
class MyBot(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None

        _load_resources()

        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        unseen = unseen_tiles(board, rack, game_info)

        # Endgame minimax — bag empty, opponent rack known exactly
        if tiles_in_bag == 0:
            return _endgame_minimax(board, moves, unseen, game_info)

        # Spread-adaptive leave weighting
        spread = game_info.get('your_score', 0) - game_info.get('opp_score', 0)
        if spread < -40:
            leave_weight = 0.6
        elif spread > 40:
            leave_weight = 1.3
        else:
            leave_weight = 1.0

        # Risk/HVT penalty weight
        risk_w = 0.7 if tiles_in_bag >= 20 else 0.3

        def _score(m):
            return (m['score']
                    + leave_weight * crossplay_leave_value(m.get('leave', ''), tiles_in_bag, unseen)
                    - risk_w * _risk_penalty(m, board)
                    - risk_w * 0.5 * _hvt_exposure(m, board, unseen))

        candidates = sorted(moves, key=_score, reverse=True)[:N_CANDIDATES]

        # Near-bag-empty: skip simulation, use adjusted score
        if tiles_in_bag < 15:
            if tiles_in_bag <= 7:
                return max(candidates,
                           key=lambda m: _score(m) + len(m.get('tiles_used', [])) * 0.5)
            return candidates[0]

        unseen_list = [t for t, cnt in unseen.items() for _ in range(cnt)]

        # Evaluate each candidate with K_SIMS Cython sims
        best_move = None
        best_equity = float('-inf')

        for move in candidates:
            avg_opp = _simulate_c(board, move, unseen_list, game_info, K_SIMS)
            equity = _score(move) - avg_opp
            if equity > best_equity:
                best_equity = equity
                best_move = move

        return best_move or candidates[0]
