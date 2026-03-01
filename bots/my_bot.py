"""
MyBot -- Crossplay-calibrated Monte Carlo simulation.

v11: Comprehensive strategic improvements:
  1. Bonus square risk penalty (opening 3W/2W/3L/2L to opponent)
  2. HVT exposure penalty (exposing DLS squares to J/Q/Z/X/K)
  3. Spread-adaptive leave weighting (trailing: chase score; leading: conservative)
  6. Endgame minimax (bag=0: opponent rack known exactly)
  7. Two-phase simulation with candidate pruning
  8. Bingo probability bonus for bingo-prone leaves
  9. Near-endgame parity (tiles_used bonus when bag <= 7)
  10. Board tightness (incorporated via risk + HVT penalties)
"""
import os
import random
from collections import Counter
from bots.base_engine import BaseEngine, get_legal_moves
from engine.config import TILE_DISTRIBUTION, BONUS_SQUARES

# ---------------------------------------------------------------------------
# Tier-aware parameters (~10ms per get_legal_moves call)
# ---------------------------------------------------------------------------
_tier = os.environ.get('BOT_TIER', 'fast')
if _tier == 'blitz':
    N_CANDIDATES = 5
    N_SAMPLES    = 8    # target ~0.5s/move
elif _tier == 'standard':
    N_CANDIDATES = 10
    N_SAMPLES    = 75   # target ~7.5s/move
elif _tier == 'deep':
    N_CANDIDATES = 12
    N_SAMPLES    = 150  # target ~18s/move
else:  # fast (default)
    N_CANDIDATES = 8
    N_SAMPLES    = 50   # target ~2.5s/move

# High-value tiles for HVT exposure penalty (tile: face value)
_HVT = {'J': 10, 'Q': 10, 'Z': 10, 'X': 8, 'K': 5}

# Tiles that contribute to bingo probability
_BINGO_TILES = set('SATINRELD?')

# ---------------------------------------------------------------------------
# Crossplay-calibrated single-tile leave values
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
        # Vowel/consonant balance heuristic
        vowels = sum(1 for t in leave if t in 'AEIOU')
        consonants = sum(1 for t in leave if t.isalpha() and t not in 'AEIOU' and t != '?')
        if vowels == 1 and consonants >= 1:
            value += 2.0
        elif vowels >= 2 and consonants == 0:
            value -= 5.0

        # Duplicate tile penalty: second copy reduces flexibility
        for tile, count in Counter(t.upper() for t in leave).items():
            if count >= 2:
                tv = CROSSPLAY_TILE_VALUES.get(tile, -1.0)
                if tile == '?':     penalty = 8.0
                elif tv >= 5.0:     penalty = 5.0   # S, Z
                elif tv >= 2.0:     penalty = 4.0   # X
                elif tv >= 0.5:     penalty = 3.0   # R, H, C, M
                else:               penalty = 1.5   # vowels, low-value tiles
                value -= penalty * (count - 1)

        # Feature 8: Bingo probability bonus for bingo-prone short leaves
        if len(leave) <= 4:
            bingo_count = sum(1 for t in leave if t.upper() in _BINGO_TILES)
            if bingo_count >= 2:
                value += 2.0

    if 'Q' in leave and unseen is not None and unseen.get('U', 0) == 0:
        value -= 8.0
    value *= _leave_decay(tiles_in_bag)
    return value


# ---------------------------------------------------------------------------
# Feature 1: Bonus square risk penalty
# ---------------------------------------------------------------------------
def _risk_penalty(move, board):
    """Penalty for opening bonus squares (3W/2W/3L/2L) to the opponent.

    Checks perpendicular neighbors of newly placed tiles. If they are empty
    bonus squares, the opponent can now hook into them.
    """
    word = move['word']
    row, col = move['row'], move['col']
    horizontal = move['direction'] == 'H'
    penalty = 0.0
    checked = set()

    for i in range(len(word)):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if not board.is_empty(r, c):
            continue  # existing tile, skip

        # Perpendicular neighbors of newly placed tile
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
# Feature 2: HVT (high-value tile) exposure penalty
# ---------------------------------------------------------------------------
def _hvt_exposure(move, board, unseen):
    """Penalty for exposing Double Letter Squares that opponent could use
    with high-value tiles (J/Q/Z/X/K).

    Expected extra damage = sum over HVT tiles of: P(opponent draws tile) * tile_value
    """
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
# Unseen tile tracking
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
# Feature 7: Simulation with variable sample count (returns avg, raw_sum)
# ---------------------------------------------------------------------------
def _simulate(board, move, unseen_list, game_info, n_samples):
    """Place candidate, sample n_samples opponent racks, return (avg_opp, raw_sum)."""
    tiles_in_bag = game_info.get('tiles_in_bag', 1)
    blanks_on_board = game_info.get('blanks_on_board', [])
    horizontal = move['direction'] == 'H'

    new_blanks = list(blanks_on_board)
    for bi in move.get('blanks_used', []):
        if horizontal:
            new_blanks.append((move['row'], move['col'] + bi, move['word'][bi]))
        else:
            new_blanks.append((move['row'] + bi, move['col'], move['word'][bi]))

    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    running_sum = 0.0
    rack_draw = min(7, len(unseen_list))

    for _ in range(n_samples):
        opp_rack = ''.join(random.sample(unseen_list, rack_draw))
        opp_moves = get_legal_moves(board, opp_rack, new_blanks)
        if opp_moves:
            best = max(opp_moves,
                       key=lambda m: m['score'] + crossplay_leave_value(
                           m.get('leave', ''), tiles_in_bag))
            running_sum += best['score']

    board.undo_move(placed)
    avg = running_sum / n_samples if n_samples > 0 else 0.0
    return avg, running_sum


# ---------------------------------------------------------------------------
# Feature 6: Endgame minimax (bag = 0, opponent rack known exactly)
# ---------------------------------------------------------------------------
def _endgame_minimax(board, moves, unseen, game_info):
    """When bag is empty, opponent rack is known. Maximize our_score - opp_best."""
    blanks_on_board = game_info.get('blanks_on_board', [])
    opp_rack = ''.join(t for t, cnt in unseen.items() for _ in range(cnt))
    if not opp_rack:
        return moves[0]

    # Cap candidates for speed (low-scoring moves won't win anyway)
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

        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        unseen = unseen_tiles(board, rack, game_info)

        # Feature 6: Endgame minimax — bag empty, opponent rack known exactly
        if tiles_in_bag == 0:
            return _endgame_minimax(board, moves, unseen, game_info)

        # Feature 3: Spread-adaptive leave weighting
        spread = game_info.get('your_score', 0) - game_info.get('opp_score', 0)
        if spread < -40:
            leave_weight = 0.6    # trailing: prioritize raw score
        elif spread > 40:
            leave_weight = 1.3    # leading: value leave quality + safety more
        else:
            leave_weight = 1.0

        # Risk/HVT penalty weight — reduce impact in endgame (board more open anyway)
        risk_w = 0.7 if tiles_in_bag >= 20 else 0.3

        def _score(m):
            return (m['score']
                    + leave_weight * crossplay_leave_value(m.get('leave', ''), tiles_in_bag, unseen)
                    - risk_w * _risk_penalty(m, board)
                    - risk_w * 0.5 * _hvt_exposure(m, board, unseen))

        candidates = sorted(moves, key=_score, reverse=True)[:N_CANDIDATES]

        # Near-bag-empty: skip simulation, use adjusted score
        if tiles_in_bag < 15:
            # Feature 9: Near-endgame parity — slightly prefer plays using more tiles
            if tiles_in_bag <= 7:
                return max(candidates,
                           key=lambda m: _score(m) + len(m.get('tiles_used', [])) * 0.5)
            return candidates[0]

        unseen_list = [t for t, cnt in unseen.items() for _ in range(cnt)]

        # Feature 7: Two-phase simulation with candidate pruning
        # Phase 1: N_SAMPLES//3 sims for all candidates
        phase1_sims = max(3, N_SAMPLES // 3)
        phase1 = {}
        for move in candidates:
            avg_opp, raw_sum = _simulate(board, move, unseen_list, game_info, phase1_sims)
            phase1[id(move)] = (_score(move) - avg_opp, raw_sum)

        # Eliminate bottom third — keep top 2/3 (minimum 2)
        n_survivors = max(2, N_CANDIDATES * 2 // 3)
        survivors = sorted(candidates,
                           key=lambda m: phase1[id(m)][0],
                           reverse=True)[:n_survivors]

        # Phase 2: remaining sims for survivors
        phase2_sims = N_SAMPLES - phase1_sims
        best_move = None
        best_equity = float('-inf')

        for move in survivors:
            _, p1_sum = phase1[id(move)]
            _, p2_sum = _simulate(board, move, unseen_list, game_info, phase2_sims)
            avg_opp = (p1_sum + p2_sum) / N_SAMPLES
            equity = _score(move) - avg_opp
            if equity > best_equity:
                best_equity = equity
                best_move = move

        return best_move or candidates[0]
