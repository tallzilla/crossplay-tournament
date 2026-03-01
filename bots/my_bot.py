"""
MyBot -- Crossplay-calibrated Monte Carlo simulation.

v10: Duplicate tile penalty added to leave evaluation.
     Penalizes holding multiple copies of the same tile (SS, ZZ, RR, etc.)
     since second copies reduce flexibility without proportional value gain.
"""
import os
import random
from collections import Counter
from bots.base_engine import BaseEngine, get_legal_moves
from engine.config import TILE_DISTRIBUTION

# Tier-aware parameters (~10ms per get_legal_moves call)
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


# ---------------------------------------------------------------------------
# Crossplay-calibrated single-tile leave values (fallback)
# ---------------------------------------------------------------------------
CROSSPLAY_TILE_VALUES = {
    '?': 20.0,   # Blank: 25.57 * (40/50 sweep scale) + 3-blank adjustment
    'S':  7.0,   # S: lower for 40-pt sweep setup (was 8.04)
    'Z':  5.12,  # Z=10 in both → unchanged
    'X':  3.31,  # X=8 in both → unchanged
    'R':  1.10,  # R=1 in both → unchanged
    'H':  0.60,  # H=3 in Crossplay (vs 4 in Scrabble) → slightly lower
    'C':  0.85,  # C=3 in both → unchanged
    'M':  0.58,  # M=3 in both → unchanged
    'D':  0.45,  # D=2 in both → unchanged
    'E':  0.35,  # E=1 in both → unchanged
    'N':  0.22,  # N=1 in both → unchanged
    'T': -0.10,  # T=1 in both → unchanged
    'L':  0.20,  # L=2 in Crossplay (vs 1) → improved
    'P': -0.46,  # P=3 in both → unchanged
    'K': -0.20,  # K=6 in Crossplay (vs 5) → less negative
    'Y': -0.63,  # Y=4 in both → unchanged
    'A': -0.63,  # A=1 in both → unchanged
    'J': -0.80,  # J=10 in Crossplay (vs 8) → less negative
    'B': -1.50,  # B=4 in Crossplay (vs 3) → less negative
    'I': -2.07,  # I=1 in both → unchanged
    'F': -2.21,  # F=4 in both → unchanged
    'O': -2.50,  # O=1 in both → unchanged
    'G': -1.80,  # G=4 in Crossplay (vs 2) → less negative
    'W': -4.50,  # W=5 in Crossplay (vs 4) → harder to play
    'U': -4.00,  # U=2 in Crossplay (vs 1) → less negative
    'V': -6.50,  # V=6 in Crossplay (vs 4) → hard to play
    'Q': -6.79,  # Q=10 in both → unchanged
}


def _leave_decay(tiles_in_bag):
    """Scale down leave weight as bag empties.
    No end-of-game tile penalty in Crossplay → leave matters less late."""
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

        # Duplicate tile penalty: second copy of any tile reduces flexibility
        for tile, count in Counter(t.upper() for t in leave).items():
            if count >= 2:
                tv = CROSSPLAY_TILE_VALUES.get(tile, -1.0)
                if tile == '?':
                    penalty = 8.0   # ?? still great but not worth 2x single blank
                elif tv >= 5.0:     # S, Z
                    penalty = 5.0   # SS/ZZ badly overvalued without this
                elif tv >= 2.0:     # X
                    penalty = 4.0
                elif tv >= 0.5:     # R, H, C, M
                    penalty = 3.0
                else:               # vowels, low-value consonants
                    penalty = 1.5
                value -= penalty * (count - 1)

    if 'Q' in leave and unseen is not None and unseen.get('U', 0) == 0:
        value -= 8.0
    value *= _leave_decay(tiles_in_bag)
    return value


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


def _simulate(board, move, unseen_list, game_info):
    """Place candidate, sample N_SAMPLES opponent racks, return avg opp best score.

    Models opponent as a leave-aware player (score + leave quality), which is
    more accurate across diverse bot types than pure score-maximization.
    Records the raw score of the opponent's selected move.
    """
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

    for _ in range(N_SAMPLES):
        opp_rack = ''.join(random.sample(unseen_list, rack_draw))
        opp_moves = get_legal_moves(board, opp_rack, new_blanks)
        if opp_moves:
            best = max(opp_moves,
                       key=lambda m: m['score'] + crossplay_leave_value(
                           m.get('leave', ''), tiles_in_bag))
            running_sum += best['score']

    board.undo_move(placed)
    return running_sum / N_SAMPLES


class MyBot(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None

        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        unseen = unseen_tiles(board, rack, game_info)

        candidates = sorted(
            moves,
            key=lambda m: m['score'] + crossplay_leave_value(
                m.get('leave', ''), tiles_in_bag, unseen),
            reverse=True
        )[:N_CANDIDATES]

        # No simulation when bag is nearly empty — leave value dominates
        if tiles_in_bag < 15:
            return candidates[0]

        unseen_list = [t for t, cnt in unseen.items() for _ in range(cnt)]

        best_move   = None
        best_equity = float('-inf')

        for move in candidates:
            avg_opp = _simulate(board, move, unseen_list, game_info)
            equity  = (move['score']
                       + crossplay_leave_value(move.get('leave', ''), tiles_in_bag, unseen)
                       - avg_opp)
            if equity > best_equity:
                best_equity = equity
                best_move   = move

        return best_move or candidates[0]
