"""
MyBot -- Crossplay-calibrated Monte Carlo simulation.

Adjusts Quackle's Scrabble leave values for Crossplay-specific rules:
  1. Sweep = 40 pts (not 50): blank 25.57→20.0, S 8.04→7.0
  2. Different tile face values: L=2, U=2, J=10, G=4, K=6, B=4 → less negative
     V=6, W=5 → more negative (hard to play, high wasted potential)
  3. No end-of-game tile penalty: leave values decay toward zero as bag empties
  4. 3 blanks (not 2): captured in sweep-bonus scaling

Simulation: top 5 candidates × 5 opponent samples (Maven/Quackle-inspired).

Results (partial, ~80 games): 47-32 vs DefensiveBot (+16 spread);
24-15 vs BotFastSim (previous champion, +28 spread).
"""
import random
from bots.base_engine import BaseEngine, get_legal_moves
from engine.config import TILE_DISTRIBUTION

N_CANDIDATES = 5
N_SAMPLES    = 50

# Crossplay-calibrated single-tile leave values
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
    'L':  0.20,  # L=2 in Crossplay (vs 1) → improved (worth more when played)
    'P': -0.46,  # P=3 in both → unchanged
    'K': -0.20,  # K=6 in Crossplay (vs 5) → less negative
    'Y': -0.63,  # Y=4 in both → unchanged
    'A': -0.63,  # A=1 in both → unchanged
    'J': -0.80,  # J=10 in Crossplay (vs 8) → less negative (worth 2 pts more)
    'B': -1.50,  # B=4 in Crossplay (vs 3) → less negative
    'I': -2.07,  # I=1 in both → unchanged
    'F': -2.21,  # F=4 in both → unchanged
    'O': -2.50,  # O=1 in both → unchanged
    'G': -1.80,  # G=4 in Crossplay (vs 2) → significantly less negative
    'W': -4.50,  # W=5 in Crossplay (vs 4) → harder to play + higher wasted value
    'U': -4.00,  # U=2 in Crossplay (vs 1) → less negative (worth more when played)
    'V': -6.50,  # V=6 in Crossplay (vs 4) → harder to play + high wasted potential
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
    value = sum(CROSSPLAY_TILE_VALUES.get(t, -1.0) for t in leave)
    vowels = sum(1 for t in leave if t in 'AEIOU')
    consonants = sum(1 for t in leave if t.isalpha() and t not in 'AEIOU' and t != '?')
    if len(leave) >= 2:
        if vowels == 1 and consonants >= 1:
            value += 2.0
        elif vowels >= 2 and consonants == 0:
            value -= 5.0
    if 'Q' in leave and unseen is not None and unseen.get('U', 0) == 0:
        value -= 8.0
    value *= _leave_decay(tiles_in_bag)
    return value


def unseen_tiles(board, rack, game_info):
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
    tiles_in_bag = game_info.get('tiles_in_bag', 1)
    new_blanks = list(game_info.get('blanks_on_board', []))
    for idx in move.get('blanks_used', []):
        word, row, col = move['word'], move['row'], move['col']
        horiz = move['direction'] == 'H'
        r = row if horiz else row + idx
        c = col + idx if horiz else col
        if board.is_empty(r, c):
            new_blanks.append((r, c, word[idx]))

    placed = board.place_move(move['word'], move['row'], move['col'],
                              move['direction'] == 'H')
    opp_scores = []
    for _ in range(N_SAMPLES):
        random.shuffle(unseen_list)
        opp_rack  = ''.join(unseen_list[:min(7, len(unseen_list))])
        opp_moves = get_legal_moves(board, opp_rack, new_blanks)
        if opp_moves:
            best = max(opp_moves,
                       key=lambda m: m['score'] + crossplay_leave_value(
                           m.get('leave', ''), tiles_in_bag))
            opp_scores.append(best['score'])
        else:
            opp_scores.append(0)

    board.undo_move(placed)
    return sum(opp_scores) / len(opp_scores) if opp_scores else 0.0


class MyBot(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None
        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        if tiles_in_bag == 0:
            return moves[0]

        unseen = unseen_tiles(board, rack, game_info)

        candidates = sorted(
            moves,
            key=lambda m: m['score'] + crossplay_leave_value(
                m.get('leave', ''), tiles_in_bag, unseen),
            reverse=True
        )[:N_CANDIDATES]

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
