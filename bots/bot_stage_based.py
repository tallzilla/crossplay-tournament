"""
Strategy E: Stage-based -- completely different priorities per game phase.

Early game (bag > 60): Leave quality is king. Build the perfect rack.
  -> score + 4*leave  (sacrifice points for great tiles)

Mid game (bag 15-60): Balanced. Score, leave, and defense all matter.
  -> score + 2*leave - defense_penalty

Late game (bag < 15): Endgame mode. Leave doesn't matter, just score + defense.
  -> score - 2*defense_penalty  (protect bonus squares, maximize points)
"""
from bots.base_engine import BaseEngine
from engine.config import BONUS_SQUARES

LEAVE_VALUES = {
    '?': 25.0, 'S': 8.0, 'R': 2.0, 'N': 1.5, 'E': 1.0,
    'T': 1.0,  'L': 0.5, 'A': 0.5, 'I': 0.0, 'D': 0.0,
    'O': -0.5, 'U': -1.5,
}
DUPLICATE_PENALTY = -3.0


def leave_value(leave):
    value = 0.0
    seen = set()
    for tile in leave:
        value += LEAVE_VALUES.get(tile, -0.5)
        if tile in seen:
            value += DUPLICATE_PENALTY
        seen.add(tile)
    vowels     = sum(1 for t in leave if t in 'AEIOU')
    consonants = sum(1 for t in leave if t.isalpha() and t not in 'AEIOU')
    if len(leave) >= 3 and (vowels == 0 or consonants == 0):
        value -= 5.0
    return value


def defense_penalty(board, move):
    word       = move['word']
    row        = move['row']
    col        = move['col']
    horizontal = move['direction'] == 'H'
    filled = set()
    for i in range(len(word)):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        filled.add((r, c))
    penalty = 0.0
    for (br, bc), btype in BONUS_SQUARES.items():
        if board.get_tile(br, bc) is not None:
            continue
        weight = 12.0 if btype == '3W' else (5.0 if btype == '2W' else 0.0)
        if weight == 0:
            continue
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (br+dr, bc+dc) in filled:
                penalty -= weight
                break
    return penalty


class BotStageBased(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None
        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        if tiles_in_bag == 0:
            return moves[0]

        if tiles_in_bag > 60:
            # Early: rack quality above all
            return max(moves[:50], key=lambda m: m['score'] + 4.0 * leave_value(m.get('leave', '')))

        elif tiles_in_bag >= 15:
            # Mid: balanced
            return max(moves[:25], key=lambda m: m['score'] + 2.0 * leave_value(m.get('leave', '')) + defense_penalty(board, m))

        else:
            # Late: score + double defense, ignore leave
            return max(moves[:25], key=lambda m: m['score'] + 2.0 * defense_penalty(board, m))
