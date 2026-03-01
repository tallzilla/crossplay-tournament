"""
Strategy B: Leave-first -- rack quality dominates, score is secondary.

Weights leave value 4x more than score. Willingly sacrifices points
to keep blanks, S tiles, and balanced racks. Bets on future turns.
"""
from bots.base_engine import BaseEngine

LEAVE_VALUES = {
    '?': 30.0, 'S': 10.0, 'R': 3.0, 'N': 2.0, 'E': 1.5,
    'T': 1.5,  'L': 1.0,  'A': 1.0, 'I': 0.0, 'D': 0.0,
    'O': -1.0, 'U': -2.0,
}
DUPLICATE_PENALTY = -4.0


def leave_value(leave):
    value = 0.0
    seen = set()
    for tile in leave:
        value += LEAVE_VALUES.get(tile, -1.0)
        if tile in seen:
            value += DUPLICATE_PENALTY
        seen.add(tile)
    vowels     = sum(1 for t in leave if t in 'AEIOU')
    consonants = sum(1 for t in leave if t.isalpha() and t not in 'AEIOU')
    if len(leave) >= 3 and (vowels == 0 or consonants == 0):
        value -= 8.0
    return value


class BotLeaveFirst(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None
        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        if tiles_in_bag == 0:
            return moves[0]
        # Leave quality weighted 4x vs score
        return max(moves[:50], key=lambda m: m['score'] + 4.0 * leave_value(m.get('leave', '')))
