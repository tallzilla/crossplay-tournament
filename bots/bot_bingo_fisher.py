"""
Strategy C: Bingo fisher -- obsessively protect blanks and S tiles,
play bingos whenever possible.

Logic:
  1. If any move uses all 7 tiles (bingo, +40 pts), play the best one.
  2. If we hold a blank: avoid spending it unless score >= BLANK_THRESHOLD.
     Among blank-preserving moves, pick highest score + leave.
  3. Otherwise: score + leave, with bonus for keeping S tiles.
"""
from bots.base_engine import BaseEngine

BLANK_THRESHOLD = 45   # min score to "spend" a blank on a non-bingo

LEAVE_VALUES = {
    '?': 28.0, 'S': 9.0, 'R': 2.5, 'N': 1.5, 'E': 1.0,
    'T': 1.0,  'L': 0.5, 'A': 0.5, 'I': 0.0, 'D': 0.0,
    'O': -0.5, 'U': -1.5,
}


def leave_value(leave):
    return sum(LEAVE_VALUES.get(t, -0.5) for t in leave)


class BotBingoFisher(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None
        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        if tiles_in_bag == 0:
            return moves[0]

        # 1. Play any bingo (uses all 7 tiles)
        bingos = [m for m in moves if len(m.get('tiles_used', [])) == 7]
        if bingos:
            return max(bingos, key=lambda m: m['score'])

        # 2. Protect blanks
        has_blank = '?' in rack
        if has_blank:
            # Moves that don't spend the blank
            safe = [m for m in moves if '?' not in m.get('tiles_used', [])]
            if safe:
                return max(safe[:30], key=lambda m: m['score'] + 2.0 * leave_value(m.get('leave', '')))
            # All moves spend the blank -- only do it for high scores
            worth_it = [m for m in moves if m['score'] >= BLANK_THRESHOLD]
            if worth_it:
                return max(worth_it, key=lambda m: m['score'])

        # 3. Default: score + leave, biased toward keeping S tiles
        return max(moves[:30], key=lambda m: m['score'] + 2.0 * leave_value(m.get('leave', '')))
