"""
Strategy D: Spread-adaptive -- change play style based on score gap.

- Losing badly (< -50): pure greedy, maximize score, take risks
- Winning comfortably (> +50): play defensively, minimize bonus square exposure
- Close game: balanced static eval (score + leave + defense)
"""
from bots.base_engine import BaseEngine
from engine.config import BONUS_SQUARES

LEAVE_VALUES = {
    '?': 25.0, 'S': 8.0, 'R': 2.0, 'N': 1.5, 'E': 1.0,
    'T': 1.0,  'L': 0.5, 'A': 0.5, 'I': 0.0, 'D': 0.0,
    'O': -0.5, 'U': -1.5,
}


def leave_value(leave):
    return sum(LEAVE_VALUES.get(t, -0.5) for t in leave)


def bonus_exposure(board, move):
    """Count open bonus squares adjacent to this move (lower is more defensive)."""
    word       = move['word']
    row        = move['row']
    col        = move['col']
    horizontal = move['direction'] == 'H'
    filled = set()
    for i in range(len(word)):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        filled.add((r, c))

    exposure = 0
    for (br, bc), btype in BONUS_SQUARES.items():
        if board.get_tile(br, bc) is not None:
            continue
        weight = 3 if btype == '3W' else (2 if btype == '2W' else 0)
        if weight == 0:
            continue
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (br+dr, bc+dc) in filled:
                exposure += weight
                break
    return exposure


class BotSpreadAdaptive(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None
        tiles_in_bag = game_info.get('tiles_in_bag', 1)
        if tiles_in_bag == 0:
            return moves[0]

        spread = game_info.get('your_score', 0) - game_info.get('opp_score', 0)

        if spread < -50:
            # Losing badly: pure greedy, need points NOW
            return moves[0]

        if spread > 50:
            # Winning comfortably: play it safe, minimize exposure
            return min(moves[:20], key=lambda m: bonus_exposure(board, m) * 10 - m['score'])

        # Close game: balanced
        return max(moves[:25], key=lambda m: m['score'] + 1.5 * leave_value(m.get('leave', '')) - bonus_exposure(board, m) * 5)
