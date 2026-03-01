"""
Strategy A: Pure greedy -- always take the highest-scoring move.
No leave eval, no defense, no modeling. Just max points every turn.
"""
from bots.base_engine import BaseEngine


class BotGreedy(BaseEngine):
    def pick_move(self, board, rack, moves, game_info):
        if not moves:
            return None
        return moves[0]  # already sorted by score descending
