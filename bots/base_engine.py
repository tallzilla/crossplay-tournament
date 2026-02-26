"""
Base class for tournament engines.

Your bot extends BaseEngine and implements pick_move().
The match runner calls get_legal_moves() for you -- your bot
just picks which move to play from the list.
"""

import os
import sys
from abc import ABC, abstractmethod

# Add project root to path so engine imports work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine.board import Board
from engine.config import TILE_VALUES, RACK_SIZE
from engine.scoring import calculate_move_score


class BaseEngine(ABC):
    """Base class for tournament engines.

    Subclass this and implement pick_move(). That's the only
    method you need. Everything else is optional.
    """

    @property
    def name(self) -> str:
        """Display name for this engine. Override if you want a custom name."""
        return self.__class__.__name__

    @abstractmethod
    def pick_move(self, board, rack, moves, game_info):
        """Choose a move from the list of legal moves.

        This is the ONLY method you must implement.

        Args:
            board: Board instance. Query tiles with board.get_tile(row, col),
                   check squares with board.is_empty(row, col), etc.
                   Rows and columns are 1-indexed (1 to 15).

            rack: str -- your tiles, e.g. "AEINRST". Blanks are "?".

            moves: list of move dicts, sorted by score (highest first).
                Each move dict has these fields:
                    word:       str   -- the word played (e.g. "QUARTZ")
                    row:        int   -- starting row (1-indexed)
                    col:        int   -- starting column (1-indexed)
                    direction:  str   -- "H" (horizontal) or "V" (vertical)
                    score:      int   -- total points including bonuses
                    tiles_used: list  -- rack letters consumed (e.g. ["Q","U","A","R","T","Z"])
                    leave:      str   -- remaining rack tiles after playing
                    blanks_used: list -- indices in word where blanks are used

                The list may be empty if no legal moves exist (you should
                return None to pass).

            game_info: dict with game state:
                your_score:      int   -- your current score
                opp_score:       int   -- opponent's current score
                tiles_in_bag:    int   -- tiles remaining in the bag
                move_number:     int   -- current move number (1-based)
                blanks_on_board: list  -- [(row, col, letter), ...] for blanks on board

        Returns:
            A move dict from the moves list, or None to pass.
        """
        raise NotImplementedError

    def notify_opponent_move(self, move, game_info):
        """Called after the opponent plays. Override if you want to track state.

        Args:
            move: dict with the same fields as in pick_move's moves list,
                  or None if opponent passed.
            game_info: same dict format as in pick_move.
        """
        pass

    def game_over(self, result, game_info):
        """Called when the game ends. Override for post-game analysis.

        Args:
            result: "win", "loss", or "tie"
            game_info: final game state dict (same format as pick_move)
        """
        pass


# =========================================================================
# Helper: generate all legal moves for a board + rack
# =========================================================================

# Lazy-loaded GADDAG (built once, reused across all games)
_gaddag = None


def _get_gaddag():
    """Load the GADDAG (builds on first run, ~48s; then loads in <1s)."""
    global _gaddag
    if _gaddag is None:
        from engine.gaddag import get_gaddag
        _gaddag = get_gaddag()
    return _gaddag


def get_legal_moves(board, rack, blanks_on_board=None):
    """Generate all legal moves for a board position and rack.

    This is called by the match runner before each pick_move() call.
    You normally don't need to call this yourself, but you can if you
    want to explore "what if" scenarios.

    Args:
        board: Board instance
        rack: str -- your tiles (e.g. "AEINRST", "?" for blank)
        blanks_on_board: list of (row, col, letter) for blanks on the board

    Returns:
        List of move dicts sorted by score (highest first).
        Each dict has: word, row, col, direction, score, tiles_used, leave, blanks_used
    """
    if blanks_on_board is None:
        blanks_on_board = []

    gaddag = _get_gaddag()

    from engine.move_finder import find_all_moves_c
    moves = find_all_moves_c(board, gaddag, rack, board_blanks=blanks_on_board)

    # Enrich each move with tiles_used and leave
    for m in moves:
        if 'tiles_used' not in m:
            # Calculate which rack tiles this move consumes
            used = []
            leave_letters = list(rack.upper())
            horizontal = m['direction'] == 'H'
            for i, letter in enumerate(m['word']):
                r = m['row'] if horizontal else m['row'] + i
                c = m['col'] + i if horizontal else m['col']
                if board.get_tile(r, c) is None:
                    # This position needs a tile from the rack
                    blanks = m.get('blanks_used', [])
                    if i in (set(blanks) if blanks else set()):
                        used.append('?')
                        if '?' in leave_letters:
                            leave_letters.remove('?')
                    elif letter in leave_letters:
                        used.append(letter)
                        leave_letters.remove(letter)
                    elif '?' in leave_letters:
                        used.append('?')
                        leave_letters.remove('?')
            m['tiles_used'] = used
            m['leave'] = ''.join(sorted(leave_letters))

    return moves
