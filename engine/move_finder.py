"""
CROSSPLAY V15 - Optimized Move Finder

Drop-in replacement for GADDAGMoveFinder.find_all_moves() targeting <100ms.

Optimizations vs the fast path in move_finder_gaddag.py:
1. Skip _validate_placement entirely — GADDAG moves are valid by construction
2. Inline all board access — direct grid[r][c] with 0-indexed coords throughout
3. Inline _get_child — unpack directly from bytearray via struct.unpack_from
4. Inline scoring — direct tile value lookup + bonus table, no function calls
5. Closure-captured locals — avoid self.attr and module-level lookups in recursion
6. List-based word building — avoid 56K string concatenations

All coordinates internal to this module are 0-indexed.
Output moves use 1-indexed coordinates (matching the rest of the codebase).
"""

import os
import sys
import struct
from collections import Counter
from typing import List, Dict, Optional, Set, Tuple

from engine.config import (
    BOARD_SIZE, CENTER_ROW, CENTER_COL, VALID_TWO_LETTER,
    TILE_VALUES, BONUS_SQUARES, BINGO_BONUS, RACK_SIZE,
)

# ---------------------------------------------------------------------------
# Optional Cython acceleration (gaddag_accel.pyd / .so)
# Falls back to pure Python if not available -- zero behavior change.
# ---------------------------------------------------------------------------
_accel = None
_accel_dir = os.path.dirname(os.path.abspath(__file__))
if _accel_dir not in sys.path:
    sys.path.insert(0, _accel_dir)
try:
    import gaddag_accel as _accel
except ImportError:
    pass

# Cache bytes(gaddag._data) to avoid 28MB copy per call
_gdata_bytes_cache = None
_gdata_source_id = None

def _get_gdata_bytes(gdata):
    global _gdata_bytes_cache, _gdata_source_id
    if _gdata_source_id != id(gdata):
        _gdata_bytes_cache = bytes(gdata)
        _gdata_source_id = id(gdata)
    return _gdata_bytes_cache

# === Pre-computed lookup tables (module-level, built once) ===

# Char <-> index for compact GADDAG
_C2I = {chr(65 + i): i for i in range(26)}
_C2I['+'] = 26
_I2C = {v: k for k, v in _C2I.items()}
_DELIM = 26
_I2C_LIST = [chr(65 + i) for i in range(26)] + ['+']

# Tile values as a list indexed by (ord(c) - 65) for A-Z
_TV = [0] * 26
for _c, _v in TILE_VALUES.items():
    if _c != '?':
        _TV[ord(_c) - 65] = _v

# Bonus grid: 0-indexed [r][c] -> (letter_mult, word_mult)
# Pre-build so scoring can do a single lookup
_BONUS = [[(1, 1)] * 15 for _ in range(15)]
for (r1, c1), btype in BONUS_SQUARES.items():
    r0, c0 = r1 - 1, c1 - 1
    if btype == '2L':
        _BONUS[r0][c0] = (2, 1)
    elif btype == '3L':
        _BONUS[r0][c0] = (3, 1)
    elif btype == '2W':
        _BONUS[r0][c0] = (1, 2)
    elif btype == '3W':
        _BONUS[r0][c0] = (1, 3)

# Dictionary for word validation
_dictionary = None

def _get_dict():
    global _dictionary
    if _dictionary is None:
        from engine.dictionary import get_dictionary
        _dictionary = get_dictionary()
    return _dictionary


def is_c_available():
    """Check if C-accelerated move finder is available."""
    return _accel is not None


def find_all_moves_c(board, gaddag, rack_str: str,
                     board_blanks: List[Tuple[int, int, str]] = None) -> List[Dict]:
    """
    Find all valid moves using C-accelerated GADDAG traversal + scoring.
    Falls back to find_all_moves_opt() if Cython extension not available.

    Same interface and output format as find_all_moves_opt().
    """
    if _accel is None:
        return find_all_moves_opt(board, gaddag, rack_str, board_blanks=board_blanks)

    rack_str = rack_str.upper()
    grid = board._grid
    dictionary = _get_dict()
    board_blank_set = {(r - 1, c - 1) for r, c, _ in (board_blanks or [])}

    gdata_bytes = _get_gdata_bytes(gaddag._data)

    # C traversal: returns list of (word, row_1idx, col_1idx, is_horiz, blanks_list)
    raw_moves = _accel.find_moves_c(
        gdata_bytes, grid, rack_str,
        dictionary._words, VALID_TWO_LETTER,
    )

    # C scoring: returns move dicts with word, row, col, direction, score, etc.
    moves = _accel.score_moves_c(
        raw_moves, grid, board_blank_set,
        _TV, _BONUS, BINGO_BONUS, RACK_SIZE,
    )

    # Post-validation: reject moves with invalid main/cross words
    validated = []
    for m in moves:
        ok = True
        word_str = m['word']
        r1, c1 = m['row'], m['col']
        r0, c0 = r1 - 1, c1 - 1
        is_h = m['direction'] == 'H'
        wlen = len(word_str)

        # Build full main-axis word (may extend beyond placed word)
        if is_h:
            full_start = c0
            fc = c0 - 1
            while fc >= 0 and grid[r0][fc] is not None:
                full_start = fc
                fc -= 1
            full_end = c0 + wlen - 1
            fc = c0 + wlen
            while fc < 15 and grid[r0][fc] is not None:
                full_end = fc
                fc += 1
            full_chars = []
            for fc in range(full_start, full_end + 1):
                if c0 <= fc < c0 + wlen:
                    full_chars.append(word_str[fc - c0])
                elif grid[r0][fc] is not None:
                    full_chars.append(grid[r0][fc])
                else:
                    ok = False
                    break
        else:
            full_start = r0
            fr = r0 - 1
            while fr >= 0 and grid[fr][c0] is not None:
                full_start = fr
                fr -= 1
            full_end = r0 + wlen - 1
            fr = r0 + wlen
            while fr < 15 and grid[fr][c0] is not None:
                full_end = fr
                fr += 1
            full_chars = []
            for fr in range(full_start, full_end + 1):
                if r0 <= fr < r0 + wlen:
                    full_chars.append(word_str[fr - r0])
                elif grid[fr][c0] is not None:
                    full_chars.append(grid[fr][c0])
                else:
                    ok = False
                    break

        if ok:
            full_word = ''.join(full_chars)
            if len(full_word) >= 2:
                if len(full_word) == 2:
                    ok = full_word in VALID_TWO_LETTER
                else:
                    ok = dictionary.is_valid(full_word)

        # Validate cross-words
        if ok:
            for cw in m.get('crosswords', []):
                cw_word = cw['word']
                if len(cw_word) == 2:
                    if cw_word not in VALID_TWO_LETTER:
                        ok = False
                        break
                elif not dictionary.is_valid(cw_word):
                    ok = False
                    break

        if ok:
            validated.append(m)

    return validated


def find_all_moves_opt(board, gaddag, rack_str: str,
                       board_blanks: List[Tuple[int, int, str]] = None) -> List[Dict]:
    """
    Find all valid moves. Drop-in replacement for GADDAGMoveFinder.find_all_moves().

    Args:
        board: Board object
        gaddag: CompactGADDAG object
        rack_str: Rack string (e.g. 'AEINRST' or 'AEIN?ST')
        board_blanks: List of (row, col, letter) for blanks on board (1-indexed)

    Returns:
        List of move dicts sorted by score descending, same format as GADDAGMoveFinder
    """
    rack_str = rack_str.upper()
    num_blanks = rack_str.count('?')
    rack_letters = rack_str.replace('?', '')

    if not rack_letters and num_blanks == 0:
        return []

    # === Capture everything as locals for the closures ===
    grid = board._grid                     # 0-indexed [r][c] -> letter|None
    gdata = gaddag._data                   # bytearray
    gmv = memoryview(gdata)                # memoryview for zero-copy slicing
    unpack = struct.unpack_from             # single function ref
    bonus_grid = _BONUS                    # pre-built bonus table
    tv = _TV                               # tile value array
    bingo = BINGO_BONUS
    rack_sz = RACK_SIZE
    c2i = _C2I
    i2c = _I2C
    delim_idx = _DELIM
    valid_2 = VALID_TWO_LETTER
    dictionary = _get_dict()
    board_blank_set = {(r - 1, c - 1) for r, c, _ in (board_blanks or [])}

    empty_board = all(grid[r][c] is None for r in range(15) for c in range(15))

    # Cross-check cache: (r0, c0, is_horizontal) -> set of valid letters | None | _UNCONSTRAINED
    _UNCONSTRAINED = None  # Means "any letter is OK"
    _NOT_COMPUTED = object()  # Sentinel for "not yet computed"
    cross_cache: Dict = {}

    # Results collection
    moves: List[Dict] = []
    seen: set = set()

    # ------------------------------------------------------------------
    # Inlined helpers (closures capturing locals)
    # ------------------------------------------------------------------

    def get_child(offset: int, char_idx: int) -> int:
        """Return child offset or -1. Inline of CompactGADDAG._get_child."""
        count = gdata[offset] & 0x1F
        off = offset + 1
        end = off + count * 5
        while off < end:
            ci = gdata[off]
            if ci == char_idx:
                return unpack('<I', gdata, off + 1)[0]
            if ci > char_idx:
                return -1
            off += 5
        return -1

    def is_terminal(offset: int) -> bool:
        return gdata[offset] > 127  # bit 7 check without &

    def iter_children(offset: int):
        """Yield (letter, child_offset) pairs."""
        count = gdata[offset] & 0x1F
        off = offset + 1
        end = off + count * 5
        while off < end:
            ci = gdata[off]
            child_off = unpack('<I', gdata, off + 1)[0]
            yield i2c[ci], child_off
            off += 5

    def is_valid_word(word: str) -> bool:
        if len(word) == 2:
            return word in valid_2
        return dictionary.is_valid(word)

    def cross_check(r0: int, c0: int, horiz: bool) -> Optional[Set[str]]:
        """Get valid letters for perpendicular constraint. 0-indexed coords."""
        key = (r0, c0, horiz)
        result = cross_cache.get(key, _NOT_COMPUTED)
        if result is not _NOT_COMPUTED:
            return result

        # Find perpendicular tiles
        above = []
        below = []
        if horiz:
            # Perpendicular is vertical — check above/below
            r = r0 - 1
            while r >= 0 and grid[r][c0] is not None:
                above.append(grid[r][c0])
                r -= 1
            above.reverse()
            r = r0 + 1
            while r < 15 and grid[r][c0] is not None:
                below.append(grid[r][c0])
                r += 1
        else:
            # Perpendicular is horizontal — check left/right
            c = c0 - 1
            while c >= 0 and grid[r0][c] is not None:
                above.append(grid[r0][c])
                c -= 1
            above.reverse()
            c = c0 + 1
            while c < 15 and grid[r0][c] is not None:
                below.append(grid[r0][c])
                c += 1

        if not above and not below:
            cross_cache[key] = None
            return None

        prefix = ''.join(above)
        suffix = ''.join(below)

        valid = set()
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if is_valid_word(prefix + letter + suffix):
                valid.add(letter)

        cross_cache[key] = valid
        return valid

    def score_move(word_chars: list, wlen: int, start_r0: int, start_c0: int,
                   horiz: bool, blanks_set: set) -> Tuple[int, List[Dict]]:
        """
        Inline scoring. All coords 0-indexed.
        Returns (total_score, crosswords_with_scores).
        """
        # Determine new tile positions + score main word in single pass
        main_score = 0
        word_mult = 1
        new_positions = []  # list of (r0, c0, word_index)

        for i in range(wlen):
            if horiz:
                r0, c0 = start_r0, start_c0 + i
            else:
                r0, c0 = start_r0 + i, start_c0

            is_new = grid[r0][c0] is None

            if i in blanks_set:
                lv = 0
            elif not is_new and (r0, c0) in board_blank_set:
                lv = 0
            else:
                lv = tv[ord(word_chars[i]) - 65]

            if is_new:
                new_positions.append((r0, c0, i))
                lm, wm = bonus_grid[r0][c0]
                lv *= lm
                word_mult *= wm

            main_score += lv

        main_score *= word_mult

        # --- Find and score crosswords in single pass ---
        cw_total = 0
        crosswords = []

        for r0, c0, wi in new_positions:
            placed_letter = word_chars[wi]
            is_blank = wi in blanks_set

            if horiz:
                # Perpendicular is vertical — check above/below
                has_perp = (r0 > 0 and grid[r0-1][c0] is not None) or \
                           (r0 < 14 and grid[r0+1][c0] is not None)
            else:
                has_perp = (c0 > 0 and grid[r0][c0-1] is not None) or \
                           (c0 < 14 and grid[r0][c0+1] is not None)

            if not has_perp:
                continue

            # Build and score crossword simultaneously
            cw_horiz = not horiz
            cw_score = 0
            cw_wmult = 1
            cw_chars = []
            cw_start_r, cw_start_c = r0, c0

            if cw_horiz:
                # Look left
                cc = c0 - 1
                pre_chars = []
                while cc >= 0 and grid[r0][cc] is not None:
                    ch = grid[r0][cc]
                    pre_chars.append(ch)
                    clv = 0 if (r0, cc) in board_blank_set else tv[ord(ch) - 65]
                    cw_score += clv
                    cw_start_c = cc
                    cc -= 1
                pre_chars.reverse()
                cw_chars.extend(pre_chars)

                # The placed letter
                cw_chars.append(placed_letter)
                if is_blank:
                    plv = 0
                else:
                    plv = tv[ord(placed_letter) - 65]
                lm, wm = bonus_grid[r0][c0]
                plv *= lm
                cw_wmult *= wm
                cw_score += plv

                # Look right
                cc = c0 + 1
                while cc < 15 and grid[r0][cc] is not None:
                    ch = grid[r0][cc]
                    cw_chars.append(ch)
                    clv = 0 if (r0, cc) in board_blank_set else tv[ord(ch) - 65]
                    cw_score += clv
                    cc += 1
            else:
                # Look up
                rr = r0 - 1
                pre_chars = []
                while rr >= 0 and grid[rr][c0] is not None:
                    ch = grid[rr][c0]
                    pre_chars.append(ch)
                    clv = 0 if (rr, c0) in board_blank_set else tv[ord(ch) - 65]
                    cw_score += clv
                    cw_start_r = rr
                    rr -= 1
                pre_chars.reverse()
                cw_chars.extend(pre_chars)

                # The placed letter
                cw_chars.append(placed_letter)
                if is_blank:
                    plv = 0
                else:
                    plv = tv[ord(placed_letter) - 65]
                lm, wm = bonus_grid[r0][c0]
                plv *= lm
                cw_wmult *= wm
                cw_score += plv

                # Look down
                rr = r0 + 1
                while rr < 15 and grid[rr][c0] is not None:
                    ch = grid[rr][c0]
                    cw_chars.append(ch)
                    clv = 0 if (rr, c0) in board_blank_set else tv[ord(ch) - 65]
                    cw_score += clv
                    rr += 1

            cw_score *= cw_wmult
            cw_total += cw_score

            crosswords.append({
                'word': ''.join(cw_chars),
                'row': cw_start_r + 1,
                'col': cw_start_c + 1,
                'horizontal': cw_horiz,
                'score': cw_score
            })

        total = main_score + cw_total

        # Bingo bonus
        if len(new_positions) >= rack_sz:
            total += bingo

        return total, crosswords

    def record_move(word_chars: list, wlen: int, start_r0: int, start_c0: int,
                    horiz: bool, blanks_used: list):
        """Record a valid move. 0-indexed input, 1-indexed output."""
        # Bounds check
        if start_r0 < 0 or start_c0 < 0:
            return
        if horiz:
            if start_c0 + wlen > 15:
                return
        else:
            if start_r0 + wlen > 15:
                return

        word = ''.join(word_chars)

        # Validate word
        if wlen == 2:
            if word not in valid_2:
                return
        elif not dictionary.is_valid(word):
            return

        # Dedup BEFORE scoring (saves ~64% of scoring work)
        key = (word, start_r0, start_c0, horiz)
        if key in seen:
            return
        seen.add(key)

        # Lightweight connectivity check
        connects = False
        uses_new = False
        for i in range(wlen):
            if horiz:
                r0, c0 = start_r0, start_c0 + i
            else:
                r0, c0 = start_r0 + i, start_c0
            if grid[r0][c0] is not None:
                connects = True
                if uses_new:
                    break  # early exit: both conditions met
            else:
                uses_new = True
                if connects:
                    break  # early exit: both conditions met
                # Check adjacency only until we find a connection
                if ((r0 > 0 and grid[r0-1][c0] is not None) or
                    (r0 < 14 and grid[r0+1][c0] is not None) or
                    (c0 > 0 and grid[r0][c0-1] is not None) or
                    (c0 < 14 and grid[r0][c0+1] is not None)):
                    connects = True
                    break  # early exit
        if not uses_new:
            return
        if empty_board:
            if not any((start_r0 + (0 if horiz else i) == 7 and
                        start_c0 + (i if horiz else 0) == 7) for i in range(wlen)):
                return
        elif not connects:
            return

        blanks_set = set(blanks_used)

        try:
            score, crosswords = score_move(word_chars, wlen, start_r0, start_c0, horiz, blanks_set)
        except Exception:
            return

        moves.append({
            'word': word,
            'row': start_r0 + 1,
            'col': start_c0 + 1,
            'direction': 'H' if horiz else 'V',
            'score': score,
            'crosswords': crosswords,
            'blanks_used': blanks_used
        })

    # ------------------------------------------------------------------
    # Core GADDAG traversal (closures, 0-indexed throughout)
    # ------------------------------------------------------------------

    def extend_right(row0, col0, horiz, offset, wchars, wlen, rack, sr0, sc0,
                     blanks_rem, blanks_used):
        """Extend word rightward. All coords 0-indexed."""
        if row0 < 0 or row0 >= 15 or col0 < 0 or col0 >= 15:
            if is_terminal(offset) and wlen >= 2:
                record_move(wchars, wlen, sr0, sc0, horiz, blanks_used)
            return

        existing = grid[row0][col0]

        if existing is not None:
            idx = c2i.get(existing)
            if idx is not None:
                child = get_child(offset, idx)
                if child >= 0:
                    wchars.append(existing)
                    if horiz:
                        extend_right(row0, col0 + 1, True, child, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem, blanks_used)
                    else:
                        extend_right(row0 + 1, col0, False, child, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem, blanks_used)
                    wchars.pop()
        else:
            cc = cross_check(row0, col0, horiz)

            if is_terminal(offset) and wlen >= 2:
                record_move(wchars, wlen, sr0, sc0, horiz, blanks_used)

            for letter in rack:
                if rack[letter] <= 0:
                    continue
                if cc is not None and letter not in cc:
                    continue
                idx = c2i.get(letter)
                if idx is None:
                    continue
                child = get_child(offset, idx)
                if child < 0:
                    continue

                rack[letter] -= 1
                wchars.append(letter)
                if horiz:
                    extend_right(row0, col0 + 1, True, child, wchars, wlen + 1,
                                 rack, sr0, sc0, blanks_rem, blanks_used)
                else:
                    extend_right(row0 + 1, col0, False, child, wchars, wlen + 1,
                                 rack, sr0, sc0, blanks_rem, blanks_used)
                wchars.pop()
                rack[letter] += 1

            if blanks_rem > 0:
                for letter, child_off in iter_children(offset):
                    if letter == '+':
                        continue
                    if cc is not None and letter not in cc:
                        continue
                    wchars.append(letter)
                    blanks_used_new = blanks_used + [wlen]
                    if horiz:
                        extend_right(row0, col0 + 1, True, child_off, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem - 1, blanks_used_new)
                    else:
                        extend_right(row0 + 1, col0, False, child_off, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem - 1, blanks_used_new)
                    wchars.pop()

    def gen_left_part(anchor_r0, anchor_c0, horiz, offset, wchars, wlen, rack,
                      limit, blanks_rem, blanks_used):
        """Generate left part of word before anchor. 0-indexed."""
        # Try crossing delimiter
        delim_child = get_child(offset, delim_idx)
        if delim_child >= 0:
            if horiz:
                sr0 = anchor_r0
                sc0 = anchor_c0 - wlen
            else:
                sr0 = anchor_r0 - wlen
                sc0 = anchor_c0

            fixed_blanks = [wlen + bi if bi < 0 else bi for bi in blanks_used]
            extend_right(anchor_r0, anchor_c0, horiz, delim_child,
                         wchars[:], wlen, rack, sr0, sc0, blanks_rem, fixed_blanks)

        # Try extending left
        if limit > 0:
            if horiz:
                pr0 = anchor_r0
                pc0 = anchor_c0 - wlen - 1
            else:
                pr0 = anchor_r0 - wlen - 1
                pc0 = anchor_c0

            if pr0 < 0 or pc0 < 0:
                return

            cc = cross_check(pr0, pc0, horiz)
            pos_idx = -(wlen + 1)

            for letter, idx in rack_letter_indices:
                if rack[letter] <= 0:
                    continue
                if cc is not None and letter not in cc:
                    continue
                child = get_child(offset, idx)
                if child < 0:
                    continue

                rack[letter] -= 1
                new_wchars = [letter] + wchars
                gen_left_part(anchor_r0, anchor_c0, horiz, child, new_wchars,
                              wlen + 1, rack, limit - 1, blanks_rem, blanks_used)
                rack[letter] += 1

            if blanks_rem > 0:
                for letter, child_off in iter_children(offset):
                    if letter == '+':
                        continue
                    if cc is not None and letter not in cc:
                        continue
                    new_wchars = [letter] + wchars
                    gen_left_part(anchor_r0, anchor_c0, horiz, child_off, new_wchars,
                                  wlen + 1, rack, limit - 1, blanks_rem - 1,
                                  blanks_used + [pos_idx])

        # Try placing the first letter AT the anchor (not left of it).
        # Must run whenever we haven't placed any left-part letters yet,
        # REGARDLESS of limit.
        if wlen == 0:
            cc = cross_check(anchor_r0, anchor_c0, horiz)

            for letter, idx in rack_letter_indices:
                if rack[letter] <= 0:
                    continue
                if cc is not None and letter not in cc:
                    continue
                letter_child = get_child(offset, idx)
                if letter_child < 0:
                    continue
                dc = get_child(letter_child, delim_idx)
                if dc < 0:
                    continue

                rack[letter] -= 1
                if horiz:
                    extend_right(anchor_r0, anchor_c0 + 1, True, dc, [letter], 1,
                                 rack, anchor_r0, anchor_c0, blanks_rem, blanks_used[:])
                else:
                    extend_right(anchor_r0 + 1, anchor_c0, False, dc, [letter], 1,
                                 rack, anchor_r0, anchor_c0, blanks_rem, blanks_used[:])
                rack[letter] += 1

            if blanks_rem > 0:
                for letter, letter_child in iter_children(offset):
                    if letter == '+':
                        continue
                    if cc is not None and letter not in cc:
                        continue
                    dc = get_child(letter_child, delim_idx)
                    if dc < 0:
                        continue
                    if horiz:
                        extend_right(anchor_r0, anchor_c0 + 1, True, dc, [letter], 1,
                                     rack, anchor_r0, anchor_c0, blanks_rem - 1, [0])
                    else:
                        extend_right(anchor_r0 + 1, anchor_c0, False, dc, [letter], 1,
                                     rack, anchor_r0, anchor_c0, blanks_rem - 1, [0])

    def extend_from_existing(anchor_r0, anchor_c0, horiz, rack_counter, blanks_rem):
        """Extend from existing tiles on board. 0-indexed."""
        prefix = []
        if horiz:
            c = anchor_c0 - 1
            while c >= 0 and grid[anchor_r0][c] is not None:
                prefix.append(grid[anchor_r0][c])
                c -= 1
            prefix.reverse()
            sc0 = c + 1
            sr0 = anchor_r0
        else:
            r = anchor_r0 - 1
            while r >= 0 and grid[r][anchor_c0] is not None:
                prefix.append(grid[r][anchor_c0])
                r -= 1
            prefix.reverse()
            sr0 = r + 1
            sc0 = anchor_c0

        # Navigate GADDAG with reversed prefix
        reversed_prefix = prefix[::-1]
        offset = 0  # root
        for ch in reversed_prefix:
            idx = c2i.get(ch)
            if idx is None:
                return
            child = get_child(offset, idx)
            if child < 0:
                return
            offset = child

        # Cross delimiter
        dc = get_child(offset, delim_idx)
        if dc < 0:
            return

        extend_right(anchor_r0, anchor_c0, horiz, dc, list(prefix), len(prefix),
                     rack_counter, sr0, sc0, blanks_rem, [])

    # ------------------------------------------------------------------
    # Find anchors (0-indexed)
    # ------------------------------------------------------------------

    if empty_board:
        anchors_0 = [(CENTER_ROW - 1, CENTER_COL - 1)]
    else:
        anchors_0 = []
        for r in range(15):
            row = grid[r]
            for c in range(15):
                if row[c] is None:
                    if ((r > 0 and grid[r-1][c] is not None) or
                        (r < 14 and grid[r+1][c] is not None) or
                        (c > 0 and row[c-1] is not None) or
                        (c < 14 and row[c+1] is not None)):
                        anchors_0.append((r, c))

    # ------------------------------------------------------------------
    # Left limit calculation (0-indexed)
    # ------------------------------------------------------------------

    def left_limit(anchor_r0, anchor_c0, horiz):
        """How far left/up from anchor (0-indexed)."""
        limit = 0
        if horiz:
            r0 = anchor_r0
            c = anchor_c0 - 1
            while c >= 0:
                if grid[r0][c] is not None:
                    break
                # Stop at another anchor (has adjacent tile)
                if ((r0 > 0 and grid[r0-1][c] is not None) or
                    (r0 < 14 and grid[r0+1][c] is not None) or
                    (c > 0 and grid[r0][c-1] is not None) or
                    (c < 14 and grid[r0][c+1] is not None)):
                    break
                limit += 1
                c -= 1
        else:
            c0 = anchor_c0
            r = anchor_r0 - 1
            while r >= 0:
                if grid[r][c0] is not None:
                    break
                if ((r > 0 and grid[r-1][c0] is not None) or
                    (r < 14 and grid[r+1][c0] is not None) or
                    (c0 > 0 and grid[r][c0-1] is not None) or
                    (c0 < 14 and grid[r][c0+1] is not None)):
                    break
                limit += 1
                r -= 1
        return limit

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    # Pre-compute rack letter -> GADDAG index pairs (avoid dict.get in hot loop)
    rack_counter = Counter(rack_letters)
    rack_letter_indices = [(letter, c2i[letter]) for letter in rack_counter]

    for ar0, ac0 in anchors_0:
        for horiz in (True, False):
            # Check for existing tile to the left/above
            if horiz:
                has_left = ac0 > 0 and grid[ar0][ac0 - 1] is not None
            else:
                has_left = ar0 > 0 and grid[ar0 - 1][ac0] is not None

            if has_left:
                extend_from_existing(ar0, ac0, horiz, rack_counter, num_blanks)
            else:
                ll = left_limit(ar0, ac0, horiz)
                gen_left_part(ar0, ac0, horiz, 0, [], 0, rack_counter,
                              ll, num_blanks, [])

    moves.sort(key=lambda m: -m['score'])
    return moves


def find_best_score_opt(grid, gdata, rack_str, board_blank_set,
                        cross_cache=None, dictionary=None, valid_2=None):
    """Find highest-scoring move for a rack. Optimized for MC simulations.

    Structurally identical GADDAG traversal to find_all_moves_opt() but only
    tracks the single best score. No dict construction, no dedup set, no
    sorting, no crossword list — just score comparison.

    Args:
        grid:            board._grid (0-indexed 15x15)
        gdata:           gaddag._data (bytearray)
        rack_str:        e.g. 'AEINRST' or 'AEIN?ST'
        board_blank_set: {(r0, c0)} for blanks on board (0-indexed)
        cross_cache:     optional shared dict, persists across calls for same board
        dictionary:      optional pre-loaded Dictionary instance
        valid_2:         optional pre-loaded VALID_TWO_LETTER set

    Returns:
        (best_score, word, row1, col1, dir_str) or (0, None, 0, 0, None)
    """
    rack_str = rack_str.upper()
    num_blanks = rack_str.count('?')
    rack_letters = rack_str.replace('?', '')

    if not rack_letters and num_blanks == 0:
        return (0, None, 0, 0, None)

    # === Capture everything as locals for the closures ===
    unpack = struct.unpack_from
    bonus_grid = _BONUS
    tv = _TV
    bingo = BINGO_BONUS
    rack_sz = RACK_SIZE
    c2i = _C2I
    i2c = _I2C
    delim_idx = _DELIM
    if valid_2 is None:
        valid_2 = VALID_TWO_LETTER
    if dictionary is None:
        dictionary = _get_dict()

    empty_board = all(grid[r][c] is None for r in range(15) for c in range(15))

    # Cross-check cache: reuse across calls for same board state
    _UNCONSTRAINED = None
    _NOT_COMPUTED = object()
    if cross_cache is None:
        cross_cache = {}

    # Best move tracking (mutable list for closure access)
    best = [0, None, 0, 0, None]  # [score, word, row1, col1, dir_str]

    # ------------------------------------------------------------------
    # Inlined helpers (closures capturing locals)
    # ------------------------------------------------------------------

    def get_child(offset, char_idx):
        count = gdata[offset] & 0x1F
        off = offset + 1
        end = off + count * 5
        while off < end:
            ci = gdata[off]
            if ci == char_idx:
                return unpack('<I', gdata, off + 1)[0]
            if ci > char_idx:
                return -1
            off += 5
        return -1

    def is_terminal(offset):
        return gdata[offset] > 127

    def iter_children(offset):
        count = gdata[offset] & 0x1F
        off = offset + 1
        end = off + count * 5
        while off < end:
            ci = gdata[off]
            child_off = unpack('<I', gdata, off + 1)[0]
            yield i2c[ci], child_off
            off += 5

    def is_valid_word(word):
        if len(word) == 2:
            return word in valid_2
        return dictionary.is_valid(word)

    def cross_check(r0, c0, horiz):
        key = (r0, c0, horiz)
        result = cross_cache.get(key, _NOT_COMPUTED)
        if result is not _NOT_COMPUTED:
            return result

        above = []
        below = []
        if horiz:
            r = r0 - 1
            while r >= 0 and grid[r][c0] is not None:
                above.append(grid[r][c0])
                r -= 1
            above.reverse()
            r = r0 + 1
            while r < 15 and grid[r][c0] is not None:
                below.append(grid[r][c0])
                r += 1
        else:
            c = c0 - 1
            while c >= 0 and grid[r0][c] is not None:
                above.append(grid[r0][c])
                c -= 1
            above.reverse()
            c = c0 + 1
            while c < 15 and grid[r0][c] is not None:
                below.append(grid[r0][c])
                c += 1

        if not above and not below:
            cross_cache[key] = None
            return None

        prefix = ''.join(above)
        suffix = ''.join(below)

        valid = set()
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if is_valid_word(prefix + letter + suffix):
                valid.add(letter)

        cross_cache[key] = valid
        return valid

    def score_and_compare(word_chars, wlen, start_r0, start_c0, horiz, blanks_set):
        """Inline scoring, compare to best. No dict/list construction."""
        main_score = 0
        word_mult = 1
        new_count = 0
        cw_total = 0

        for i in range(wlen):
            if horiz:
                r0, c0 = start_r0, start_c0 + i
            else:
                r0, c0 = start_r0 + i, start_c0

            is_new = grid[r0][c0] is None

            if i in blanks_set:
                lv = 0
            elif not is_new and (r0, c0) in board_blank_set:
                lv = 0
            else:
                lv = tv[ord(word_chars[i]) - 65]

            if is_new:
                new_count += 1
                lm, wm = bonus_grid[r0][c0]
                lv *= lm
                word_mult *= wm

                # Inline crossword scoring (no list, just total)
                if horiz:
                    has_perp = (r0 > 0 and grid[r0-1][c0] is not None) or \
                               (r0 < 14 and grid[r0+1][c0] is not None)
                else:
                    has_perp = (c0 > 0 and grid[r0][c0-1] is not None) or \
                               (c0 < 14 and grid[r0][c0+1] is not None)

                if has_perp:
                    cw_s = 0
                    cw_wmult = 1
                    if horiz:
                        # Vertical crossword
                        rr = r0 - 1
                        while rr >= 0 and grid[rr][c0] is not None:
                            cw_s += 0 if (rr, c0) in board_blank_set else tv[ord(grid[rr][c0]) - 65]
                            rr -= 1
                        plv = 0 if i in blanks_set else tv[ord(word_chars[i]) - 65]
                        lm2, wm2 = bonus_grid[r0][c0]
                        cw_s += plv * lm2
                        cw_wmult *= wm2
                        rr = r0 + 1
                        while rr < 15 and grid[rr][c0] is not None:
                            cw_s += 0 if (rr, c0) in board_blank_set else tv[ord(grid[rr][c0]) - 65]
                            rr += 1
                    else:
                        # Horizontal crossword
                        cc = c0 - 1
                        while cc >= 0 and grid[r0][cc] is not None:
                            cw_s += 0 if (r0, cc) in board_blank_set else tv[ord(grid[r0][cc]) - 65]
                            cc -= 1
                        plv = 0 if i in blanks_set else tv[ord(word_chars[i]) - 65]
                        lm2, wm2 = bonus_grid[r0][c0]
                        cw_s += plv * lm2
                        cw_wmult *= wm2
                        cc = c0 + 1
                        while cc < 15 and grid[r0][cc] is not None:
                            cw_s += 0 if (r0, cc) in board_blank_set else tv[ord(grid[r0][cc]) - 65]
                            cc += 1
                    cw_total += cw_s * cw_wmult

            main_score += lv

        total = main_score * word_mult + cw_total
        if new_count >= rack_sz:
            total += bingo

        if total > best[0]:
            word = ''.join(word_chars)
            best[0] = total
            best[1] = word
            best[2] = start_r0 + 1
            best[3] = start_c0 + 1
            best[4] = 'H' if horiz else 'V'

    def try_record_best(word_chars, wlen, start_r0, start_c0, horiz, blanks_used):
        """Validate word and compare score to best. No dict/dedup."""
        if start_r0 < 0 or start_c0 < 0:
            return
        if horiz:
            if start_c0 + wlen > 15:
                return
        else:
            if start_r0 + wlen > 15:
                return

        word = ''.join(word_chars)

        if wlen == 2:
            if word not in valid_2:
                return
        elif not dictionary.is_valid(word):
            return

        # Lightweight connectivity check
        new_count = 0
        connects = False
        for i in range(wlen):
            if horiz:
                r0, c0 = start_r0, start_c0 + i
            else:
                r0, c0 = start_r0 + i, start_c0
            if grid[r0][c0] is not None:
                connects = True
                if new_count > 0:
                    break
            else:
                new_count += 1
                if connects:
                    break
                if ((r0 > 0 and grid[r0-1][c0] is not None) or
                    (r0 < 14 and grid[r0+1][c0] is not None) or
                    (c0 > 0 and grid[r0][c0-1] is not None) or
                    (c0 < 14 and grid[r0][c0+1] is not None)):
                    connects = True
                    break
        if new_count == 0:
            return
        if empty_board:
            if not any((start_r0 + (0 if horiz else i) == 7 and
                        start_c0 + (i if horiz else 0) == 7) for i in range(wlen)):
                return
        elif not connects:
            return

        blanks_set = set(blanks_used)

        try:
            score_and_compare(word_chars, wlen, start_r0, start_c0, horiz, blanks_set)
        except Exception:
            return

    # ------------------------------------------------------------------
    # Core GADDAG traversal (closures, 0-indexed throughout)
    # ------------------------------------------------------------------

    def extend_right(row0, col0, horiz, offset, wchars, wlen, rack, sr0, sc0,
                     blanks_rem, blanks_used):
        if row0 < 0 or row0 >= 15 or col0 < 0 or col0 >= 15:
            if gdata[offset] > 127 and wlen >= 2:
                try_record_best(wchars, wlen, sr0, sc0, horiz, blanks_used)
            return

        existing = grid[row0][col0]

        if existing is not None:
            idx = ord(existing) - 65
            if 0 <= idx < 26:
                # Inline get_child(offset, idx)
                if offset == 0:
                    _child = _root_children.get(idx, -1)
                else:
                    _cnt = gdata[offset] & 0x1F
                    _off = offset + 1
                    _end = _off + _cnt * 5
                    _child = -1
                    while _off < _end:
                        _ci = gdata[_off]
                        if _ci == idx:
                            _child = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                            break
                        if _ci > idx:
                            break
                        _off += 5
                if _child >= 0:
                    wchars.append(existing)
                    if horiz:
                        extend_right(row0, col0 + 1, True, _child, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem, blanks_used)
                    else:
                        extend_right(row0 + 1, col0, False, _child, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem, blanks_used)
                    wchars.pop()
        else:
            cc = cross_check(row0, col0, horiz)

            if gdata[offset] > 127 and wlen >= 2:
                try_record_best(wchars, wlen, sr0, sc0, horiz, blanks_used)

            for letter in rack:
                if rack[letter] <= 0:
                    continue
                if cc is not None and letter not in cc:
                    continue
                idx = ord(letter) - 65
                if idx < 0 or idx >= 26:
                    continue
                # Inline get_child(offset, idx)
                if offset == 0:
                    _child = _root_children.get(idx, -1)
                else:
                    _cnt = gdata[offset] & 0x1F
                    _off = offset + 1
                    _end = _off + _cnt * 5
                    _child = -1
                    while _off < _end:
                        _ci = gdata[_off]
                        if _ci == idx:
                            _child = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                            break
                        if _ci > idx:
                            break
                        _off += 5
                if _child < 0:
                    continue

                rack[letter] -= 1
                wchars.append(letter)
                if horiz:
                    extend_right(row0, col0 + 1, True, _child, wchars, wlen + 1,
                                 rack, sr0, sc0, blanks_rem, blanks_used)
                else:
                    extend_right(row0 + 1, col0, False, _child, wchars, wlen + 1,
                                 rack, sr0, sc0, blanks_rem, blanks_used)
                wchars.pop()
                rack[letter] += 1

            if blanks_rem > 0:
                # Inline iter_children(offset)
                _cnt = gdata[offset] & 0x1F
                _off = offset + 1
                for _j in range(_cnt):
                    _ci = gdata[_off]
                    if _ci == 26:  # delimiter
                        _off += 5
                        continue
                    _letter = _I2C_LIST[_ci]
                    _child_off = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                    _off += 5
                    if cc is not None and _letter not in cc:
                        continue
                    wchars.append(_letter)
                    blanks_used_new = blanks_used + [wlen]
                    if horiz:
                        extend_right(row0, col0 + 1, True, _child_off, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem - 1, blanks_used_new)
                    else:
                        extend_right(row0 + 1, col0, False, _child_off, wchars, wlen + 1,
                                     rack, sr0, sc0, blanks_rem - 1, blanks_used_new)
                    wchars.pop()

    def gen_left_part(anchor_r0, anchor_c0, horiz, offset, wchars, wlen, rack,
                      limit, blanks_rem, blanks_used):
        # Inline get_child(offset, delim_idx)
        if offset == 0:
            delim_child = _root_children.get(delim_idx, -1)
        else:
            _cnt = gdata[offset] & 0x1F
            _off = offset + 1
            _end = _off + _cnt * 5
            delim_child = -1
            while _off < _end:
                _ci = gdata[_off]
                if _ci == delim_idx:
                    delim_child = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                    break
                if _ci > delim_idx:
                    break
                _off += 5
        if delim_child >= 0:
            if horiz:
                sr0 = anchor_r0
                sc0 = anchor_c0 - wlen
            else:
                sr0 = anchor_r0 - wlen
                sc0 = anchor_c0

            fixed_blanks = [wlen + bi if bi < 0 else bi for bi in blanks_used]
            extend_right(anchor_r0, anchor_c0, horiz, delim_child,
                         wchars[:], wlen, rack, sr0, sc0, blanks_rem, fixed_blanks)

        if limit > 0:
            if horiz:
                pr0 = anchor_r0
                pc0 = anchor_c0 - wlen - 1
            else:
                pr0 = anchor_r0 - wlen - 1
                pc0 = anchor_c0

            if pr0 < 0 or pc0 < 0:
                return

            cc = cross_check(pr0, pc0, horiz)
            pos_idx = -(wlen + 1)

            for letter, idx in rack_letter_indices:
                if rack[letter] <= 0:
                    continue
                if cc is not None and letter not in cc:
                    continue
                # Inline get_child(offset, idx)
                if offset == 0:
                    _child = _root_children.get(idx, -1)
                else:
                    _cnt = gdata[offset] & 0x1F
                    _off = offset + 1
                    _end = _off + _cnt * 5
                    _child = -1
                    while _off < _end:
                        _ci = gdata[_off]
                        if _ci == idx:
                            _child = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                            break
                        if _ci > idx:
                            break
                        _off += 5
                if _child < 0:
                    continue

                rack[letter] -= 1
                new_wchars = [letter] + wchars
                gen_left_part(anchor_r0, anchor_c0, horiz, _child, new_wchars,
                              wlen + 1, rack, limit - 1, blanks_rem, blanks_used)
                rack[letter] += 1

            if blanks_rem > 0:
                # Inline iter_children(offset)
                _cnt = gdata[offset] & 0x1F
                _off = offset + 1
                for _j in range(_cnt):
                    _ci = gdata[_off]
                    if _ci == 26:  # delimiter
                        _off += 5
                        continue
                    _letter = _I2C_LIST[_ci]
                    _child_off = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                    _off += 5
                    if cc is not None and _letter not in cc:
                        continue
                    new_wchars = [_letter] + wchars
                    gen_left_part(anchor_r0, anchor_c0, horiz, _child_off, new_wchars,
                                  wlen + 1, rack, limit - 1, blanks_rem - 1,
                                  blanks_used + [pos_idx])

        if wlen == 0 and limit == 0:
            cc = cross_check(anchor_r0, anchor_c0, horiz)

            for letter, idx in rack_letter_indices:
                if rack[letter] <= 0:
                    continue
                if cc is not None and letter not in cc:
                    continue
                # offset is always 0 here — use root cache
                letter_child = _root_children.get(idx, -1)
                if letter_child < 0:
                    continue
                # level-1 node — use level2 cache
                dc = _level2_cache.get((idx, delim_idx), -1)
                if dc < 0:
                    continue

                rack[letter] -= 1
                if horiz:
                    extend_right(anchor_r0, anchor_c0 + 1, True, dc, [letter], 1,
                                 rack, anchor_r0, anchor_c0, blanks_rem, blanks_used[:])
                else:
                    extend_right(anchor_r0 + 1, anchor_c0, False, dc, [letter], 1,
                                 rack, anchor_r0, anchor_c0, blanks_rem, blanks_used[:])
                rack[letter] += 1

            if blanks_rem > 0:
                # Inline iter_children at root — use _root_children directly
                for _ci, _letter_child in _root_children.items():
                    if _ci == 26:  # delimiter
                        continue
                    _letter = _I2C_LIST[_ci]
                    if cc is not None and _letter not in cc:
                        continue
                    # level-1 node — use level2 cache
                    dc = _level2_cache.get((_ci, delim_idx), -1)
                    if dc < 0:
                        continue
                    if horiz:
                        extend_right(anchor_r0, anchor_c0 + 1, True, dc, [_letter], 1,
                                     rack, anchor_r0, anchor_c0, blanks_rem - 1, [0])
                    else:
                        extend_right(anchor_r0 + 1, anchor_c0, False, dc, [_letter], 1,
                                     rack, anchor_r0, anchor_c0, blanks_rem - 1, [0])

    def extend_from_existing(anchor_r0, anchor_c0, horiz, rack_counter, blanks_rem):
        prefix = []
        if horiz:
            c = anchor_c0 - 1
            while c >= 0 and grid[anchor_r0][c] is not None:
                prefix.append(grid[anchor_r0][c])
                c -= 1
            prefix.reverse()
            sc0 = c + 1
            sr0 = anchor_r0
        else:
            r = anchor_r0 - 1
            while r >= 0 and grid[r][anchor_c0] is not None:
                prefix.append(grid[r][anchor_c0])
                r -= 1
            prefix.reverse()
            sr0 = r + 1
            sc0 = anchor_c0

        reversed_prefix = prefix[::-1]
        offset = 0
        for ch in reversed_prefix:
            idx = ord(ch) - 65
            if idx < 0 or idx >= 26:
                return
            # Inline get_child with root cache for offset==0
            if offset == 0:
                _child = _root_children.get(idx, -1)
            else:
                _cnt = gdata[offset] & 0x1F
                _off = offset + 1
                _end = _off + _cnt * 5
                _child = -1
                while _off < _end:
                    _ci = gdata[_off]
                    if _ci == idx:
                        _child = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                        break
                    if _ci > idx:
                        break
                    _off += 5
            if _child < 0:
                return
            offset = _child

        # Inline get_child(offset, delim_idx)
        _cnt = gdata[offset] & 0x1F
        _off = offset + 1
        _end = _off + _cnt * 5
        dc = -1
        while _off < _end:
            _ci = gdata[_off]
            if _ci == delim_idx:
                dc = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
                break
            if _ci > delim_idx:
                break
            _off += 5
        if dc < 0:
            return

        extend_right(anchor_r0, anchor_c0, horiz, dc, list(prefix), len(prefix),
                     rack_counter, sr0, sc0, blanks_rem, [])

    # ------------------------------------------------------------------
    # Find anchors (0-indexed)
    # ------------------------------------------------------------------

    if empty_board:
        anchors_0 = [(7, 7)]
    else:
        anchors_0 = []
        for r in range(15):
            row = grid[r]
            for c in range(15):
                if row[c] is None:
                    if ((r > 0 and grid[r-1][c] is not None) or
                        (r < 14 and grid[r+1][c] is not None) or
                        (c > 0 and row[c-1] is not None) or
                        (c < 14 and row[c+1] is not None)):
                        anchors_0.append((r, c))

    # ------------------------------------------------------------------
    # Left limit calculation (0-indexed)
    # ------------------------------------------------------------------

    def left_limit(anchor_r0, anchor_c0, horiz):
        limit = 0
        if horiz:
            r0 = anchor_r0
            c = anchor_c0 - 1
            while c >= 0:
                if grid[r0][c] is not None:
                    break
                if ((r0 > 0 and grid[r0-1][c] is not None) or
                    (r0 < 14 and grid[r0+1][c] is not None) or
                    (c > 0 and grid[r0][c-1] is not None) or
                    (c < 14 and grid[r0][c+1] is not None)):
                    break
                limit += 1
                c -= 1
        else:
            c0 = anchor_c0
            r = anchor_r0 - 1
            while r >= 0:
                if grid[r][c0] is not None:
                    break
                if ((r > 0 and grid[r-1][c0] is not None) or
                    (r < 14 and grid[r+1][c0] is not None) or
                    (c0 > 0 and grid[r][c0-1] is not None) or
                    (c0 < 14 and grid[r][c0+1] is not None)):
                    break
                limit += 1
                r -= 1
        return limit

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    rack_counter = Counter(rack_letters)
    rack_letter_indices = [(letter, ord(letter) - 65) for letter in rack_counter]

    # ------------------------------------------------------------------
    # Shallow GADDAG cache: root (level 0) and level 1 children
    # Avoids repeated linear scans of the largest nodes (~27 children each)
    # ------------------------------------------------------------------
    _root_children = {}
    _cnt = gdata[0] & 0x1F
    _off = 1
    for _ in range(_cnt):
        _ci = gdata[_off]
        _child = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
        _root_children[_ci] = _child
        _off += 5

    _level2_cache = {}
    for _pci, _poff in _root_children.items():
        _cnt = gdata[_poff] & 0x1F
        _off = _poff + 1
        for _ in range(_cnt):
            _ci = gdata[_off]
            _child = gdata[_off+1] | (gdata[_off+2] << 8) | (gdata[_off+3] << 16) | (gdata[_off+4] << 24)
            _level2_cache[(_pci, _ci)] = _child
            _off += 5

    for ar0, ac0 in anchors_0:
        for horiz in (True, False):
            if horiz:
                has_left = ac0 > 0 and grid[ar0][ac0 - 1] is not None
            else:
                has_left = ar0 > 0 and grid[ar0 - 1][ac0] is not None

            if has_left:
                extend_from_existing(ar0, ac0, horiz, rack_counter, num_blanks)
            else:
                ll = left_limit(ar0, ac0, horiz)
                gen_left_part(ar0, ac0, horiz, 0, [], 0, rack_counter,
                              ll, num_blanks, [])

    return (best[0], best[1], best[2], best[3], best[4])
