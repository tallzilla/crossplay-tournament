"""
Real word-based risk calculation for opened bonus squares.
Detects multi-bonus threats (e.g., 3L + 3W combo).
"""

from collections import Counter
import math
from typing import Tuple, List, Optional, Set

from engine import config

def calculate_real_risk(
    board, 
    move: dict,
    unseen: Counter,
    dictionary,
    bonus_squares: dict,
    tile_values: dict,
    blocked_cache=None
) -> Tuple[str, float, int, List[dict]]:
    """
    Calculate realistic expected damage from opened bonus squares.
    
    Args:
        blocked_cache: Optional BlockedSquareCache for fast lookups.
                      If provided, skips known-blocked squares.
    
    Returns:
        (risk_string, expected_damage, max_damage, threats_list)
    """
    # Clear _calc_prob cache at the start of each move evaluation
    # (unseen changes per move, so cached values from prior moves are stale)
    _calc_prob._cache_owner = getattr(_calc_prob, '_cache_owner', None)
    cache_key_check = id(unseen)
    if _calc_prob._cache_owner != cache_key_check:
        if hasattr(_calc_prob, '__wrapped_cache'):
            _calc_prob.__wrapped_cache.clear()
        _calc_prob._cache_owner = cache_key_check
    
    word = move['word']
    row, col = move['row'], move['col']
    horiz = move['direction'] == 'H'
    
    # Simulate board after move — use direct grid access for speed
    sim_tiles = {}
    for i, letter in enumerate(word):
        if horiz:
            sim_tiles[(row, col + i)] = letter
        else:
            sim_tiles[(row + i, col)] = letter
    
    # Fast get_tile using direct board grid access (avoids method call overhead)
    board_grid = board._grid if hasattr(board, '_grid') else None
    if board_grid is not None:
        def get_tile(r, c):
            if (r, c) in sim_tiles:
                return sim_tiles[(r, c)]
            if 1 <= r <= 15 and 1 <= c <= 15:
                v = board_grid[r-1][c-1]
                return v if v and v != '.' else None
            return None
    else:
        def get_tile(r, c):
            if (r, c) in sim_tiles:
                return sim_tiles[(r, c)]
            return board.get_tile(r, c)
    
    # Find opened squares (empty and adjacent to our word)
    word_squares = set(sim_tiles.keys())
    opened_squares = set()
    
    for (r, c) in word_squares:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr <= 15 and 1 <= nc <= 15:
                if (nr, nc) not in word_squares and not board.get_tile(nr, nc):
                    opened_squares.add((nr, nc))
    
    if not opened_squares:
        return "-", 0.0, 0, []
    
    # Filter out squares that are blocked by crossword constraints
    # Use cache for O(1) lookup if available, otherwise check each
    playable_squares = set()
    for (r, c) in opened_squares:
        if blocked_cache is not None:
            # Fast path: use cache (but cache doesn't know about sim_tiles)
            # For squares NOT adjacent to sim_tiles, cache is valid
            # For squares adjacent to new word, need fresh check
            if (r, c) in word_squares:
                continue  # Part of our word, not opened
            
            # Check if any adjacent to simulated tiles
            adjacent_to_new = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in sim_tiles:
                    adjacent_to_new = True
                    break
            
            if adjacent_to_new:
                # Must check with simulated board
                if _is_square_playable(r, c, get_tile, dictionary):
                    playable_squares.add((r, c))
            else:
                # Can use cache
                if not blocked_cache.is_blocked(r, c):
                    playable_squares.add((r, c))
        else:
            # No cache - check each square
            if _is_square_playable(r, c, get_tile, dictionary):
                playable_squares.add((r, c))
    
    if not playable_squares:
        return "-", 0.0, 0, []
    
    opened_squares = playable_squares
    
    total_unseen = sum(unseen.values())
    if total_unseen == 0:
        return "-", 0.0, 0, []
    hand_size = min(7, total_unseen)

    # Group opened squares by column and row
    by_col = {}
    by_row = {}
    
    for (r, c) in opened_squares:
        by_col.setdefault(c, []).append(r)
        by_row.setdefault(r, []).append(c)
    
    all_threats = []
    
    # Find vertical threats
    for col_num, opened_rows in by_col.items():
        # Skip column if all bonus squares in it are blocked/occupied
        if blocked_cache is not None:
            col_has_playable_bonus = False
            for r in range(1, 16):
                if (r, col_num) in bonus_squares:
                    if not blocked_cache.is_unavailable(r, col_num):
                        col_has_playable_bonus = True
                        break
            if not col_has_playable_bonus:
                continue  # Skip this column entirely
        
        constraints = {}
        for r in range(1, 16):
            if col_num > 1:
                left = get_tile(r, col_num - 1)
                if left:
                    constraints[r] = ('left', left)
            if col_num < 15 and r not in constraints:
                right = get_tile(r, col_num + 1)
                if right:
                    constraints[r] = ('right', right)
        
        bonuses = [(r, bonus_squares.get((r, col_num))) 
                   for r in range(1, 16) 
                   if bonus_squares.get((r, col_num)) in ('3W', '2W', '3L', '2L')]
        
        threats = _find_vertical_threats(
            get_tile, col_num, opened_rows, constraints, bonuses,
            unseen, total_unseen, dictionary, bonus_squares, tile_values,
            blocked_cache, hand_size=hand_size
        )
        all_threats.extend(threats)

    # Find horizontal threats
    for row_num, opened_cols in by_row.items():
        # Skip row if all bonus squares in it are blocked/occupied
        if blocked_cache is not None:
            row_has_playable_bonus = False
            for c in range(1, 16):
                if (row_num, c) in bonus_squares:
                    if not blocked_cache.is_unavailable(row_num, c):
                        row_has_playable_bonus = True
                        break
            if not row_has_playable_bonus:
                continue  # Skip this row entirely
        
        constraints = {}
        for c in range(1, 16):
            if row_num > 1:
                above = get_tile(row_num - 1, c)
                if above:
                    constraints[c] = ('above', above)
            if row_num < 15 and c not in constraints:
                below = get_tile(row_num + 1, c)
                if below:
                    constraints[c] = ('below', below)
        
        bonuses = [(c, bonus_squares.get((row_num, c)))
                   for c in range(1, 16)
                   if bonus_squares.get((row_num, c)) in ('3W', '2W', '3L', '2L')]
        
        threats = _find_horizontal_threats(
            get_tile, row_num, opened_cols, constraints, bonuses,
            unseen, total_unseen, dictionary, bonus_squares, tile_values,
            blocked_cache, hand_size=hand_size
        )
        all_threats.extend(threats)

    # Deduplicate
    seen = set()
    unique_threats = []
    for t in all_threats:
        key = (t['word'], t['row'], t['col'], t['horizontal'])
        if key not in seen:
            seen.add(key)
            unique_threats.append(t)

    if not unique_threats:
        return "-", 0.0, 0, []

    # Sort by EV
    unique_threats.sort(key=lambda x: -x['ev'])

    # Aggregate expected damage across all threats that exploit each bonus
    # square.  A single bonus square reachable from both H and V has roughly
    # the sum of individual EVs (independent events at low probabilities).
    # The worst-case bonus square determines expected_damage.
    bonus_ev = {}  # (r, c) -> cumulative EV from all threats using it
    for t in unique_threats:
        for (r, c, _btype) in t.get('bonus_positions', []):
            bonus_ev[(r, c)] = bonus_ev.get((r, c), 0.0) + t['ev']

    if bonus_ev:
        expected_damage = max(bonus_ev.values())
    else:
        # Threats that don't hit any bonus square -- fall back to top EV
        expected_damage = unique_threats[0]['ev']

    # Max realistic threat (prob > 0.5%)
    realistic = [t for t in unique_threats if t['prob'] > 0.005]
    max_threat = max(realistic, key=lambda t: t['score']) if realistic else unique_threats[0]
    max_damage = max_threat['score']

    # Build risk string showing max potential damage
    bonuses_opened = set()
    for r, c in opened_squares:
        b = bonus_squares.get((r, c))
        if b:
            bonuses_opened.add(b)

    risk_parts = []
    for b in ['3W', '2W', '3L', '2L']:
        if b in bonuses_opened:
            risk_parts.append(f"{b}({max_damage})")
            break  # Just show first major bonus with max damage

    if not risk_parts:
        risk_parts.append(f"({max_damage})")

    risk_str = risk_parts[0]

    # Collect results: top by EV, max threat, and top high-score threats
    result_threats = unique_threats[:config.THREAT_PER_MOVE_TOP_EV]

    # Add high-score realistic threats if not already included
    high_score = sorted(realistic, key=lambda t: -t['score'])
    for t in high_score[:config.THREAT_PER_MOVE_TOP_SCORE]:
        if t not in result_threats:
            result_threats.append(t)

    return risk_str, expected_damage, max_damage, result_threats


def _find_vertical_threats(
    get_tile, col, opened_rows, constraints, bonuses,
    unseen, total_unseen, dictionary, bonus_squares, tile_values,
    blocked_cache=None, hand_size=7
) -> List[dict]:
    """Find vertical words that use opened squares."""
    threats = []
    
    if not constraints:
        return threats
    
    min_opened = min(opened_rows)
    max_opened = max(opened_rows)
    bonus_rows = {r for r, _ in bonuses}
    
    # Precompute cross-check sets: for each constrained position, which letters
    # form valid crosswords? This replaces thousands of per-word is_valid calls.
    cross_valid = {}  # pos -> set of valid letters
    for pos in constraints:
        valid = set()
        side, adj = constraints[pos]
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            # Build the full crossword at this position
            row_at = pos
            # Go left
            cross_letters = []
            c = col - 1
            while c >= 1 and get_tile(row_at, c):
                cross_letters.insert(0, get_tile(row_at, c))
                c -= 1
            cross_letters.append(letter)
            c = col + 1
            while c <= 15 and get_tile(row_at, c):
                cross_letters.append(get_tile(row_at, c))
                c += 1
            cross = ''.join(cross_letters)
            if len(cross) <= 1 or dictionary.is_valid(cross):
                valid.add(letter)
        cross_valid[pos] = valid
    
    for length in range(2, 8):  # threats >7 are rare
        for start_r in range(max(1, min_opened - length + 1), min(16 - length + 1, max_opened + 1)):
            end_r = start_r + length - 1
            if end_r > 15:
                continue
            
            if not any(start_r <= r <= end_r for r in opened_rows):
                continue
            
            # CRITICAL: Check if word would be extended by existing tiles
            # Check tile BEFORE start
            if start_r > 1 and get_tile(start_r - 1, col):
                continue  # Would extend backward - skip this pattern
            # Check tile AFTER end
            if end_r < 15 and get_tile(end_r + 1, col):
                continue  # Would extend forward - skip this pattern
            
            # Check bonuses hittable
            bonuses_hittable = [r for r in bonus_rows if start_r <= r <= end_r]
            
            pattern = []
            positions_needed = []
            skip_pattern = False
            
            for r in range(start_r, end_r + 1):
                t = get_tile(r, col)
                if t:
                    pattern.append(t)
                else:
                    # Skip if this empty position is blocked
                    if blocked_cache is not None and blocked_cache.is_blocked(r, col):
                        skip_pattern = True
                        break
                    pattern.append('?')
                    positions_needed.append(r)
            
            if skip_pattern:
                continue
            
            if not positions_needed:
                continue
            
            # Inject single-letter crossword constraints into pattern
            # before find_words() to narrow search space dramatically
            # (e.g., '??????' -> 'B?????' reduces 16706 -> 1187 matches)
            optimized = list(pattern)
            for r in positions_needed:
                if r in cross_valid and len(cross_valid[r]) == 1:
                    optimized[r - start_r] = next(iter(cross_valid[r]))
            pattern_str = ''.join(optimized)
            matches = dictionary.find_words(pattern_str)

            # Check more matches when hitting multiple bonuses
            # Raise limits for heavily-wildcarded patterns
            wc = pattern_str.count('?')
            wild = wc >= config.THREAT_WILDCARD_THRESHOLD
            if len(bonuses_hittable) >= 2:
                limit = config.THREAT_LIMIT_MULTI_BONUS_WILD if wild else config.THREAT_LIMIT_MULTI_BONUS
            elif bonuses_hittable:
                limit = config.THREAT_LIMIT_SINGLE_BONUS_WILD if wild else config.THREAT_LIMIT_SINGLE_BONUS
            else:
                limit = config.THREAT_LIMIT_NO_BONUS_WILD if wild else config.THREAT_LIMIT_NO_BONUS

            for word in list(matches)[:limit]:
                if not _check_crosswords_fast(word, start_r, positions_needed, cross_valid):
                    continue

                threat = _evaluate_threat(
                    word, start_r, col, positions_needed, constraints, False,
                    unseen, total_unseen, bonus_squares, tile_values,
                    hand_size=hand_size
                )
                if threat:
                    threats.append(threat)

    return threats


def _find_horizontal_threats(
    get_tile, row, opened_cols, constraints, bonuses,
    unseen, total_unseen, dictionary, bonus_squares, tile_values,
    blocked_cache=None, hand_size=7
) -> List[dict]:
    """Find horizontal words that use opened squares."""
    threats = []
    
    if not constraints:
        return threats
    
    min_opened = min(opened_cols)
    max_opened = max(opened_cols)
    bonus_cols = {c for c, _ in bonuses}
    
    # Precompute cross-check sets: for each constrained position, which letters
    # form valid crosswords? Replaces thousands of per-word is_valid calls.
    cross_valid = {}
    for pos in constraints:
        valid = set()
        side, adj = constraints[pos]
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            cross_letters = []
            r = row - 1
            while r >= 1 and get_tile(r, pos):
                cross_letters.insert(0, get_tile(r, pos))
                r -= 1
            cross_letters.append(letter)
            r = row + 1
            while r <= 15 and get_tile(r, pos):
                cross_letters.append(get_tile(r, pos))
                r += 1
            cross = ''.join(cross_letters)
            if len(cross) <= 1 or dictionary.is_valid(cross):
                valid.add(letter)
        cross_valid[pos] = valid
    
    for length in range(2, 8):  # threats >7 are rare
        for start_c in range(max(1, min_opened - length + 1), min(16 - length + 1, max_opened + 1)):
            end_c = start_c + length - 1
            if end_c > 15:
                continue
            
            if not any(start_c <= c <= end_c for c in opened_cols):
                continue
            
            # CRITICAL: Check if word would be extended by existing tiles
            # Check tile BEFORE start
            if start_c > 1 and get_tile(row, start_c - 1):
                continue  # Would extend backward - skip this pattern
            # Check tile AFTER end
            if end_c < 15 and get_tile(row, end_c + 1):
                continue  # Would extend forward - skip this pattern
            
            bonuses_hittable = [c for c in bonus_cols if start_c <= c <= end_c]
            
            pattern = []
            positions_needed = []
            skip_pattern = False
            
            for c in range(start_c, end_c + 1):
                t = get_tile(row, c)
                if t:
                    pattern.append(t)
                else:
                    # Skip if this empty position is blocked
                    if blocked_cache is not None and blocked_cache.is_blocked(row, c):
                        skip_pattern = True
                        break
                    pattern.append('?')
                    positions_needed.append(c)
            
            if skip_pattern:
                continue
            
            if not positions_needed:
                continue
            
            # Inject single-letter crossword constraints into pattern
            optimized = list(pattern)
            for c in positions_needed:
                if c in cross_valid and len(cross_valid[c]) == 1:
                    optimized[c - start_c] = next(iter(cross_valid[c]))
            pattern_str = ''.join(optimized)
            matches = dictionary.find_words(pattern_str)

            # Raise limits for heavily-wildcarded patterns
            wc = pattern_str.count('?')
            wild = wc >= config.THREAT_WILDCARD_THRESHOLD
            if len(bonuses_hittable) >= 2:
                limit = config.THREAT_LIMIT_MULTI_BONUS_WILD if wild else config.THREAT_LIMIT_MULTI_BONUS
            elif bonuses_hittable:
                limit = config.THREAT_LIMIT_SINGLE_BONUS_WILD if wild else config.THREAT_LIMIT_SINGLE_BONUS
            else:
                limit = config.THREAT_LIMIT_NO_BONUS_WILD if wild else config.THREAT_LIMIT_NO_BONUS

            for word in list(matches)[:limit]:
                if not _check_crosswords_fast(word, start_c, positions_needed, cross_valid):
                    continue

                threat = _evaluate_threat(
                    word, row, start_c, positions_needed, constraints, True,
                    unseen, total_unseen, bonus_squares, tile_values,
                    hand_size=hand_size
                )
                if threat:
                    threats.append(threat)

    return threats


def _check_crosswords_fast(word, start, positions_needed, cross_valid):
    """Fast crossword check using precomputed valid letter sets.
    
    cross_valid: dict mapping position -> set of valid letters
    """
    for i, letter in enumerate(word):
        pos = start + i
        if pos not in positions_needed:
            continue
        if pos in cross_valid and letter not in cross_valid[pos]:
            return False
    return True


def _check_crosswords(word, start, positions_needed, constraints, dictionary, vertical, get_tile=None, col_or_row=None):
    """Check if all crosswords formed are valid.
    
    When placing a vertical word, check horizontal crosswords.
    When placing a horizontal word, check vertical crosswords.
    
    If get_tile is provided, we check the FULL crossword formed (extending in both directions).
    Otherwise, fall back to single-letter constraint check.
    """
    for i, letter in enumerate(word):
        pos = start + i
        if pos not in positions_needed:
            continue  # This position already has a tile
            
        if pos not in constraints:
            continue  # No adjacent tile, no crossword formed
        
        side, adj = constraints[pos]
        
        # If we have get_tile, check the FULL crossword
        if get_tile is not None and col_or_row is not None:
            if vertical:
                # Placing vertical word at column col_or_row
                # Check horizontal crossword at row=pos
                row = pos
                col = col_or_row
                
                # Find full horizontal extent
                cross_letters = []
                # Go left
                c = col - 1
                while c >= 1 and get_tile(row, c):
                    cross_letters.insert(0, get_tile(row, c))
                    c -= 1
                # Add our letter
                cross_letters.append(letter)
                # Go right
                c = col + 1
                while c <= 15 and get_tile(row, c):
                    cross_letters.append(get_tile(row, c))
                    c += 1
                
                cross = ''.join(cross_letters)
            else:
                # Placing horizontal word at row=col_or_row
                # Check vertical crossword at col=pos
                row = col_or_row
                col = pos
                
                # Find full vertical extent
                cross_letters = []
                # Go up
                r = row - 1
                while r >= 1 and get_tile(r, col):
                    cross_letters.insert(0, get_tile(r, col))
                    r -= 1
                # Add our letter
                cross_letters.append(letter)
                # Go down
                r = row + 1
                while r <= 15 and get_tile(r, col):
                    cross_letters.append(get_tile(r, col))
                    r += 1
                
                cross = ''.join(cross_letters)
            
            # Only validate if crossword is more than 1 letter
            if len(cross) > 1 and not dictionary.is_valid(cross):
                return False
        else:
            # Fallback: simple 2-letter check
            if vertical:
                cross = adj + letter if side == 'left' else letter + adj
            else:
                cross = adj + letter if side == 'above' else letter + adj
            if not dictionary.is_valid(cross):
                return False
    return True


def _evaluate_threat(
    word, row_or_start, col_or_pos, positions_needed, constraints, horizontal,
    unseen, total_unseen, bonus_squares, tile_values, hand_size=7
) -> dict:
    """Evaluate a potential threat, accounting for blank-only plays."""
    
    if horizontal:
        row = row_or_start
        start_col = col_or_pos
        needed_str = ''.join(word[c - start_col] for c in positions_needed)
    else:
        start_row = row_or_start
        col = col_or_pos
        needed_str = ''.join(word[r - start_row] for r in positions_needed)
    
    needed = Counter(needed_str)
    
    # Check availability - can use real tiles or blanks
    blanks_available = unseen.get('?', 0)
    blanks_needed = 0
    tiles_needing_blank = set()  # Track which tiles must use a blank
    
    for tile, cnt in needed.items():
        real_available = unseen.get(tile, 0)
        if real_available < cnt:
            # Need blanks to cover the shortfall
            shortfall = cnt - real_available
            blanks_needed += shortfall
            if real_available == 0:
                tiles_needing_blank.add(tile)  # ALL instances of this tile need blank
    
    if blanks_needed > blanks_available:
        return None  # Can't play this word
    
    # Score main word - tiles that MUST use blank score 0
    score = 0
    word_mult = 1
    bonuses_used = []
    bonus_positions = []  # (row, col, type) for each bonus square hit

    for i, letter in enumerate(word):
        if horizontal:
            r, c = row, start_col + i
            pos = c
        else:
            r, c = start_row + i, col
            pos = r

        # If this letter must be played with a blank, it scores 0
        if letter in tiles_needing_blank:
            ls = 0
        else:
            ls = tile_values.get(letter, 0)

        if pos in positions_needed:
            bonus = bonus_squares.get((r, c))
            if bonus == '2L':
                ls *= 2
                bonuses_used.append('2L')
                bonus_positions.append((r, c, '2L'))
            elif bonus == '3L':
                ls *= 3
                bonuses_used.append('3L')
                bonus_positions.append((r, c, '3L'))
            elif bonus == '2W':
                word_mult *= 2
                bonuses_used.append('2W')
                bonus_positions.append((r, c, '2W'))
            elif bonus == '3W':
                word_mult *= 3
                bonuses_used.append('3W')
                bonus_positions.append((r, c, '3W'))
        
        score += ls
    
    main_score = score * word_mult
    
    # Crossword scores - also account for blanks
    cross_score = 0
    for i, letter in enumerate(word):
        if horizontal:
            r, c = row, start_col + i
            pos = c
        else:
            r, c = start_row + i, col
            pos = r
        
        if pos in constraints and pos in positions_needed:
            side, adj = constraints[pos]
            adj_val = tile_values.get(adj, 0)
            
            # If this letter must be played with a blank, it scores 0
            if letter in tiles_needing_blank:
                new_val = 0
            else:
                new_val = tile_values.get(letter, 0)
            
            bonus = bonus_squares.get((r, c))
            if bonus == '2L':
                new_val *= 2
            elif bonus == '3L':
                new_val *= 3
            
            cs = adj_val + new_val
            if bonus in ('2W', '3W'):
                cs *= (2 if bonus == '2W' else 3)
            
            cross_score += cs
    
    total_score = main_score + cross_score
    
    prob = _calc_prob(needed_str, unseen, total_unseen, hand_size=hand_size)

    if total_score < config.THREAT_MIN_SCORE or prob < config.THREAT_MIN_PROB:
        return None
    
    if horizontal:
        result_row = row
        result_col = start_col
    else:
        result_row = start_row
        result_col = col
    
    return {
        'word': word,
        'row': result_row,
        'col': result_col,
        'horizontal': horizontal,
        'score': total_score,
        'main': main_score,
        'cross': cross_score,
        'needed': needed_str,
        'prob': prob,
        'ev': total_score * prob,
        'bonuses': list(set(bonuses_used)),
        'bonus_positions': bonus_positions,
    }


def _calc_prob(needed_str, unseen, total_unseen, hand_size=7, _cache={}) -> float:
    """
    Calculate exact hypergeometric probability of opponent having needed tiles.
    
    Uses multivariate hypergeometric distribution with memoization.
    Iterative implementation replaces recursive enumerate_combinations.
    """
    if not needed_str or total_unseen < hand_size or len(needed_str) > hand_size:
        return 0.0
    
    # Memoization key: sorted needed_str + tile availability
    # We use needed_str sorted + avails tuple for cache key
    needed = Counter(needed_str)
    tile_types = sorted(needed.keys())  # Sort for consistent cache key
    avails = tuple(unseen.get(t, 0) for t in tile_types)
    mins = tuple(needed[t] for t in tile_types)
    
    cache_key = (mins, avails, total_unseen, hand_size)
    if cache_key in _cache:
        return _cache[cache_key]
    
    # Check if enough tiles exist
    for avail, min_needed in zip(avails, mins):
        if avail < min_needed:
            _cache[cache_key] = 0.0
            return 0.0
    
    # "Other" tiles = tiles not in the needed set
    other_total = total_unseen - sum(avails)
    
    # Total ways to draw a hand
    total_ways = math.comb(total_unseen, hand_size)
    if total_ways == 0:
        _cache[cache_key] = 0.0
        return 0.0
    
    n_types = len(tile_types)
    
    if n_types == 1:
        # Fast path for single tile type (most common case)
        valid_ways = 0
        for count in range(mins[0], min(avails[0], hand_size) + 1):
            remaining = hand_size - count
            if 0 <= remaining <= other_total:
                valid_ways += math.comb(avails[0], count) * math.comb(other_total, remaining)
    elif n_types == 2:
        # Fast path for two tile types
        valid_ways = 0
        for c0 in range(mins[0], min(avails[0], hand_size) + 1):
            w0 = math.comb(avails[0], c0)
            rem1 = hand_size - c0
            for c1 in range(mins[1], min(avails[1], rem1) + 1):
                remaining = rem1 - c1
                if 0 <= remaining <= other_total:
                    valid_ways += w0 * math.comb(avails[1], c1) * math.comb(other_total, remaining)
    else:
        # General case: iterative with stack (avoids recursive function call overhead)
        valid_ways = 0
        # Stack: (type_index, remaining_hand, accumulated_ways)
        stack = [(0, hand_size, 1)]
        while stack:
            idx, remaining, ways = stack.pop()
            if idx == n_types:
                if 0 <= remaining <= other_total:
                    valid_ways += ways * math.comb(other_total, remaining)
                continue
            min_draw = mins[idx]
            max_draw = min(avails[idx], remaining)
            for count in range(min_draw, max_draw + 1):
                stack.append((idx + 1, remaining - count, ways * math.comb(avails[idx], count)))
    
    prob = valid_ways / total_ways
    _cache[cache_key] = prob
    return prob


def _is_square_playable(r: int, c: int, get_tile, dictionary) -> bool:
    """
    Check if at least one letter can legally be placed at (r, c).
    A letter is legal if it forms valid crosswords with ALL adjacent tiles.
    """
    # Find all adjacent tiles that would form crosswords
    constraints = []
    
    # Check above
    if r > 1:
        above = get_tile(r - 1, c)
        if above:
            constraints.append(('above', above))
    
    # Check below
    if r < 15:
        below = get_tile(r + 1, c)
        if below:
            constraints.append(('below', below))
    
    # Check left
    if c > 1:
        left = get_tile(r, c - 1)
        if left:
            constraints.append(('left', left))
    
    # Check right
    if c < 15:
        right = get_tile(r, c + 1)
        if right:
            constraints.append(('right', right))
    
    if not constraints:
        # No adjacent tiles - any letter can go here
        return True
    
    # Check if any letter satisfies ALL constraints
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        all_valid = True
        for direction, adj_letter in constraints:
            if direction == 'above':
                cross = adj_letter + letter
            elif direction == 'below':
                cross = letter + adj_letter
            elif direction == 'left':
                cross = adj_letter + letter
            elif direction == 'right':
                cross = letter + adj_letter
            
            if not dictionary.is_valid(cross):
                all_valid = False
                break
        
        if all_valid:
            return True
    
    # No letter works - square is unplayable
    return False


def analyze_existing_threats(
    board,
    unseen: Counter,
    dictionary,
    bonus_squares: dict,
    tile_values: dict,
    blocked_cache=None
) -> Tuple[str, float, int, List[dict]]:
    """
    Analyze threats that ALREADY exist on the current board.
    
    These are squares adjacent to existing words that opponent could play on.
    Useful for showing player what vulnerabilities exist before they move.
    
    Returns:
        (risk_string, expected_damage, max_damage, threats_list)
    """
    # Find all empty squares adjacent to existing tiles
    existing_open = set()
    
    for r in range(1, 16):
        for c in range(1, 16):
            if board.get_tile(r, c):
                # This square has a tile - check adjacent empties
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 1 <= nr <= 15 and 1 <= nc <= 15:
                        if not board.get_tile(nr, nc):
                            existing_open.add((nr, nc))
    
    if not existing_open:
        return "-", 0.0, 0, []
    
    def get_tile(r, c):
        return board.get_tile(r, c)
    
    # Filter blocked squares
    playable = set()
    for (r, c) in existing_open:
        if blocked_cache is not None:
            if not blocked_cache.is_blocked(r, c):
                playable.add((r, c))
        else:
            if _is_square_playable(r, c, get_tile, dictionary):
                playable.add((r, c))
    
    if not playable:
        return "-", 0.0, 0, []
    
    total_unseen = sum(unseen.values())
    if total_unseen == 0:
        return "-", 0.0, 0, []
    hand_size = min(7, total_unseen)

    # Group by column and row
    by_col = {}
    by_row = {}

    for (r, c) in playable:
        by_col.setdefault(c, []).append(r)
        by_row.setdefault(r, []).append(c)

    all_threats = []

    # Find vertical threats
    for col_num, rows in by_col.items():
        constraints = {}
        for r in range(1, 16):
            if col_num > 1:
                left = get_tile(r, col_num - 1)
                if left:
                    constraints[r] = ('left', left)
            if col_num < 15 and r not in constraints:
                right = get_tile(r, col_num + 1)
                if right:
                    constraints[r] = ('right', right)

        if not constraints:
            continue

        bonuses = [(r, bonus_squares.get((r, col_num)))
                   for r in range(1, 16)
                   if bonus_squares.get((r, col_num)) in ('3W', '2W', '3L', '2L')]

        threats = _find_vertical_threats(
            get_tile, col_num, rows, constraints, bonuses,
            unseen, total_unseen, dictionary, bonus_squares, tile_values,
            blocked_cache, hand_size=hand_size
        )
        all_threats.extend(threats)

    # Find horizontal threats
    for row_num, cols in by_row.items():
        constraints = {}
        for c in range(1, 16):
            if row_num > 1:
                above = get_tile(row_num - 1, c)
                if above:
                    constraints[c] = ('above', above)
            if row_num < 15 and c not in constraints:
                below = get_tile(row_num + 1, c)
                if below:
                    constraints[c] = ('below', below)

        if not constraints:
            continue

        bonuses = [(c, bonus_squares.get((row_num, c)))
                   for c in range(1, 16)
                   if bonus_squares.get((row_num, c)) in ('3W', '2W', '3L', '2L')]

        threats = _find_horizontal_threats(
            get_tile, row_num, cols, constraints, bonuses,
            unseen, total_unseen, dictionary, bonus_squares, tile_values,
            blocked_cache, hand_size=hand_size
        )
        all_threats.extend(threats)
    
    # Deduplicate
    seen = set()
    unique_threats = []
    for t in all_threats:
        key = (t['word'], t['row'], t['col'], t['horizontal'])
        if key not in seen:
            seen.add(key)
            unique_threats.append(t)
    
    if not unique_threats:
        return "-", 0.0, 0, []

    # Sort by expected value
    unique_threats.sort(key=lambda t: -t['ev'])

    # Calculate summary stats
    max_damage = max(t['score'] for t in unique_threats)

    # Aggregate expected damage by bonus square position.
    # A bonus square attackable from both H and V has combined EV from
    # all threats using it (independent events at low probabilities).
    bonus_ev = {}  # (r, c) -> cumulative EV
    for t in unique_threats:
        for (r, c, _btype) in t.get('bonus_positions', []):
            bonus_ev[(r, c)] = bonus_ev.get((r, c), 0.0) + t['ev']

    if bonus_ev:
        expected_damage = max(bonus_ev.values())
    else:
        expected_damage = unique_threats[0]['ev'] if unique_threats else 0.0

    # Find which bonus squares are exposed
    bonuses_exposed = set()
    for t in unique_threats[:10]:  # Check top threats
        for (r, c, btype) in t.get('bonus_positions', []):
            bonuses_exposed.add(btype)
    
    # Build risk string
    risk_parts = []
    for b in ['3W', '2W', '3L', '2L']:
        if b in bonuses_exposed:
            risk_parts.append(b)
    
    if risk_parts:
        risk_str = ','.join(risk_parts) + f"({max_damage})"
    else:
        risk_str = f"({max_damage})"
    
    # Collect results: top by EV, plus top high-score threats
    result_threats = unique_threats[:config.THREAT_TOP_BY_EV]

    # Add high-score threats not already included
    # (pattern injection finds high-damage plays that have low EV)
    high_score = sorted(unique_threats, key=lambda t: -t['score'])
    added = 0
    for t in high_score:
        if added >= config.THREAT_TOP_BY_SCORE:
            break
        if t not in result_threats:
            result_threats.append(t)
            added += 1

    return risk_str, expected_damage, max_damage, result_threats
