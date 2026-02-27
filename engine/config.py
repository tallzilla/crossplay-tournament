"""
Crossplay Tournament -- Configuration Constants (DO NOT EDIT)

Tile values, distribution, bonus squares, and board layout for Crossplay.
These differ from Scrabble -- see RULES.md for details.
"""

from typing import Dict, List, Tuple

# =============================================================================
# TILE VALUES
# =============================================================================

TILE_VALUES: Dict[str, int] = {
    'A': 1, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 4, 'H': 3,
    'I': 1, 'J': 10, 'K': 6, 'L': 2, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 2, 'V': 6, 'W': 5, 'X': 8,
    'Y': 4, 'Z': 10, '?': 0  # Blank
}

# =============================================================================
# TILE DISTRIBUTION (100 tiles total, 3 blanks)
# =============================================================================

TILE_DISTRIBUTION: Dict[str, int] = {
    'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 3,
    'I': 8, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 5, 'O': 8, 'P': 2,
    'Q': 1, 'R': 6, 'S': 5, 'T': 6, 'U': 3, 'V': 2, 'W': 2, 'X': 1,
    'Y': 2, 'Z': 1, '?': 3
}

TOTAL_TILES: int = sum(TILE_DISTRIBUTION.values())  # 100

# =============================================================================
# BOARD
# =============================================================================

BOARD_SIZE: int = 15
CENTER_ROW: int = 8  # 1-indexed
CENTER_COL: int = 8  # 1-indexed
RACK_SIZE: int = 7
BINGO_BONUS: int = 40  # Bonus for using all 7 tiles (Crossplay uses 40, not 50)

# Bonus squares (all 1-indexed): (row, col) -> bonus_type
BONUS_SQUARES: Dict[Tuple[int, int], str] = {
    # Triple Letter (3L)
    (1, 1): '3L', (1, 15): '3L', (15, 1): '3L', (15, 15): '3L',
    (2, 7): '3L', (2, 9): '3L', (14, 7): '3L', (14, 9): '3L',
    (7, 2): '3L', (9, 2): '3L', (7, 14): '3L', (9, 14): '3L',
    (5, 6): '3L', (5, 10): '3L', (11, 6): '3L', (11, 10): '3L',
    (6, 5): '3L', (6, 11): '3L', (10, 5): '3L', (10, 11): '3L',

    # Double Letter (2L)
    (1, 8): '2L', (15, 8): '2L', (8, 1): '2L', (8, 15): '2L',
    (3, 5): '2L', (3, 11): '2L', (13, 5): '2L', (13, 11): '2L',
    (5, 3): '2L', (5, 13): '2L', (11, 3): '2L', (11, 13): '2L',
    (4, 4): '2L', (4, 12): '2L', (12, 4): '2L', (12, 12): '2L',
    (6, 8): '2L', (10, 8): '2L', (8, 6): '2L', (8, 10): '2L',

    # Triple Word (3W)
    (1, 4): '3W', (1, 12): '3W', (15, 4): '3W', (15, 12): '3W',
    (4, 1): '3W', (4, 15): '3W', (12, 1): '3W', (12, 15): '3W',

    # Double Word (2W)
    (2, 2): '2W', (2, 14): '2W', (14, 2): '2W', (14, 14): '2W',
    (4, 8): '2W', (12, 8): '2W', (8, 4): '2W', (8, 12): '2W',

    # Center square: NO BONUS in Crossplay
}

TRIPLE_WORD_SQUARES: List[Tuple[int, int]] = [
    pos for pos, bonus in BONUS_SQUARES.items() if bonus == '3W'
]
DOUBLE_WORD_SQUARES: List[Tuple[int, int]] = [
    pos for pos, bonus in BONUS_SQUARES.items() if bonus == '2W'
]

# =============================================================================
# VALID 2-LETTER WORDS
# =============================================================================

# =============================================================================
# THREAT ANALYZER CONSTANTS (used by real_risk.py)
# =============================================================================

# Pattern match limits (tiered by bonus count and wildcard density)
THREAT_LIMIT_NO_BONUS = 100
THREAT_LIMIT_NO_BONUS_WILD = 500
THREAT_LIMIT_SINGLE_BONUS = 500
THREAT_LIMIT_SINGLE_BONUS_WILD = 2000
THREAT_LIMIT_MULTI_BONUS = 2000
THREAT_LIMIT_MULTI_BONUS_WILD = 5000
THREAT_WILDCARD_THRESHOLD = 4       # wildcards >= this -> use "wild" limits

# Threat filtering
THREAT_MIN_SCORE = 6                # ignore threats scoring below this
THREAT_MIN_PROB = 0.001             # ignore threats with P < 0.1%

# Result collection
THREAT_TOP_BY_EV = 20               # top threats by expected value (board-wide)
THREAT_TOP_BY_SCORE = 5             # extra top threats by raw score (board-wide)
THREAT_PER_MOVE_TOP_EV = 6          # top threats by EV (per-move)
THREAT_PER_MOVE_TOP_SCORE = 3       # extra top threats by score (per-move)

# =============================================================================
# VALID 2-LETTER WORDS
# =============================================================================

VALID_TWO_LETTER: set = {
    'AA', 'AB', 'AD', 'AE', 'AG', 'AH', 'AI', 'AL', 'AM', 'AN', 'AR', 'AS', 'AT', 'AW', 'AX', 'AY',
    'BA', 'BE', 'BI', 'BO', 'BY',
    'DA', 'DE', 'DO',
    'ED', 'EF', 'EH', 'EL', 'EM', 'EN', 'ER', 'ES', 'ET', 'EW', 'EX',
    'FA', 'FE',
    'GI', 'GO',
    'HA', 'HE', 'HI', 'HM', 'HO',
    'ID', 'IF', 'IN', 'IS', 'IT',
    'JO',
    'KA', 'KI',
    'LA', 'LI', 'LO',
    'MA', 'ME', 'MI', 'MM', 'MO', 'MU', 'MY',
    'NA', 'NE', 'NO', 'NU',
    'OD', 'OE', 'OF', 'OH', 'OI', 'OK', 'OM', 'ON', 'OP', 'OR', 'OS', 'OW', 'OX', 'OY',
    'PA', 'PE', 'PI', 'PO',
    'QI',
    'RE',
    'SH', 'SI', 'SO',
    'TA', 'TE', 'TI', 'TO',
    'UH', 'UM', 'UN', 'UP', 'US', 'UT',
    'WE', 'WO',
    'XI', 'XU',
    'YA', 'YE', 'YO',
    'ZA',
}
