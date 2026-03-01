#!/usr/bin/env python3
"""
Generate Crossplay-calibrated SuperLeaves table from greedy self-play.

Method:
  - Play N greedy games (moves[0] every turn = max raw score)
  - After each move, record (leave_key, score_on_next_turn)
  - Average scores by leave combination
  - Normalize: leave_equity = mean_score(leave) - global_mean_score

The resulting table maps tuple(sorted(leave)) -> equity_value, where
positive = above-average future scoring from this leave.

Usage:
    python generate_leaves.py              # 5000 games (~2-4 min)
    python generate_leaves.py --games 20000  # more data for rare leaves
"""

import os
import sys
import random
import pickle
import time
import argparse
from collections import defaultdict

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from engine.board import Board
from engine.config import TILE_DISTRIBUTION, RACK_SIZE
from bots.base_engine import get_legal_moves

OUTPUT_PATH = os.path.join(_root, 'bots', 'crossplay_leaves.pkl')
MIN_SAMPLES = 5   # minimum observations before trusting an estimate


def make_bag():
    bag = []
    for letter, count in TILE_DISTRIBUTION.items():
        bag.extend([letter] * count)
    random.shuffle(bag)
    return bag


def draw_tiles(bag, rack_len):
    drawn = []
    while rack_len + len(drawn) < RACK_SIZE and bag:
        drawn.append(bag.pop())
    return drawn


def play_data_game(leave_sums, leave_counts):
    """Play one greedy vs greedy game, accumulate (leave, next_turn_score) data."""
    board = Board()
    bag = make_bag()
    blanks_on_board = []

    rack0 = ''.join(draw_tiles(bag, 0))
    rack1 = ''.join(draw_tiles(bag, 0))
    racks = [rack0, rack1]
    scores = [0, 0]

    # pending[p] = leave_key from the previous turn.
    # We close it out at the START of the next turn by recording chosen['score'].
    pending = [None, None]

    final_turns_left = None
    consecutive_passes = 0

    for _ in range(200):   # safety cap (games shouldn't exceed ~40 turns each)
        for player_idx in range(2):
            rack = racks[player_idx]

            # Game-over check
            if final_turns_left is not None:
                if final_turns_left <= 0:
                    return
                final_turns_left -= 1

            # (pending is now closed out below, after choosing the move)

            moves = get_legal_moves(board, rack, blanks_on_board)

            if not moves:
                consecutive_passes += 1
                if pending[player_idx] is not None:
                    # Player passed — counts as 0 score on "next turn"
                    leave_sums[pending[player_idx]] += 0
                    leave_counts[pending[player_idx]] += 1
                    pending[player_idx] = None
                if consecutive_passes >= 4:
                    return
                continue

            consecutive_passes = 0
            chosen = moves[0]   # greedy: highest-scoring move

            # Close out pending from the previous turn: record THIS move's score
            # as what the player earned after holding that leave.
            if pending[player_idx] is not None:
                leave_sums[pending[player_idx]] += chosen['score']
                leave_counts[pending[player_idx]] += 1
                pending[player_idx] = None

            # Record current leave for next-turn lookup
            # (only when bag is non-empty — leave has no drawing value at end)
            if len(bag) > 0:
                leave = chosen.get('leave', '')
                pending[player_idx] = tuple(sorted(leave.upper()))

            # Place move on board
            word, row, col = chosen['word'], chosen['row'], chosen['col']
            horizontal = chosen['direction'] == 'H'
            board.place_word(word, row, col, horizontal)
            scores[player_idx] += chosen['score']

            # Track blanks
            for bi in chosen.get('blanks_used', []):
                if bi < 0:
                    bi = -bi
                if horizontal:
                    blanks_on_board.append((row, col + bi, word[bi]))
                else:
                    blanks_on_board.append((row + bi, col, word[bi]))

            # Update rack: remove played tiles, draw replacements
            tiles_used = chosen.get('tiles_used', list(word))
            new_rack = list(rack)
            for t in tiles_used:
                if t in new_rack:
                    new_rack.remove(t)
                elif '?' in new_rack:
                    new_rack.remove('?')

            drawn = draw_tiles(bag, len(new_rack))
            new_rack.extend(drawn)
            racks[player_idx] = ''.join(new_rack)

            if len(bag) == 0 and final_turns_left is None:
                final_turns_left = 2


def build_table(leave_sums, leave_counts, min_samples):
    """Convert raw (sum, count) data into normalized equity values."""
    # Only use leaves with enough observations
    valid = {k: leave_sums[k] / leave_counts[k]
             for k in leave_sums
             if leave_counts[k] >= min_samples}

    if not valid:
        print("WARNING: No leaves met the minimum sample requirement.")
        return {}

    # Global mean: weighted average across all leave types
    total_weighted = sum(leave_sums[k] for k in valid)
    total_count = sum(leave_counts[k] for k in valid)
    global_mean = total_weighted / total_count

    # Equity = deviation from global mean
    table = {k: v - global_mean for k, v in valid.items()}
    return table


def main():
    parser = argparse.ArgumentParser(description='Generate Crossplay SuperLeaves table')
    parser.add_argument('--games', type=int, default=5000,
                        help='Number of self-play games (default: 5000)')
    args = parser.parse_args()
    n_games = args.games

    print(f"Generating SuperLeaves from {n_games} greedy self-play games...")
    print(f"Building GADDAG (first run ~48s)...")

    leave_sums = defaultdict(float)
    leave_counts = defaultdict(int)

    t_start = time.time()
    for i in range(n_games):
        play_data_game(leave_sums, leave_counts)

        if (i + 1) % max(1, n_games // 20) == 0:
            elapsed = time.time() - t_start
            gps = (i + 1) / elapsed
            n_leaves = sum(1 for k in leave_counts if leave_counts[k] >= MIN_SAMPLES)
            total_obs = sum(leave_counts.values())
            print(f"  [{i+1}/{n_games}] {gps:.1f} games/s | "
                  f"{total_obs:,} observations | "
                  f"{n_leaves} unique leaves (>={MIN_SAMPLES} samples)")

    elapsed = time.time() - t_start
    print(f"\nCompleted {n_games} games in {elapsed:.1f}s ({n_games/elapsed:.1f} games/s)")

    table = build_table(leave_sums, leave_counts, MIN_SAMPLES)

    # Report coverage
    total_leaves = len(leave_counts)
    qualified = len(table)
    total_obs = sum(leave_counts.values())
    print(f"\nLeave statistics:")
    print(f"  Total unique leaves observed: {total_leaves}")
    print(f"  Leaves with >= {MIN_SAMPLES} samples: {qualified}")
    print(f"  Total (leave, score) observations: {total_obs:,}")

    # Show most/least valuable leaves
    sorted_leaves = sorted(table.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 15 most valuable leaves:")
    for k, v in sorted_leaves[:15]:
        print(f"  {''.join(k) or '(empty)':10s}  {v:+.2f}  (n={leave_counts[k]})")
    print(f"\nTop 15 least valuable leaves:")
    for k, v in sorted_leaves[-15:]:
        print(f"  {''.join(k) or '(empty)':10s}  {v:+.2f}  (n={leave_counts[k]})")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(table, f)
    print(f"\nSaved {qualified} leave values to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
