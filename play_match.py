#!/usr/bin/env python3
"""
Crossplay Tournament -- Match Runner

Usage:
    python play_match.py random_bot my_bot              # 10 games, show summary
    python play_match.py random_bot my_bot --games 100  # 100 games
    python play_match.py random_bot my_bot --watch      # 1 game, show board each turn
    python play_match.py --tournament --games 50         # round-robin all bots/
    python play_match.py dadbot my_bot --seed 12345     # reproducible games
    python play_match.py dadbot my_bot --watch --seed 42 # replay a specific game

First run builds the GADDAG (~48 seconds). After that it loads in under 1 second.
"""

import argparse
import importlib
import os
import sys
import random
import time
from collections import Counter

# Add project root to path
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from engine.board import Board
from engine.config import TILE_DISTRIBUTION, TILE_VALUES, RACK_SIZE, BINGO_BONUS
from engine.scoring import calculate_move_score
from bots.base_engine import BaseEngine, get_legal_moves


# =========================================================================
# Game simulation
# =========================================================================

def make_bag():
    """Create a shuffled tile bag."""
    bag = []
    for letter, count in TILE_DISTRIBUTION.items():
        bag.extend([letter] * count)
    random.shuffle(bag)
    return bag


def draw_tiles(bag, rack, count):
    """Draw tiles from the bag to fill the rack."""
    drawn = []
    while len(rack) + len(drawn) < count and bag:
        drawn.append(bag.pop())
    return drawn


def play_game(engine1, engine2, watch=False, seed=None):
    """Play a single game between two engines.

    Args:
        engine1: BaseEngine instance (goes first)
        engine2: BaseEngine instance
        watch: if True, print board after each move
        seed: if provided, seed the RNG for reproducible tile draws

    Returns:
        dict with game results (includes 'seed' for replay)
    """
    if seed is not None:
        random.seed(seed)

    board = Board()
    bag = make_bag()
    blanks_on_board = []

    # Draw initial racks
    rack1 = ''.join(draw_tiles(bag, '', RACK_SIZE))
    rack2 = ''.join(draw_tiles(bag, '', RACK_SIZE))

    score1 = 0
    score2 = 0
    move_number = 0
    consecutive_passes = 0
    final_turns_left = None  # None = mid-game, 2 = bag just emptied, 1/0 = final turns

    engines = [engine1, engine2]
    racks = [rack1, rack2]
    scores = [0, 0]
    # Per-engine timing stats
    move_times = [[], []]  # list of pick_move durations per engine

    if watch:
        print(f"\n{'='*60}")
        print(f"  {engine1.name} vs {engine2.name}")
        print(f"{'='*60}\n")

    while True:
        for player_idx in range(2):
            engine = engines[player_idx]
            opp_idx = 1 - player_idx
            rack = racks[player_idx]
            move_number += 1

            # Check game over conditions
            if final_turns_left is not None:
                if final_turns_left <= 0:
                    # Game over
                    return _game_result(engines, scores, move_times, watch, seed)
                final_turns_left -= 1

            # Build game_info
            game_info = {
                'your_score': scores[player_idx],
                'opp_score': scores[opp_idx],
                'tiles_in_bag': len(bag),
                'move_number': move_number,
                'blanks_on_board': list(blanks_on_board),
            }

            # Generate legal moves
            moves = get_legal_moves(board, rack, blanks_on_board)

            # Ask engine to pick a move (timed)
            t_pick = time.time()
            chosen = engine.pick_move(board, rack, moves, game_info)
            move_times[player_idx].append(time.time() - t_pick)

            if chosen is None or not moves:
                # Pass
                consecutive_passes += 1
                if watch:
                    print(f"  Turn {move_number} ({engine.name}): PASS")
                    print(f"  Score: {engines[0].name} {scores[0]} - {engines[1].name} {scores[1]} | Bag: {len(bag)}\n")

                # Notify opponent
                engines[opp_idx].notify_opponent_move(None, {
                    'your_score': scores[opp_idx],
                    'opp_score': scores[player_idx],
                    'tiles_in_bag': len(bag),
                    'move_number': move_number,
                    'blanks_on_board': list(blanks_on_board),
                })

                if consecutive_passes >= 4:
                    return _game_result(engines, scores, move_times, watch, seed)
                continue

            consecutive_passes = 0

            # Place the move on the board
            word = chosen['word']
            row = chosen['row']
            col = chosen['col']
            horizontal = chosen['direction'] == 'H'
            move_score = chosen['score']

            board.place_word(word, row, col, horizontal)
            scores[player_idx] += move_score

            # Track blanks placed on board
            blanks_used = chosen.get('blanks_used', [])
            for bi in blanks_used:
                if bi < 0:
                    bi = -bi
                if horizontal:
                    blanks_on_board.append((row, col + bi, word[bi]))
                else:
                    blanks_on_board.append((row + bi, col, word[bi]))

            # Remove used tiles from rack, draw new ones
            tiles_used = chosen.get('tiles_used', list(word))
            new_rack = list(rack)
            for t in tiles_used:
                if t in new_rack:
                    new_rack.remove(t)
                elif '?' in new_rack:
                    new_rack.remove('?')

            # Draw from bag
            drawn = draw_tiles(bag, ''.join(new_rack), RACK_SIZE)
            new_rack.extend(drawn)
            racks[player_idx] = ''.join(new_rack)

            # Check if bag just emptied
            if len(bag) == 0 and final_turns_left is None:
                # Both players get one more turn each
                final_turns_left = 2

            if watch:
                pos = f"R{row}C{col} {'H' if horizontal else 'V'}"
                print(f"  Turn {move_number} ({engine.name}): {word} at {pos} for {move_score} pts")
                print(board.display())
                print(f"  Score: {engines[0].name} {scores[0]} - {engines[1].name} {scores[1]} | Bag: {len(bag)}\n")

            # Notify opponent
            engines[opp_idx].notify_opponent_move(chosen, {
                'your_score': scores[opp_idx],
                'opp_score': scores[player_idx],
                'tiles_in_bag': len(bag),
                'move_number': move_number,
                'blanks_on_board': list(blanks_on_board),
            })

    return _game_result(engines, scores, move_times, watch, seed)


def _game_result(engines, scores, move_times, watch, seed=None):
    """Build and optionally display game result."""
    if scores[0] > scores[1]:
        result = 'win'
    elif scores[1] > scores[0]:
        result = 'loss'
    else:
        result = 'tie'

    # Notify engines
    for i, eng in enumerate(engines):
        opp = 1 - i
        gi = {'your_score': scores[i], 'opp_score': scores[opp],
              'tiles_in_bag': 0, 'move_number': 0, 'blanks_on_board': []}
        r = 'win' if scores[i] > scores[opp] else ('loss' if scores[i] < scores[opp] else 'tie')
        eng.game_over(r, gi)

    if watch:
        print(f"\n{'='*60}")
        print(f"  FINAL: {engines[0].name} {scores[0]} - {engines[1].name} {scores[1]}")
        spread = scores[0] - scores[1]
        winner = engines[0].name if spread > 0 else (engines[1].name if spread < 0 else "TIE")
        if spread != 0:
            print(f"  Winner: {winner} by {abs(spread)}")
        else:
            print(f"  Result: TIE")
        print(f"{'='*60}\n")

    return {
        'engine1': engines[0].name,
        'engine2': engines[1].name,
        'score1': scores[0],
        'score2': scores[1],
        'spread': scores[0] - scores[1],
        'winner': engines[0].name if scores[0] > scores[1] else (engines[1].name if scores[1] > scores[0] else 'tie'),
        'move_times_1': move_times[0],
        'move_times_2': move_times[1],
        'seed': seed,
    }


# =========================================================================
# Engine loading
# =========================================================================

def load_engine(name):
    """Load an engine class by module name.

    Looks in bots/ and examples/ directories.
    """
    # Try bots/ first
    for prefix in ['bots', 'examples']:
        module_name = f"{prefix}.{name}"
        try:
            mod = importlib.import_module(module_name)
            # Find the BaseEngine subclass
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, BaseEngine)
                        and attr is not BaseEngine):
                    return attr()
            print(f"Warning: {module_name} has no BaseEngine subclass")
        except ImportError:
            continue

    print(f"Error: Could not find engine '{name}' in bots/ or examples/")
    sys.exit(1)


def find_all_bots():
    """Find all bot modules in bots/ directory."""
    bots_dir = os.path.join(_root, 'bots')
    bot_names = []
    for f in sorted(os.listdir(bots_dir)):
        if f.endswith('.py') and f != 'base_engine.py' and f != '__init__.py':
            bot_names.append(f[:-3])
    return bot_names


# =========================================================================
# Match modes
# =========================================================================

def run_match(engine1, engine2, num_games, watch=False, master_seed=None,
              game_seeds=None):
    """Run a match (series of games) between two engines.

    game_seeds: optional list of pre-assigned per-game seeds (one per game).
                When provided, overrides master_seed for seed generation.
    """
    if game_seeds is not None:
        # Pre-assigned seeds from tournament runner -- use directly
        assert len(game_seeds) == num_games, (
            f"Expected {num_games} seeds, got {len(game_seeds)}")
    else:
        # Generate per-game seeds from master seed (or random if none)
        if master_seed is not None:
            rng = random.Random(master_seed)
        else:
            rng = random.Random()
        game_seeds = [rng.randint(0, 2**31) for _ in range(num_games)]

    if watch:
        # Single game in watch mode
        print(f"\nBuilding GADDAG (first run takes ~48s, then cached)...")
        print(f"  Game seed: {game_seeds[0]}")
        result = play_game(engine1, engine2, watch=True, seed=game_seeds[0])
        return

    print(f"\n{engine1.name} vs {engine2.name} ({num_games} games)")
    if master_seed is not None:
        print(f"Master seed: {master_seed}")
    print(f"Building GADDAG (first run takes ~48s, then cached)...")

    wins1 = 0
    wins2 = 0
    ties = 0
    total_spread = 0
    total_score1 = 0
    total_score2 = 0
    # Per-engine timing aggregation (keyed by engine name)
    all_move_times = {engine1.name: [], engine2.name: []}
    t_start = time.time()

    for i in range(num_games):
        game_seed = game_seeds[i]
        # Alternate who goes first
        try:
            if i % 2 == 0:
                result = play_game(engine1, engine2, seed=game_seed)
            else:
                result = play_game(engine2, engine1, seed=game_seed)
                # Flip scores for consistent tracking
                result['score1'], result['score2'] = result['score2'], result['score1']
                result['spread'] = -result['spread']
        except Exception as e:
            first = engine1.name if i % 2 == 0 else engine2.name
            second = engine2.name if i % 2 == 0 else engine1.name
            print(f"\n  [CRASH] Game {i+1} (seed={game_seed}, "
                  f"{first} vs {second}): {type(e).__name__}: {e}")
            print(f"  Replay: python play_match.py {first.lower()} "
                  f"{second.lower()} --watch --seed {game_seed}")
            continue

        # Aggregate move times (engine order alternates each game)
        if i % 2 == 0:
            all_move_times[engine1.name].extend(result.get('move_times_1', []))
            all_move_times[engine2.name].extend(result.get('move_times_2', []))
        else:
            all_move_times[engine2.name].extend(result.get('move_times_1', []))
            all_move_times[engine1.name].extend(result.get('move_times_2', []))

        total_score1 += result['score1']
        total_score2 += result['score2']
        total_spread += result['spread']

        if result['score1'] > result['score2']:
            wins1 += 1
        elif result['score2'] > result['score1']:
            wins2 += 1
        else:
            ties += 1

        # Progress
        if (i + 1) % max(1, num_games // 10) == 0 or i + 1 == num_games:
            elapsed = time.time() - t_start
            gps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_games}] {wins1}-{wins2}"
                  f" ({ties} ties) spread: {total_spread//(i+1):+d}"
                  f" ({gps:.1f} games/s)")

    print(f"\n{'='*60}")
    print(f"  Results: {engine1.name} vs {engine2.name} ({num_games} games)")
    print(f"{'='*60}")
    print(f"  {engine1.name} wins: {wins1:>4}  |  {engine2.name} wins: {wins2:>4}  |  Ties: {ties}")
    print(f"  Avg spread:    {total_spread / num_games:>+.1f}")
    print(f"  {engine1.name} avg: {total_score1 / num_games:>6.1f}  |  {engine2.name} avg: {total_score2 / num_games:>6.1f}")
    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.1f}s ({num_games / elapsed:.1f} games/s)")
    print(f"{'='*60}")

    # Speed report
    tier = os.environ.get('BOT_TIER', None)
    tier_targets = {'blitz': 1.0, 'fast': 3.0, 'standard': 10.0, 'deep': 30.0}
    target = tier_targets.get(tier) if tier else None

    print(f"\n  Speed Report{f' (tier: {tier})' if tier else ''}:")
    for name in [engine1.name, engine2.name]:
        times = all_move_times.get(name, [])
        if not times:
            print(f"    {name}: no moves recorded")
            continue
        avg_t = sum(times) / len(times)
        max_t = max(times)
        p95_idx = int(len(times) * 0.95)
        sorted_t = sorted(times)
        p95_t = sorted_t[min(p95_idx, len(sorted_t) - 1)]
        status = ""
        if target:
            if avg_t <= target * 1.2:
                status = " [OK]"
            else:
                status = f" [OVER -- target ~{target:.0f}s]"
        print(f"    {name}: avg {avg_t:.2f}s, p95 {p95_t:.2f}s, max {max_t:.2f}s"
              f" ({len(times)} moves){status}")
    print()


def run_tournament(num_games, master_seed=None):
    """Round-robin tournament among all bots in bots/."""
    bot_names = find_all_bots()
    if len(bot_names) < 2:
        print("Need at least 2 bots in bots/ for a tournament")
        return

    print(f"\nTournament: {', '.join(bot_names)} ({num_games} games per matchup)")
    if master_seed is not None:
        print(f"Master seed: {master_seed}")
    print(f"Building GADDAG (first run takes ~48s, then cached)...\n")

    # Generate seeds for all matchup games
    if master_seed is not None:
        rng = random.Random(master_seed)
    else:
        rng = random.Random()

    results = {}
    all_move_times = {}  # per-bot timing aggregation
    for name in bot_names:
        results[name] = {'wins': 0, 'losses': 0, 'ties': 0, 'spread': 0, 'games': 0}
        all_move_times[name] = []

    for i in range(len(bot_names)):
        for j in range(i + 1, len(bot_names)):
            e1 = load_engine(bot_names[i])
            e2 = load_engine(bot_names[j])
            print(f"  {e1.name} vs {e2.name}...", end=' ', flush=True)
            matchup_seeds = [rng.randint(0, 2**31) for _ in range(num_games)]

            w1 = w2 = t = sp = 0
            for g in range(num_games):
                if g % 2 == 0:
                    r = play_game(e1, e2, seed=matchup_seeds[g])
                    all_move_times[e1.name].extend(r.get('move_times_1', []))
                    all_move_times[e2.name].extend(r.get('move_times_2', []))
                else:
                    r = play_game(e2, e1, seed=matchup_seeds[g])
                    all_move_times[e2.name].extend(r.get('move_times_1', []))
                    all_move_times[e1.name].extend(r.get('move_times_2', []))
                    r['score1'], r['score2'] = r['score2'], r['score1']
                    r['spread'] = -r['spread']

                sp += r['spread']
                if r['score1'] > r['score2']:
                    w1 += 1
                elif r['score2'] > r['score1']:
                    w2 += 1
                else:
                    t += 1

            print(f"{w1}-{w2} ({t} ties, spread {sp//num_games:+d})")

            results[bot_names[i]]['wins'] += w1
            results[bot_names[i]]['losses'] += w2
            results[bot_names[i]]['ties'] += t
            results[bot_names[i]]['spread'] += sp
            results[bot_names[i]]['games'] += num_games

            results[bot_names[j]]['wins'] += w2
            results[bot_names[j]]['losses'] += w1
            results[bot_names[j]]['ties'] += t
            results[bot_names[j]]['spread'] -= sp
            results[bot_names[j]]['games'] += num_games

    # Print standings
    standings = sorted(results.items(), key=lambda x: (-x[1]['wins'], -x[1]['spread']))
    print(f"\n{'='*60}")
    print(f"  TOURNAMENT STANDINGS")
    print(f"{'='*60}")
    print(f"  {'Bot':<20} {'W':>4} {'L':>4} {'T':>4} {'Spread':>8}")
    print(f"  {'-'*44}")
    for name, stats in standings:
        avg_sp = stats['spread'] / max(1, stats['games'])
        print(f"  {name:<20} {stats['wins']:>4} {stats['losses']:>4} {stats['ties']:>4} {avg_sp:>+7.1f}")
    print(f"{'='*60}")

    # Speed report
    tier = os.environ.get('BOT_TIER', None)
    tier_targets = {'blitz': 1.0, 'fast': 3.0, 'standard': 10.0, 'deep': 30.0}
    target = tier_targets.get(tier) if tier else None

    print(f"\n  Speed Report{f' (tier: {tier})' if tier else ''}:")
    for name in bot_names:
        times = all_move_times.get(name, [])
        if not times:
            print(f"    {name}: no moves recorded")
            continue
        avg_t = sum(times) / len(times)
        max_t = max(times)
        p95_idx = int(len(times) * 0.95)
        sorted_t = sorted(times)
        p95_t = sorted_t[min(p95_idx, len(sorted_t) - 1)]
        status = ""
        if target:
            if avg_t <= target * 1.2:
                status = " [OK]"
            else:
                status = f" [OVER -- target ~{target:.0f}s]"
        print(f"    {name}: avg {avg_t:.2f}s, p95 {p95_t:.2f}s, max {max_t:.2f}s"
              f" ({len(times)} moves){status}")
    print()


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Crossplay Tournament Match Runner')
    parser.add_argument('engine1', nargs='?', help='First engine (module name in bots/ or examples/)')
    parser.add_argument('engine2', nargs='?', help='Second engine')
    parser.add_argument('--games', type=int, default=10, help='Number of games (default: 10)')
    parser.add_argument('--watch', action='store_true', help='Watch a single game with board display')
    parser.add_argument('--tournament', action='store_true', help='Round-robin all bots in bots/')
    parser.add_argument('--tier', choices=['blitz', 'fast', 'standard', 'deep'],
                        default=None, help='Speed tier (sets BOT_TIER env var)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Master seed for reproducible games (per-game seeds derived from this)')
    parser.add_argument('--game-seeds', type=str, default=None,
                        help='Comma-separated per-game seeds (overrides --seed)')

    args = parser.parse_args()

    # Set BOT_TIER env var before loading engines
    if args.tier:
        os.environ['BOT_TIER'] = args.tier

    # Parse pre-assigned per-game seeds (from run_tourney.py)
    game_seeds_list = None
    if args.game_seeds:
        game_seeds_list = [int(s) for s in args.game_seeds.split(',')]

    if args.tournament:
        run_tournament(args.games, master_seed=args.seed)
    elif args.engine1 and args.engine2:
        e1 = load_engine(args.engine1)
        e2 = load_engine(args.engine2)
        run_match(e1, e2, args.games, watch=args.watch,
                  master_seed=args.seed, game_seeds=game_seeds_list)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python play_match.py random_bot my_bot              # 10 games")
        print("  python play_match.py random_bot my_bot --games 100  # 100 games")
        print("  python play_match.py random_bot my_bot --watch      # watch 1 game")
        print("  python play_match.py --tournament --games 50        # round-robin")


if __name__ == '__main__':
    main()
