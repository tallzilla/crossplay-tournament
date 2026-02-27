"""Run DadBot vs MyBot tournament across all tiers."""
import subprocess
import sys
import datetime
import os
import random

PYTHON = sys.executable
SCRIPT = "play_match.py"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(WORK_DIR, "tourney_results.txt")

TIERS = [
    ("blitz", 20),
    ("fast", 20),
    ("standard", 20),
    ("deep", 5),
]


def main():
    # Generate a master seed, then derive one unique seed per game across
    # the entire tournament.  Every game gets a provably unique seed so
    # no two games (even in different tiers) can share starting positions.
    master_seed = random.randint(0, 2**31)
    rng = random.Random(master_seed)
    total_games = sum(g for _, g in TIERS)
    all_seeds = [rng.randint(0, 2**31) for _ in range(total_games)]

    # Partition seeds across tiers
    tier_seeds = {}
    offset = 0
    for tier, games in TIERS:
        tier_seeds[tier] = all_seeds[offset:offset + games]
        offset += games

    with open(OUTPUT, "w") as f:
        f.write(f"=== DADBOT vs MYBOT TOURNAMENT ===\n")
        f.write(f"Started: {datetime.datetime.now()}\n")
        f.write(f"Master seed: {master_seed}\n")
        f.write(f"Total games: {total_games}\n")
        for tier, _ in TIERS:
            seeds = tier_seeds[tier]
            f.write(f"  {tier}: seeds {seeds[0]}..{seeds[-1]} "
                    f"({len(seeds)} games)\n")
        f.write(f"\n")
        f.flush()

        for tier, games in TIERS:
            seeds_csv = ','.join(str(s) for s in tier_seeds[tier])

            f.write(f"{'='*50}\n")
            f.write(f"TIER: {tier} ({games} games)\n")
            f.write(f"Started: {datetime.datetime.now()}\n")
            f.write(f"{'='*50}\n")
            f.flush()

            result = subprocess.run(
                [PYTHON, SCRIPT, "dadbot", "my_bot",
                 "--games", str(games), "--tier", tier,
                 "--game-seeds", seeds_csv],
                cwd=WORK_DIR,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            f.write(f"\nCompleted: {datetime.datetime.now()}\n")
            f.write(f"Exit code: {result.returncode}\n\n")
            f.flush()

        f.write(f"\n=== TOURNAMENT COMPLETE ===\n")
        f.write(f"Finished: {datetime.datetime.now()}\n")


if __name__ == "__main__":
    main()
