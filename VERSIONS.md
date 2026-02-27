# Bot Versions

Each version is tagged in git. To test a version:
```
git checkout <tag>
python play_match.py my_bot defensive_bot --games 100
git checkout main
```

---

## v1-greedy
**Tag:** `v1-greedy`
**Strategy:** Pick highest-scoring move (greedy).
**Results:** Beats RandomBot ~100-0. Loses to smarter bots.

---

## v2-leave-eval
**Tag:** `v2-leave-eval`
**Strategy:** Greedy + leave evaluation (keep blanks=25, S=8, etc.).
**Results:** Beats GreedyBot ~55-45.

---

## v3-static-eval
**Tag:** `v3-static-eval`
**Strategy:** Leave eval + defensive penalties (-12 per 3W opened, -5 per 2W)
+ opponent modeling (scale defense when bag ≤ 7 and opponent likely has blanks/S).
**Results:** vs DefensiveBot: **50-50**, avg spread +10.9 (~3 min/100 games)

---

## v4-sim-20
**Tag:** `v4-sim-20`
**Strategy:** v3-static-eval + 1-ply Monte Carlo simulation (N_CANDIDATES=5, N_SAMPLES=20).
Opponent picks by score+leave, we record their raw score.
**Results:** vs DefensiveBot: **14-6 (70%)** over 20 games, avg spread +35 (~40 min/20 games)

---

## v5-greedy
**Tag:** `v5-greedy`
**Strategy:** Pure greedy — always take the highest-scoring move. No leave eval, no defense.
**Results:** vs DefensiveBot: **51-49**, avg spread +5.2 (~2.5 min/100 games)

### 5-strategy comparison (all vs DefensiveBot, 100 games each)

| Strategy | Wins | Losses | Win% | Avg Spread | Avg Score |
|----------|------|--------|------|------------|-----------|
| Greedy (pure score) | 51 | 49 | **51%** | **+5.2** | 425.7 |
| BingoFisher (protect blanks) | 48 | 52 | 48% | -15.2 | 434.2 |
| SpreadAdaptive (adjust by gap) | 42 | 58 | 42% | -5.9 | 431.9 |
| StageBased (phases by bag size) | 34 | 66 | 34% | -34.8 | 420.6 |
| LeaveFirst (4x leave weight) | 28 | 72 | 28% | -59.3 | 394.4 |

**Takeaway:** Greedy wins. Against a defensive opponent, maximizing score
each turn is optimal. Complexity (leave eval, stage logic, spread adaptation)
all hurt performance. BingoFisher scores highest on average (434.2) but
still loses more games — holding blanks is costly.

---

## v6-fast-sim (current)
**Tag:** `v6-fast-sim`
**Strategy:** Monte Carlo simulation with Quackle-calibrated leave values.
- Quackle single-tile leave values (O'Laughlin calibration): ?=25.57, S=8.04, Z=5.12, X=3.31, vowels negative (U=-5.10, O=-2.50, I=-2.07, A=-0.63)
- Vowel/consonant balance adjustments (+2 for balanced 1V+1C, -5 for pure vowels)
- Q-without-U penalty (-8 when no U in unseen)
- 1-ply Monte Carlo: top 5 candidates × 5 opponent samples → pick min avg_opp
- Static eval only when bag < 15
**Results:** vs DefensiveBot: **60-40**, avg spread +22.5 (~52 min/100 games)

### 5-strategy research comparison (all vs DefensiveBot, 100 games each)

| Strategy | Wins | Losses | Win% | Avg Spread | Time |
|----------|------|--------|------|------------|------|
| FastSim (Monte Carlo + Quackle leave) | **60** | 40 | **60%** | **+22.5** | 3099s |
| QuackleLeave (calibrated static eval) | 55 | 45 | 55% | +13.7 | 214s |
| TileEfficiency (turnover bonus) | 52 | 47 | 52% | +12.7 | 250s |
| MinVariance (minimize opp variance) | 51 | 48 | 51% | ±0.0 | 2828s |
| EndgameExpert (minimax when bag=0) | 42 | 56 | 42% | -18.7 | 345s |

**Takeaway:** Simulation wins. Quackle-calibrated leave values are a clear
upgrade over naive heuristics. Minimizing opponent variance (MinVariance) adds
no value at N_SAMPLES=5 — too noisy. Endgame minimax backfired due to
over-aggressive pre-endgame defense. Pure calibrated leave (QuackleLeave) is
the best fast option at 55% with normal speed.
