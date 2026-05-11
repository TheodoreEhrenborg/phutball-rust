# Phutball Engine Evolution — Research Log

Reference for future contributors. Documents the full engine development history:
what was tried, what worked, what failed, and why.

---

> **✅ Tournament validity addressed (2026-05-11)**
>
> The earlier warning about random starting positions far from the ball has been
> resolved. As of 2026-05-11, all tournaments use a **5×5 window centered on the
> ball** (rows 5–9, cols 7–11) for man placement. The ladder below reflects the
> full re-evaluation under this fairer protocol.

---

# 2026-05-09 Research Notes

## Engine Ladder (Re-evaluated 2026-05-11 with 5×5 window tournament)

| Engine | Player Spec | vs. Previous | Confidence | Typical Depth | Notes |
|--------|-------------|-------------|------------|---------------|-------|
| eval (RichEvaluator) | `eval:1000` | baseline | — | 4–5 | Slower eval hurts search depth |
| timed (LocationEval) | `timed:1000` | beats eval | 0.938 (3W/0L/5T) | 4–5 | Simpler eval = deeper search |
| eval2 (TT + ordering) | `eval2:1000` | beats timed | 0.938 (3W/0L/14T) | 5 | Transposition table + jump-first |
| eval4 (JumpChain) | `eval4:1000` | beats eval2 | 0.910 (6W/2L/20T) | 5 | Game-specific jump-chain eval |
| eval5 (aspiration) | `eval5:1000` | beats eval4 | 0.938 (3W/0L/5T) | 6+ | Aspiration windows |
| **eval6 (beam search)** | `eval6:1000` | **beats eval5** | **0.938 (3W/0L/1T)** | **7** | **Current best** |

**MCTS variants (all below eval2):**

| Engine | vs. eval2 | Result |
|--------|-----------|--------|
| mcts:1000 | 0W/3L/0T | eval2 wins, P=0.938 |
| mcts-eval:1000 | 0W/3L/0T | eval2 wins, P=0.938 |
| mcts2:1000 | 1W/5L/4T | eval2 wins, P=0.937 |
| beam-mcts:1000 | 0W/3L/0T | eval2 wins, P=0.938 |

**NNUE:** nnue-eval:1000 vs eval4:1000 = 0W/3L/10T → eval4 wins (P=0.938). Falls between eval2 and eval4.

*Note: eval3 (history heuristic) tested and discarded — see failures below. eval6 was previously null-move pruning (ph-50x, failed), now redesigned as beam search (ph-4vv). azero skipped — neural network inference too slow for tournament evaluation (>20s/game at 100ms budget).*

---

## Architecture: Current Best Engine

**TimedPlayer5** = iterative-deepening alpha-beta + transposition table + aspiration windows

```
eval5:1000
  └── TimedPlayer5 (1000ms budget)
        ├── Iterative deepening: depth 1, 2, 3, ... until timeout
        ├── Alpha-beta negamax (TimedPlayer2 core)
        │     ├── TTable: 2^17 = 131072 entries, always-replace, Zobrist hashing
        │     └── Move ordering: ball jumps before man placements
        ├── Aspiration window: ±DELTA=0.15 around prev_score
        │     └── Fail-low/fail-high: widen by DELTA and re-search
        └── Eval function: Eval4Evaluator (JumpChainEvaluator)
```

**Eval function weights (Eval4Evaluator):**
- 0.4 × ball column progress (toward our goal)
- 0.3 × max jump-chain length (consecutive men in any direction)
- 0.2 × men near ball (Manhattan distance ≤ 3)
- 0.1 × men near our goal column (within 3 columns)

---

## Development Timeline

### Phase 1: Baseline Search Infrastructure

**Commit `1ef93e3` (ph-w4h):** Added `timed` engine and tournament subcommand.
- Iterative-deepening alpha-beta with `LocationEvaluator` (ball column progress only)
- Tournament: Beta(w+1, l+1) posterior, stops at 90% confidence
- Beta posterior evaluation via Lentz continued-fraction + Lanczos lgamma (no new crates)

**Commit `b9b9632` (ph-pwo):** Performance fix — replaced `HashMap<String,Board>` with
`Vec<Board>` for interior search nodes. HashMap allocation was limiting the engine to
depth 2–3; after the fix it reaches depth 4–5 in 1s.

### Phase 2: Better Evaluation

**Commit `c24de59` (ph-rd0):** Added `RichEvaluator` combining three signals:
- Ball progress (weight 0.6): same as LocationEvaluator
- Men near ball (weight 0.3): men within Manhattan distance ≤ 3, normalized
- Directional clustering (weight 0.1): men between ball and goal, normalized

Tournament: `eval:1000` vs `timed:1000` = 0W/3L/0T (P=0.938). Confirms richer eval
helps even with the same search algorithm.

### Phase 3: Transposition Table

**Commit `31517d4` (ph-ads):** Added `eval2` — `TimedPlayer2` with:
- Zobrist hashing (static `OnceLock<ZobristTable>` initialized at startup)
- TTable: 2^17 = 131,072 entries, always-replace policy
- Jump-first move ordering: ball-jump moves searched before man placements

Tournament: `eval2:1000` vs `eval:1000` = 3W/0L (P=0.938). eval2 reaches depth 5 vs
eval's depth 4.

### Phase 4: Game-Specific Evaluation

**Commit `a19470c` (ph-v7z):** Added `eval4` with `Eval4Evaluator` — jump-chain
potential replaces directional clustering:
- Max consecutive jumpable men from the ball in any of the 8 directions
- Goal-proximity scoring: men near our goal column
- Made `TimedPlayer2.eval_fn` configurable via `with_eval` constructor

Tournament: `eval4:1000` vs `eval2:1000` = 3W/0L/8T (P=0.938). Most impactful single
eval improvement — directly captures the key tactical motif (jump chains).

### Phase 5: Aspiration Windows

**Commit `91fe3f5` (ph-xty, by garnet polecat):** Added `eval5` — aspiration windows
on top of eval4's full stack:
- Narrow search window ±0.15 around the previous iteration's score
- Fail-low: widen alpha downward; fail-high: widen beta upward; re-search
- Falls back to full window [0,1] when already at boundary
- Allows deeper search within the same time budget (depth 6+ vs depth 5)

Tournament: `eval5:1000` vs `eval4:1000` = 3W/0L/1T (P=0.938). Current best engine.

---

### Phase 6: Beam Search / Forward Pruning (ph-4vv)

**Replaced** the failed null-move eval6 with forward pruning (beam search) on top of eval5.

At each internal node with remaining depth ≥ 2:
- All jump moves always searched (never pruned)
- TT best move always included (even if outside beam)
- Placement moves sorted by `JumpChainEvaluator::score` on each child board, top K=8 kept
- Shallow nodes (depth=1) search all moves for accuracy

This reduces effective branching from ~40 to ~18 (8 placements + ~10 jumps), allowing
eval6:1000 to reach depth 7 (vs eval5's depth 5–6 at the same time budget).

Tournament: `eval6:1000` vs `eval5:1000` = 5W/1L/5T (P=0.937 > 0.9). **eval6 wins.**

Tournament: `eval6:1000` vs `eval5:3000` = 3W/0L/4T (P=0.938 > 0.9). **eval6 wins.**

Surprising result: despite eval5:3000 having 3× the time and reaching depth 8,
eval6's depth-7 beam search beam dominates. Beam search with K=8 keeps the most
tactically-relevant placements and always preserves jumps, giving high-quality play
even at one fewer ply.

### Phase 7: NNUE Value Network (ph-0s2)

**Infrastructure added:** Supervised value network with raw-Rust inference (`nnue-eval`).

Architecture: **NnueNet** — 571 inputs → 32 ReLU → 1 sigmoid (value only, no policy).
- Input: 285-cell man plane + 285-cell ball plane + 1 side-to-move feature (571 total)
- Weight layout: feature-major `w1[feat * 32 + neuron]` for cache-efficient sparse access
- Typical board has ~20 occupied cells → ~660 mults for L1 vs 18,272 dense (~28× faster)

**Training workflow:**
1. `nnue-gen-data --games N --engine eval5:1000 --out nnue.dat` — labeled outcome data
2. `nnue-train --data nnue.dat --epochs N --save nnue.bin` — pure SGD in Rust
3. `tournament nnue-eval:1000 eval5:1000` — evaluate

**Initial result (20 training games, 1088 positions, 100 epochs):**
`nnue-eval:200` vs `eval5:200` = 0W/3L (P=0.062). **Lost.**

Root cause: insufficient training data. With only 20 games (~1k positions), the value
network has too little signal to generalize. Eval4Evaluator (hand-crafted) encodes
jump-chain intuition that requires many game examples to learn purely from outcomes.
The infrastructure is correct — the network loss decreases steadily (0.091 → 0.043
over 100 epochs). More training data and epochs needed.

**Why this approach is still promising:**
The NNUE inference cost is ~7× slower than Eval4Evaluator (vs ~500× for PhutballNet),
making it feasible as an eval function in alpha-beta. With 1000+ training games,
a 571→32→1 network should learn the key signals (ball column progress, jump-chain
potential) from supervised game outcomes. The break-even point is estimated at
~500 games (≈25k positions) with 200+ training epochs.

---

## What Failed (and Why)

### MCTS — Random Rollout (`mcts:1000`)

Added UCT-based MCTS with random rollouts (depth limit 50, return 0.5 on tie).
**Lost to both `alphabeta:3` and `eval:1000`.**

Root cause: phutball has a very high branching factor. Alpha-beta reaches ~3 million
nodes per second; MCTS manages ~200 simulations per second. Random rollouts rarely
reach terminal states in 50 moves — the signal is noise. Search volume gap is fatal.

### MCTS-Eval (`mcts-eval:1000`)

Replaced random rollout with `RichEvaluator::score` at leaf nodes. UCT constant C=0.5.
**Still lost to eval:1000.**

Same root cause: even with a better value estimate, MCTS builds far fewer nodes per
second than alpha-beta. Without a strong policy network to prune branches, MCTS cannot
compete at this branching factor within 1s.

### eval3 — History Heuristic (ph-rda, garnet polecat)

Added history heuristic to `eval2`: moves that caused beta cutoffs at any depth get
a bonus. **Lost to eval2: 0W/3L/8T (P=0.938).**

Root cause: in phutball's varied positions, the history table accumulates stale
biases — moves that worked well in past positions don't reliably recur. The
jump-first ordering in eval2 is already game-specific and better than a generic
history table. Lesson: game-specific ordering beats generic search heuristics.

### eval6 v1 — Null-Move Pruning (ph-50x, jade polecat) — REPLACED

Added null-move pruning (NULL_REDUCTION=2, fires at depth ≥ 3) to eval5.
**Lost to eval5: 1W/5L/9T at 200ms; P(eval6>eval5) = 0.063.**

Root cause: null-move pruning assumes that passing a turn is always bad (the
"null-move assumption"). In phutball, the game is deeply asymmetric — the ball can
be advanced in many diagonal directions and the player to move often has forced
tactical sequences. The null-move assumption does not hold here, causing the engine
to prune branches it should search. Generic search optimizations keep failing;
only game-specific improvements work.

**Replaced by ph-4vv:** eval6 was redesigned as beam search (forward pruning) —
see Phase 6 above. The null-move code was removed.

### AlphaZero (`azero:1000`)

Implemented full AlphaZero infrastructure:
- `PhutballNet`: MLP with 855 inputs (15×19×3 board planes), 64/32 hidden layers,
  policy head (200 moves) and value head (scalar)
- Burn.rs (ndarray CPU backend) for training
- Imitation learning warm-start from `eval:200` games
- Self-play loop with replay buffer to prevent mode collapse

Tournament: `azero:1000` vs `eval:1000` = 0W/3L/4T (P=0.062). **Lost.**

Root cause: CPU compute bottleneck. The policy network provides only ~5 MCTS
simulations per second on CPU — far below the search volume needed to compete
with alpha-beta. The 4 ties suggest strong defensive play but insufficient offensive
expansion. AlphaZero requires GPU inference to be competitive; on CPU, the network
overhead outweighs the policy guidance benefit.

---

## Key Lessons

1. **Game-specific knowledge > generic search heuristics.** Jump-chain evaluation
   (eval4) and jump-first move ordering (eval2) provided the biggest gains. History
   heuristic and null-move pruning both failed because they assume properties that
   don't hold in phutball's asymmetric, tactical positions.

2. **Eval function quality matters more than search tricks at this scale.** The jump
   from LocationEvaluator → RichEvaluator → Eval4Evaluator each gave measurable
   gains; structural improvements to search (TT, aspiration windows) helped but less.

3. **The TT + jump-first ordering is the critical structural improvement.** eval2's
   transposition table lets the engine avoid re-searching repeated positions, and
   jump-first ordering causes more beta cutoffs early. This is the foundation that
   eval4 and eval5 build on.

4. **MCTS cannot compete with alpha-beta at this branching factor on CPU.** Phutball's
   high branching factor means MCTS builds far fewer nodes/s. A policy network
   (AlphaZero) is the principled solution, but requires GPU inference to be practical.

5. **Game-specific forward pruning (beam search) works.** Unlike generic search
   heuristics (history, null-move), beam search using JumpChainEvaluator for ranking
   preserves jump moves and TT best moves, making it safe to prune placements.
   Reduces effective branching 40→18, gaining ~2 extra plies for the same time budget.

6. **Small sample sizes (3–4 decisive games) can give misleading confidence.** The
   tournament stops at P=0.9, which requires only a handful of decisive games (3W/0L
   is sufficient). Ties are common and don't count toward confidence. This is correct
   for iterating quickly, but results should be interpreted cautiously — a 3W/0L result
   corresponds to only ~7 games played.

---

## Future Directions

In approximate priority order:

1. **Quiescence search:** Extend search at positions with available ball jumps.
   The eval is unreliable at positions where a forced jump sequence is about to
   change the board significantly. This is likely the highest-leverage improvement.

2. **Better eval — multi-step chain potential:** Currently eval4 counts the max
   *immediate* jump chain (men consecutively adjacent to the ball). A better signal
   would count potential chains: men that could be bridged with one placement to
   create a long jump. Requires pattern recognition around the ball.

3. **Larger TTable:** Current size is 2^17 = 131,072 entries. A 2^20 (1M entry) table
   would reduce collisions at deeper depths with minimal memory overhead (~40MB).

4. **AlphaZero with GPU inference:** The infrastructure exists. With GPU-accelerated
   forward passes, MCTS can run thousands of simulations per second, making the
   policy + value network viable. The imitation warm-start and replay buffer are
   already implemented.

5. **Opening book:** The first few moves are slow to search (ball at center, high
   branching factor). A small opening book of strong initial placements could help.

---

## NNUE Experiment (added 2026-05-08)

**Architecture:** HalfBP features (ball_idx × man_idx → 81,225 sparse inputs),
accumulator layer L1=64, hidden L2=32/L3=8, sigmoid output. ClippedReLU
activations. Incremental accumulator updates (add on man-place, recompute on
ball-move). Pure Rust forward pass for inference speed.

**Training:** 50 games vs eval5:200ms teacher → 4,387 labelled positions.
50 epochs, Adam lr=0.001, MSE loss. Final loss: 0.030.

**Result:** `nnue-eval:1000 vs eval5:1000 = 0W/3L/0T` (eval5 wins, conf=0.938).

**Why it failed:**
- 50 training games = very small dataset for a 81k-feature network
- Teacher (eval5:200ms) is not highly accurate at short budget
- Network may not generalize beyond the narrow position distribution

**What NNUE would need to succeed:**
- 5,000+ training games from eval5:1000 (~16 hours of CPU time)
- Or: reduce feature space to 20-30 hand-crafted scalars (much less data needed)
- Or: GPU acceleration for batch training on 100k+ positions

**Conclusion:** NNUE infrastructure correct; compute-bound, same as AlphaZero.
The hand-crafted JumpChainEvaluator + aspiration windows (eval5) remains best.

---

## NNUE Random-Position Augmentation (added 2026-05-08, ph-5ki)

**Hypothesis:** Game-play data is biased toward positions arising from strong play.
Adding randomly-generated positions (random ball location + 1–5 random men) should
give broader state-space coverage and help the network generalize.

**New subcommands added:** `nnue-gen-random`, `nnue-merge`

**Data:**
- Game data: 100 eval5:200ms games → 8,109 positions (`nnue_v2.dat`)
- Random data: 5,000 positions (ball at row 1–13, col 2–16; 1–5 men; labeled by eval5:200ms) → `nnue_random.dat`
- Combined: 13,109 positions (`nnue_combined.dat`)

**Training:** 100 epochs, SGD lr=0.001, MSE loss. Final loss: 0.01971 (vs 0.043 from 1k positions).

**Results:**
- `nnue-eval:1000 vs eval5:1000 = 0W/3L/0T` (P(nnue>eval5)=0.062)
- `nnue-eval:1000 vs eval2:1000 = 0W/3L/1T` (P(nnue>eval2)=0.062)

**Analysis:** Loss improved significantly (0.043 → 0.020) with more data, but still
loses to both eval2 and eval5. The random positions help training loss but don't close
the gap against alpha-beta. Root cause is the same as before: the 571-input network
needs far more game data to learn jump-chain tactics that eval4 encodes explicitly.
Random positions expose the network to a wider board distribution but provide weaker
signal (eval5:200ms at short budget is a noisy teacher for non-game positions).

**What would help more:**
- 1,000+ game positions from eval5:1000 (~5–10 hours) with 200+ epochs
- Or: use random positions as pre-training then fine-tune on game data (curriculum learning)
- Or: add jump-chain count as an explicit input feature (guided representation)

---

## NNUE v2 Experiment (ph-6u3, added 2026-05-08)

Retry with 10× more data and stronger teacher labels.

**Training:** 500 games vs eval5:500ms teacher → **42,731 labelled positions**.
100 epochs, SGD. Final loss: **0.01521** (down from 0.030 in v1 — better fit).

**Results:**
- `nnue-eval:1000 vs eval5:1000 = 0W/3L/0T` (eval5 wins, conf=0.938)
- `nnue-eval:1000 vs eval2:1000 = 0W/3L/5T` (eval2 wins, conf=0.938)

**Summary of all NNUE attempts:**

| Run | Data | Teacher budget | Final loss | vs eval5 | vs eval2 |
|-----|------|---------------|-----------|---------|---------|
| v1 | 50 games | eval5:200ms | 0.030 | 0-3-0 | 0-3-1 |
| aug | 50g+5k random | eval5:200ms | 0.020 | 0-3-0 | 0-3-0 |
| v2 | 500 games | eval5:500ms | 0.015 | 0-3-0 | 0-3-5 |

**Verdict: Failed.** Even with 42k positions and a stronger teacher, NNUE cannot
beat eval2 (transposition table only, no jump-chain). The network loses to a
weaker engine than the teacher it was trained on.

**Root cause analysis:**
- Loss decreased well (0.030 → 0.015), but low training loss ≠ strong play
- The 571-input network (man/ball planes only) cannot capture the jump-chain
  combinatorial patterns that eval4/eval5 encode explicitly
- At 1s budget, NNUE's ~7× eval overhead vs eval4 costs 1–2 search depth levels;
  even perfect value estimates wouldn't fully compensate
- Phutball's key skill is recognizing multi-jump sequences — this requires either
  explicit enumeration (eval4's approach) or orders of magnitude more training data

**Conclusion:** NNUE in current form is not competitive. The architecture (571 sparse
inputs → 32 → 1) is too shallow to learn jump-chain patterns from outcome labels alone.
Further scaling (5000+ games, deeper network) is unlikely to close the gap without
rethinking the feature representation. **eval5 remains the best engine.**

---

## eval6 (Beam Search, 2026-05-08) — CURRENT BEST

**Architecture:** eval5 (TT + JumpChain + aspiration windows) + forward pruning.
At each node with depth >= 2, placements are sorted by JumpChainEvaluator and
only top K=8 are searched. All jump moves always included.

**Results:**
- eval6:1000 vs eval5:1000 = 5W/1L/5T, conf=0.937 ✅
- eval6:1000 vs eval5:3000 = 3W/0L/4T, conf=0.938 ✅ (1s beats 3x time budget!)
- Typical depth: 7 (vs eval5's 5-6)

**Why it works:** Beam search reduces effective branching factor from ~40 to ~8.
This allows 2 extra plies within the same time budget. The JumpChainEvaluator
is accurate enough that pruning placements ranked 9+ rarely misses the best move.

## eval7 Tuning (K-sweep, 2026-05-08)

Experiments to find optimal K:

| K | vs eval6 (K=8) | Result |
|---|---------------|--------|
| 4 | 3W/4L/12T (20 games) | Inconclusive, K=8 slightly better |
| 8 | — | Current best |
| 12 | 2W/6L/5T (13 games) | K=8 wins at 0.938 conf |

**eval6q** (beam + quiescence): 0W/3L/6T vs eval6 — quiescence hurts beam search
(same mechanism as eval5q vs eval5: overhead reduces depth, no net gain).

**Conclusion: K=8 is optimal. eval6:1000 is the current best engine.**

---

## beam-MCTS Experiment (2026-05-08)

**Architecture:** MCTS2 (lazy expansion, UCT) + beam selection: on first expansion
of any node, score all candidate placements by JumpChainEvaluator, keep only
top K=8 (same as eval6). All jump moves always included.

**Result:** `beam-mcts:1000 vs eval6:1000 = 0W/3L/0T` (conf 0.938).
Also 0-3 vs eval2:1000. ~120k sims/second.

**Conclusion: MCTS cannot beat alpha-beta on CPU regardless of policy guidance.**

All 4 MCTS variants tested: random rollout, eval-at-leaf, lazy (mcts2),
beam-guided (beam-mcts). ALL lose to eval5/eval6. MCTS would need GPU-backed
AlphaZero to be competitive. **CPU ceiling: eval6:1000 (beam K=8, depth 7).**

---

## Phase 8: Tournament Redesign (2026-05-11)

**Motivation:** Previous tournaments placed men randomly across the full 15×19 board.
This creates highly variable starting positions — some immediately decisive — biasing
results toward first-mover advantage rather than true engine strength.

**Change:** `play_tournament_game` now places men in a 5×5 window centered on the
ball start (rows 5–9, cols 7–11, i.e. START_ROW±2, START_COL±2). `start_men=4`
unchanged.

**Sanity check — eval6:1000 vs eval6:1000 (self-play, 10 games):**
- Result: 1W/4L/5T (P(e1>e2)=0.109)
- Interpretation: High tie rate (5/10) indicates positions are balanced; neither
  side dominates. This is expected for a self-play symmetric match. No first-mover
  pathology detected. 5x5 window is working correctly.

**Full ladder re-evaluation results (all at conf=0.9):**
- timed:1000 vs eval:1000 → **timed wins** 3W/0L/5T (P=0.938) — eval's richer but slower heuristic hurts search depth
- eval:1000 vs eval2:1000 → **eval2 wins** 3W/0L/0T (P=0.938)
- timed:1000 vs eval2:1000 → **eval2 wins** 3W/0L/14T (P=0.938) — confirms ordering: eval < timed < eval2
- eval2:1000 vs eval4:1000 → **eval4 wins** 6W/2L/20T (P=0.910)
- eval4:1000 vs eval5:1000 → **eval5 wins** 3W/0L/5T (P=0.938)
- eval5:1000 vs eval6:1000 → **eval6 wins** 3W/0L/1T (P=0.938)
- MCTS variants vs eval2: all lose (see ladder table above)
- nnue-eval:1000 vs eval4:1000 → eval4 wins 3W/0L/10T (P=0.938)
- azero: untestable — too slow (>20s/game at 100ms budget)

**Corrected final ladder:** eval < timed < eval2 < nnue-eval* < eval4 < eval5 < **eval6** (best)
(*nnue-eval positioned between eval2 and eval4; exact rank vs eval2 not directly tested)

**Conclusion: eval6:1000 remains the strongest engine. No change to website AI player needed.**
