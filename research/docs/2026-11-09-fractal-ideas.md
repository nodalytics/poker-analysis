# Fractal ideas for the blueprint (2026-11-09)

The user asked to borrow ideas from fractal software (18 links, listed at the end). Most of the links are
renderers: Mandelbrot and Julia escape-time, Newton basins, IFS and L-systems. The rest are analysis tools:
box-counting dimension, multifractal spectra, and Hurst exponents of time series. This note maps each family to
the blueprint, tests the three ideas that could be tested in minutes on trained tables, and ranks the rest.

**Summary.** None of the three tests found a lever. Two of them confirm assumptions we had been relying on
without checking:

| Idea | Test | Result |
|---|---|---|
| Box counting along hand strength | Coarsen each node's strategy to *k* groups of adjacent buckets and measure the change | Strategies are **rough**: the error falls like k^−0.3 (run 2) and k^−0.1 (runs 5, vec), not like the 1/k of a smooth curve. The roughness is mostly **real**, not training noise |
| Mariani–Silver fill (store a flat region once) | Store each node at the coarsest *k* within 2% of its full strategy | Saves only **4–28%**: the 8-bit pack stays as it is |
| Hurst exponent (detrended fluctuation analysis) | Per-hand winnings in 14 Slumbot sessions (3,000–5,000 hands each) | Exponent **0.46–0.52**, mean 0.496, and the shuffled control gives 0.50. Hands are independent, so the bb/100 intervals are valid |

Script: `scripts/fractal_ideas.py`; results: `results/fractal/`.

## 1. Box counting along hand strength

A box-counting dimension measures how fast a curve's detail disappears as it is viewed at coarser scales. Here
the "curve" is a betting node's strategy as hand strength rises.

At each post-flop decision node, the buckets are sorted by mean strength. For the distribution-aware buckets,
strength is the mean of the bucket's centre. The strategy is then replaced by its reach-weighted mean over *k*
equal groups of adjacent buckets. The error E(k) is the reach-weighted total-variation distance to the original,
averaged over nodes by reach. A control shuffles the bucket order before grouping.

| Table | Street | E(1) | E(4) | E(16) | E(64) | E(256) | slope, k = 4–64 | shuffled E(4) |
|---|---|---|---|---|---|---|---|---|
| Run 2 (200 percentile buckets) | flop | 0.38 | 0.16 | 0.10 | 0.07 | 0 | −0.30 | 0.37 |
| | turn | 0.37 | 0.21 | 0.14 | 0.09 | 0 | −0.30 | 0.36 |
| | river | 0.38 | 0.29 | 0.20 | 0.13 | 0 | −0.28 | 0.37 |
| Run 5 (500 distribution-aware) | flop | 0.39 | 0.23 | 0.20 | 0.18 | 0.11 | −0.08 | 0.39 |
| | turn | 0.38 | 0.25 | 0.20 | 0.18 | 0.09 | −0.12 | 0.37 |
| | river | 0.33 | 0.24 | 0.18 | 0.13 | 0.08 | −0.22 | 0.32 |
| runvec (500 distribution-aware) | flop | 0.41 | 0.24 | 0.21 | 0.19 | 0.11 | −0.08 | 0.41 |
| | turn | 0.40 | 0.27 | 0.23 | 0.20 | 0.10 | −0.11 | 0.39 |
| | river | 0.34 | 0.25 | 0.18 | 0.14 | 0.08 | −0.22 | 0.33 |

(E(256) is 0 for run 2 because it has only 200 buckets.)

- **Strength explains part of the strategy.** Two groups (weak and strong) remove 26–42% of the variation on the
  flop and turn and 10–28% on the river, while the same split of a shuffled order removes 1%.
- **The rest is rough.** A smooth strategy curve would give a slope near −1. The distribution-aware buckets are
  rougher along strength (−0.1) because they were never meant to be one-dimensional. A drawing hand and a made
  hand with the same mean strength sit in different buckets and play differently, which is why those buckets
  beat percentile ones.
- **Is the roughness noise? Mostly not.** Run 5 and runvec trained the same game independently. Bucket by
  bucket, they disagree by a reach-weighted total-variation distance of 0.07–0.08 post-flop (0.13 pre-flop;
  `results/fractal/twin.py`). Merging only *pairs* of neighbouring buckets (E(256)) moves the strategy by
  0.08–0.11. Neighbouring buckets therefore differ by more than two whole training runs differ from each other.
  Smoothing the table along strength would mostly erase real distinctions, so it was not tried against LBR.

## 2. Mariani–Silver: fill a flat region once

Fast Mandelbrot renderers trace a rectangle's border and fill it in one colour when the border is uniform. The
analogue is to store a node at the coarsest *k* whose strategy is within 2% (total variation) of the full one,
with its group edges.

| Table | flop | turn | river |
|---|---|---|---|
| Run 2 | 72% of the storage | 84% | 96% |
| Run 5 | 96% | 97% | 99% |
| runvec | 91% | 93% | 98% |

The saving is small, and smallest exactly where the pack is largest (the river and the 500-bucket tables). The
advisor pack keeps the 8-bit table.

## 3. Hurst exponent of the results

Every bb/100 interval in this project assumes that hands are independent. If results came in streaks (a
persistent series, Hurst exponent above 0.5), the intervals would be too narrow. That could happen through a
bot's state, through Slumbot's, or through the session machinery. We had never checked it.

The test is detrended fluctuation analysis (order 1, window sizes 8 to 256 hands) on each worker's per-hand
winnings in play order. Each session is compared with 20 shuffles of its own hands. A second check is the
variance ratio: the variance of the sum of *m* consecutive hands, divided by *m* times the variance of one hand.

| | Exponent | Shuffled | Variance ratio, m = 50 |
|---|---|---|---|
| 14 sessions (blueprints, solver bot, adaptive bot, Monte, MLP) | 0.462 to 0.521, mean 0.496 | 0.488 to 0.510 | 0.65 to 1.30, mean 0.94 |

There is no persistence. The intervals stand as computed. The variance ratios at m = 200 range from 0.47 to 2.4,
but only 15–25 blocks enter each one, so they are noise.

## 4. The other ideas, ranked

| Fractal idea (where it came from) | Blueprint analogue | Verdict |
|---|---|---|
| **Fractal trainability boundary** (escape-time sets; Sohl-Dickstein 2024 showed the boundary between hyperparameters that train and those that diverge is fractal) | Our MLP self-play converged with a strong magnet and stalled or cycled with a weak one | **Worth a run.** A grid of learning rate × KL strength, 3 minutes each on the idle RTX 4090 (about 2 GPU-hours), would map where self-play converges. It would show whether the edge we picked by hand is a cliff to keep well away from. Not run yet |
| **Coarse-to-fine zoom** (deep-zoom renderers refine only where detail appears) | Train a coarse abstraction, then split buckets only where strategies differ and warm-start the finer table from the coarse one (Brown and Sandholm 2015, simultaneous abstraction and equilibrium finding) | **Plausible, larger.** Section 1 says *where* to split: the river and the distribution-aware flop are the roughest. Needs a regret-transfer step in `mccfr_vec`. A candidate after run 7 |
| **Self-similarity** (a fractal contains copies of itself) | Each street's subgame looks like the whole game, so it can be solved by the same solver with values at its leaves (DeepStack; Brown et al. 2018, depth-limited solving) | **Already on the plan** as turn re-solving, the next step after the river re-solve |
| Multifractal spectrum, lacunarity (imfractal) | Heavy-tailed per-hand results | **Nothing to find.** Section 3 found plain independent hands; luck adjustment already handles the tails |
| Newton basins of attraction (fractpy) | Which equilibrium a learning run lands in | Same question as the trainability boundary; covered there |
| IFS / chaos game, L-systems (4031651/fractals) | Sampling from a self-similar measure; bet-size trees from a grammar | **No use.** The betting tree is already generated by rules, and chance sampling is exact |
| Speed tricks (numba, CUDA, thread pools, Cython: fractPY, FractalPy, geomandel) | | Already in place: C++ with OpenMP, CUDA for the MLPs, MinGW for the advisor |

## Sources

Read through page summaries (READMEs), not source code.

| Link | What it is |
|---|---|
| github.com/wanglab-georgetown/fractal | Python: box counting (several variants), correlation sum, scale detection, for 2-D point sets |
| github.com/rbaravalle/imfractal | Python: multifractal spectra and the Sandbox method for images and volumes |
| pypi.org/project/fractal-analysis | Page did not load (two tries) |
| github.com/yujiading/fractals | Python: Hurst-exponent estimation, tests for fractional and multifractional Brownian motion, simulation |
| github.com/comp-comb/StereoFractAnalyzer | Python: box counting of 3-D surfaces and 2-D images |
| github.com/mariodosreis/fractal | R with a C core: Mandelbrot escape-time, logistic map |
| github.com/MarkSeufert/FractalRendererCpp | C++/OpenGL: Mandelbrot, Julia, Burning Ship viewer |
| github.com/4031651/fractals | TypeScript: IFS (Barnsley fern), L-systems |
| github.com/abdulfatir/fractal-library | Julia, Mandelbrot, IFS, Newton fractals |
| github.com/txtatech/frac-crawl | Text-to-colour-grid encoding; not fractal analysis |
| github.com/crapp/geomandel | C++11: Mandelbrot, Julia, Tricorn, Burning Ship, with a thread pool |
| github.com/justinlubin/fractastic | C: generalised Mandelbrot and Julia sets |
| practical-engineer.ai article | Python tutorial: Sierpinski, Koch, Mandelbrot, Julia, zoom videos |
| github.com/PatrickKalkman/python-fractals | Python: Julia sets and animation |
| medium.com/@bekelet article | HTTP 403, not read |
| github.com/wdeleon/fractPY | Julia sets on CPU threads and on CUDA, as a benchmark |
| github.com/asinghgaba/fractpy | Newton fractals for polynomials |
| github.com/Fergus-OH/FractalPy | Mandelbrot and Julia with numba and zoom animations |
