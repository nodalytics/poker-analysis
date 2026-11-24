# A Blueprint Strategy for Heads-Up No-Limit Texas Hold'em: Abstraction, Regret Minimisation, Neural Self-Play and Measurement

> **Target: 9-max (decided 2026-11-09).** The product is a **9-max no-limit hold'em** blueprint: 100bb cash games
> and tournaments (with ICM). Everything in this folder so far is the **heads-up prototype**: it proves the methods
> (abstraction, CFR variants, neural self-play, re-solving, measurement) but plays only two-handed. What carries
> over, what is rebuilt and the build order are in [`../docs/2026-11-09-nine-max-plan.md`](../docs/2026-11-09-nine-max-plan.md).

*poker-analysis / research, November 2026. This document describes the heads-up prototype in this folder. Section
8 gives the measured results; `README.md` and `../docs/2026-11-0{7,8}-*.md` hold the logs and history.*

## Abstract

We compute an approximate Nash equilibrium ("blueprint") for heads-up no-limit Texas hold'em at 200 big-blind
stacks, the format of the public benchmark bot Slumbot. The game is made tractable by two abstractions:

- **actions:** a fixed menu of bet sizes;
- **cards:** suit isomorphism followed by 200 strength-percentile buckets, or 500 distribution-aware buckets per
  street.

The abstract game is solved by Monte Carlo counterfactual regret minimisation (MCCFR) in five variants:

- external sampling with linear weighting;
- discounted CFR with regret pruning;
- predictive CFR+;
- vector public-chance-sampling CFR;
- a neural policy trained by regularised self-play on a GPU.

Real-game play maps opponent bets into the abstraction by pseudo-harmonic translation, purifies low-probability
actions and enforces a minimum-defence floor.

We measure the strategy four ways:

- local best response (LBR) in the real game;
- a cross-validated best response inside the abstraction;
- luck-adjusted play against Slumbot;
- head-to-head play against engines, language models and MLP clones of human professionals.

The best blueprints reduce LBR's edge from about +140 to about +50–60 bb/100 (24-hour runs). They beat every engine and language model
we have, and the clones of the professionals by +43 to +64 bb/100. They are level with or slightly ahead of
Slumbot.

---

## 1. The game

Heads-up no-limit Texas hold'em (HUNL) is a two-player, zero-sum, extensive-form game with imperfect information
and chance.

- **Players and chips.** Two players $i \in \{1, 2\}$ each start a hand with $S = 20{,}000$ chips. Blinds are
  $50/100$, so the big blind is $\text{bb} = 100$ and $S = 200\,\text{bb}$.
- **Rounds.** There are four betting rounds (pre-flop, flop, turn, river). Chance deals two private cards to each
  player, then three, one and one public cards.
- **Histories.** Let $H$ be the set of histories (sequences of chance outcomes and actions), $Z \subset H$ the
  terminal histories, and $P(h) \in \{1, 2, c\}$ the player to act at $h$, where $c$ is chance.
- **Information sets.** An information set $I \in \mathcal{I}_i$ groups the histories player $i$ cannot tell apart:
  the same public actions and board, any opponent cards.
- **Utilities.** $u_1(z) = -u_2(z)$. With $c_i(z)$ the chips player $i$ has put in,
  $$u_i(z) = \begin{cases} +c_{-i}(z) & \text{if } -i \text{ folded} \\ -c_i(z) & \text{if } i \text{ folded} \\ \min(c_1, c_2)\cdot \operatorname{sgn}(r_{-i} - r_i) & \text{at showdown,} \end{cases}$$
  where $r_i$ is the rank of player $i$'s best five-card hand (lower is stronger; PokerHandEvaluator).
- **Strategies.** A behavioural strategy $\sigma_i$ gives a distribution $\sigma_i(I, \cdot)$ over the actions
  $A(I)$ at each $I \in \mathcal{I}_i$. The reach probability $\pi^\sigma(h)$ factors into each player's part and
  chance's, $\pi^\sigma(h) = \pi^\sigma_1(h)\,\pi^\sigma_2(h)\,\pi_c(h)$. We write $\pi^\sigma_{-i}$ for the
  opponent-and-chance part.
- **Best response and exploitability.** A profile $\sigma^*$ is a Nash equilibrium if no player gains by deviating.
  The exploitability of a profile is
  $$\varepsilon(\sigma) = \tfrac{1}{2}\Big(\max_{\sigma_1'} u_1(\sigma_1', \sigma_2) + \max_{\sigma_2'} u_2(\sigma_1, \sigma_2')\Big) \;\ge\; 0,$$
  with equality exactly at equilibrium. All measures below report bb per 100 hands (bb/100). With
  $\text{bb} = 100$ chips, this equals chips per hand.

HUNL has about $10^{161}$ decision points (`../docs/2026-11-07-game-size-and-solving.md`), so $\sigma^*$ cannot be
computed directly. We solve an abstraction instead.

## 2. Abstraction

An abstraction maps the real game $G$ to a smaller game $\hat G$ by merging information sets (card abstraction) and
removing actions (action abstraction). A strategy for $\hat G$ is played in $G$ through a translation (Section 5).

### 2.1 Action abstraction

The betting tree (`tree.hpp`) allows the following actions (sizes are total chips for that street):

| Situation | Actions |
|---|---|
| Pre-flop, first in | fold, call (limp), open to 2, 2.5 or 3 bb, all-in |
| Pre-flop, facing a limp | check, raise to 3 or 4 bb, all-in |
| Pre-flop, facing an open | fold, call, 3-bet to 3× or 4×, all-in |
| Pre-flop, facing a 3-bet | fold, call, 4-bet to 2.3×, all-in |
| Post-flop, first bet | check, bet $\tfrac13$, $\tfrac23$, 1 or 2 pots, all-in |
| Post-flop, facing a bet | fold, call, raise to 1 pot, raise 2.5×, all-in |

Post-flop sizes are fractions of the pot after calling. There are two rules:

- at most four raises pre-flop and three per street after the flop;
- a raise that would leave the raiser less than a third of their stack becomes all-in.

The tree has 277,406 nodes: 101,568 decision nodes, 87,919 folds and 87,919 showdowns.

### 2.2 Lossless card abstraction

Suits are interchangeable, so hands are first reduced to canonical classes with Waugh's hand-isomorphism indexer.
The rounds are indexed as hole + board groups $\{2\}, \{2,3\}, \{2,4\}, \{2,5\}$, giving

$$|\mathcal{C}_0| = 169,\quad |\mathcal{C}_1| = 1{,}286{,}792,\quad |\mathcal{C}_2| = 13{,}960{,}050,\quad |\mathcal{C}_3| = 123{,}156{,}254 .$$

### 2.3 Lossy card abstraction: percentile buckets (`abstraction.cpp`, `out/`)

For a river class $x$ (hole cards $h$, full board $b$), **hand strength** is the share of opponent holdings it beats,
ties counting half:

$$\mathrm{HS}(x) = \frac{1}{|O(x)|} \sum_{o \in O(x)} \Big( \mathbb{1}[r(h,b) < r(o,b)] + \tfrac12 \mathbb{1}[r(h,b) = r(o,b)] \Big),$$

where $O(x)$ is the set of 990 opponent holdings that do not share a card with $h$ or $b$.

On the turn and flop the future is averaged over the remaining cards $\omega$:

$$\mathrm{EHS}(x) = \mathbb{E}_\omega[\mathrm{HS}(x,\omega)], \qquad \mathrm{EHS}^2(x) = \mathbb{E}_\omega[\mathrm{HS}(x,\omega)^2] .$$

Using $\mathrm{EHS}^2$ rather than $\mathrm{EHS}$ rewards variance, so drawing hands sort above made hands of the same
mean. Each post-flop round is cut into $B$ equal-population buckets on these values ($\mathrm{EHS}^2$ on the flop
and turn, $\mathrm{HS}$ on the river): $\beta_t(x) = \lfloor B \cdot F_t(v(x)) \rfloor$, where $F_t$ is the
empirical distribution function. Pre-flop keeps all 169 classes. Run 2 uses $B = 200$, run 3 $B = 1{,}000$.

### 2.4 Distribution-aware buckets (`abstraction_pa.cpp`, `out_pa/`)

Two hands with the same $\mathrm{EHS}$ can have very different futures. Following Johanson et al. (2013), a hand is
described by the whole distribution of its next-round strength:

- **Flop and turn:** a 50-bin cumulative histogram of next-round strength over the next card. Write
  $\mathbf{F}(x) \in [0,1]^{50}$, with $F_k(x) = \Pr_\omega[\,s(x,\omega) \le k/50\,]$, where $s$ is the turn's
  $\mathrm{EHS}$ (for flop hands) or the river's $\mathrm{HS}$ (for turn hands).
- **River: opponent-cluster hand strength (OCHS).** The 169 starting classes are ranked by mean river strength and
  cut into $G = 8$ groups of equal combinations. The feature is
  $\mathrm{HS}_g(x)$, hand strength against group $g$ alone, for $g = 1..8$.

The features are clustered by $k$-means ($k = 500$, $k$-means++ start, 15 Lloyd iterations on a sample of
1.5 million), and every class is assigned to its nearest centre. In one dimension the earth mover's distance between
two distributions is the $L_1$ distance between their CDFs. We use squared $L_2$ between CDFs as a smooth proxy:

$$\mathrm{EMD}(p,q) = \int |F_p - F_q| \;\approx\; \sqrt{\textstyle\sum_k (F_{p,k} - F_{q,k})^2}\ \text{(up to scale)}.$$

## 3. Counterfactual regret minimisation

For an information set $I$ of player $i$ and an action $a$, the **counterfactual value** is

$$v_i^\sigma(I, a) = \sum_{h \in I} \sum_{z \sqsupseteq ha} \pi^\sigma_{-i}(h)\, \pi^\sigma(ha, z)\, u_i(z), \qquad v_i^\sigma(I) = \sum_a \sigma(I,a)\, v_i^\sigma(I,a).$$

CFR (Zinkevich et al. 2007) iterates $t = 1..T$ and accumulates the **regrets**

$$R^T(I,a) = \sum_{t=1}^T w_t \big( v_i^{\sigma^t}(I,a) - v_i^{\sigma^t}(I) \big).$$

It plays by **regret matching**:

$$\sigma^{T+1}(I,a) = \frac{R^T_+(I,a)}{\sum_b R^T_+(I,b)} \quad\text{(uniform if the denominator is } 0\text{)}, \qquad x_+ = \max(x, 0).$$

The **average strategy** is

$$\bar\sigma^T(I,a) = \frac{\sum_t w_t\, \pi_i^{\sigma^t}(I)\, \sigma^t(I,a)}{\sum_t w_t\, \pi_i^{\sigma^t}(I)} .$$

With $w_t = 1$, $\varepsilon(\bar\sigma^T) \le \Delta\, |\mathcal{I}| \sqrt{|A|}/\sqrt{T}$. Here $\Delta$ is the range of
utilities, so the average strategy converges to equilibrium. The last iterate need not converge (Section 9).

### 3.1 External-sampling MCCFR (`mccfr.cpp`, runs 2–4)

Each iteration samples one deal: both players' hole cards and the full board. It then traverses the tree for a
traverser $i$:

- **at $i$'s nodes,** every action is explored and $R(I,a) \mathrel{+}= \tilde v(I,a) - \tilde v(I)$;
- **at the opponent's nodes,** one action is sampled from $\sigma^t$ and its probabilities are added to
  $\bar\sigma$;
- **chance is sampled.**

The sampled values are unbiased estimates of the counterfactual values, scaled by $1/q(z)$, the probability of
sampling $z$ (Lanctot et al. 2009).

Implementation details:

- regret and strategy tables are shared, lock-free, across threads (as in Pluribus);
- each dealt hand is traversed for both players;
- every information set is the pair (betting node, bucket), stored at `offset(node) + bucket * |A(node)|`.

**Linear weighting.** Following Pluribus, every 10 minutes for the first 400, the tables are scaled by
$t/(t+1)$. This approximates $w_t \propto t$.

### 3.2 Discounted CFR (`--dcfr`, run 5)

Brown and Sandholm (2019) discount at the end of each period $t$ (60 s of sampling) with $\alpha = 1.5$, $\beta = 0$,
$\gamma = 2$:

$$R(I,a) \leftarrow \begin{cases} R(I,a)\,\dfrac{t^{\alpha}}{t^{\alpha}+1} & R(I,a) > 0 \\[1ex] R(I,a)\,\dfrac{t^\beta}{t^\beta+1} = \tfrac12 R(I,a) & \text{otherwise,} \end{cases} \qquad \bar S(I,a) \leftarrow \bar S(I,a) \Big(\frac{t}{t+1}\Big)^{\gamma},$$

where $\bar S$ is the unnormalised average-strategy sum. Early noisy iterations fade, and large negative regrets
recover quickly.

### 3.3 Regret-based pruning (run 5)

After a warm-up, at the traverser's pre-river nodes, an action with $R(I,a) < -C$ is skipped with probability 0.95
($C = 10^5$ chips). Regret matching gives such an action probability 0, so skipping it leaves
$\tilde v(I) = \sum_a \sigma(I,a)\, \tilde v(I,a)$ exact. Only that action's own regret goes stale.

### 3.4 Predictive CFR+ (`--pcfr`, run 6)

Regrets are floored at zero after every update (CFR+). Each information set also stores the last instantaneous
regret $m(I,a) = \tilde v(I,a) - \tilde v(I)$ as a prediction of the next one (Farina, Kroer and Sandholm 2021). The
strategy uses the regrets plus the prediction:

$$\sigma^{t+1}(I,a) \propto \big(R^t(I,a) + m^t(I,a)\big)_+, \qquad R^{t}(I,a) = \big(R^{t-1}(I,a) + \tilde v(I,a) - \tilde v(I)\big)_+ .$$

Optimism damps the cycling of regret dynamics (Section 9), and in published tests it converges several times faster
than CFR+ or DCFR.

### 3.5 Vector public-chance-sampling CFR (`mccfr_vec.cpp`, runvec)

Johanson et al. (2012) sample only the **public** chance outcome, the board $b$. The private cards stay as a vector.
With $\mathcal{H}_b$ the 1,081 holdings not touching $b$, a traversal carries for every holding $h$:

- the traverser's reach $\rho_t[h]$;
- the opponent's reach $\rho_o[h]$.

It returns the traverser's counterfactual values $\mathbf{v}[h]$, so one traversal updates about
$1{,}081^2 \approx 1.17 \times 10^6$ holding pairs.

**Terminal values with card removal.** Let $\mathcal{H}_b(h) = \{o \in \mathcal{H}_b : o \cap h = \emptyset\}$. At a
fold node, with $p$ the chips the folder paid,

$$\mathbf{v}[h] = \pm p \sum_{o \in \mathcal{H}_b(h)} \rho_o[o] = \pm p\Big( \Sigma - \Sigma_{c_1(h)} - \Sigma_{c_2(h)} + \rho_o[h] \Big)$$

by inclusion–exclusion. Here $\Sigma$ is the total opponent reach and $\Sigma_c$ the reach of holdings containing
card $c$; the last term adds back the one holding counted twice. At showdown, holdings are sorted by rank, and two
passes (weakest up, strongest down) keep running totals and per-card totals over tie groups. They give

$$\mathbf{v}[h] = \min(c_1,c_2)\Big( \sum_{o \in \mathcal{H}_b(h):\, r_o > r_h} \rho_o[o] \;-\; \sum_{o \in \mathcal{H}_b(h):\, r_o < r_h} \rho_o[o] \Big)$$

in $O(|\mathcal{H}_b|)$ per node after one $O(|\mathcal{H}_b| \log |\mathcal{H}_b|)$ sort per board.

A self-test checks both terminal formulas against $O(|\mathcal{H}_b|^2)$ brute force at every start: largest error
$2\times 10^{-4}$.

**Decision nodes.**

- **Opponent's nodes:** the opponent's reach is split by $\sigma(\beta(h), a)$ and the children's values are summed.
- **Traverser's nodes:** each holding's regret increment is $\mathbf{v}_a[h] - \mathbf{v}[h]$. All holdings in the
  same bucket share one information set, so the increments are summed per bucket and written once. Strategies are
  likewise computed once per distinct bucket on the board.
- **Zero reach:** a node whose opponent-reach vector is all zero returns 0. External sampling never visits such
  nodes either.

## 4. A neural policy by regularised self-play (`../scripts/mlp_selfplay.py`)

### 4.1 Model

A single network $f_\theta$ serves both seats. Its inputs are:

- node features $\phi(n)$: street, seat, pot, price, stack, raises this street, the last action;
- a learned embedding $e_{t}(\beta)$ of the holding's bucket on the current street, plus a scalar strength.

It has three layers of 256 GELU units and two heads: logits over an action vocabulary, masked to $A(n)$, and a
value head $V_\theta$. Games are played in $\hat G$ (the same tree and buckets as the table blueprints), 65,536 at a
time on a GPU. Cards come from a C++ dealer (`bp_deal` in `libbp3.so`).

### 4.2 Objective

For a decision with return $G$ (big blinds won by the acting player), advantage $\hat A = G - V_\theta(s)$
(normalised by a running mean absolute value) and deal weight $w$:

$$\mathcal{L}(\theta) = \frac{1}{\sum w}\sum w \Big[ -\hat A \log \pi_\theta(a\,|\,s) + \tfrac{1}{50}\big(V_\theta(s) - G\big)^2 + \alpha\, \mathrm{KL}\big(\pi_\theta(\cdot|s)\,\|\,\pi_{\bar\theta}(\cdot|s)\big) - \eta\, \mathcal{H}\big(\pi_\theta(\cdot|s)\big) \Big].$$

$\pi_{\bar\theta}$ is a **magnet**: an exponential moving average of the weights, $\bar\theta \leftarrow (1-\tau)\bar\theta + \tau\theta$.

- With $\tau > 0$ this is regularised Nash dynamics (R-NaD; Perolat et al. 2022) and magnetic mirror descent
  (Sokota et al. 2023). Its fixed points approach equilibrium, whereas unregularised self-play gradients cycle.
- With $\tau = 0$ and $\bar\theta$ a distilled blueprint, it is the reference-model penalty of RLHF.

The exported policy is a slower average of $\theta$.

### 4.3 Importance-weighted dealing

Let $T(x) \in \{\text{weak}, \text{medium}, \text{strong}\}$ be the tier of a final river strength:
$\mathrm{HS}$-percentile $< \tfrac13$, between $\tfrac13$ and $0.8$, or $\ge 0.8$. The deal's cell is
$k = (T(x_1), T(x_2))$, one of 9.

Half of each batch is dealt fairly, with cell probabilities $p_k$. The other half is spread uniformly over the
cells. The proposal is therefore $q_k = \tfrac12 p_k + \tfrac1{18}$, and each deal carries weight $w = p_k / q_k$.
Then

$$\mathbb{E}_q[w\, g] = \mathbb{E}_p[g]$$

for any per-deal gradient $g$, so the strategy learned is for fair dealing while rare cells (coolers, weak against
weak) are seen often. The fair shares are $p_{\text{weak,weak}} = 0.109$ and $p_{\text{strong,strong}} = 0.034$.

### 4.4 Distillation (MLP + blueprint)

Phase 1 copies a table $\bar\sigma$: the table plays itself, and the network minimises
$-\sum_a \bar\sigma(I,a) \log \pi_\theta(a\,|\,I)$ over the states reached. Phase 2 fine-tunes with Section 4.2,
$\bar\theta$ fixed at the copy.

### 4.5 Opponent adaptation (phase B, `--opponent`)

A share $p$ of each batch is played against a fixed opponent table $\sigma_{\text{opp}}$, and the rest is
self-play. The network learns only from its own seats.

This is a restricted Nash response (Johanson et al. 2008): the opponent is $\sigma_{\text{opp}}$ with probability
$p$ and an adversary otherwise. The KL anchor additionally keeps the result near the blueprint.

The opponent tables are clones of human professionals (Section 8.4), mapped into $\hat G$ by `../scripts/clone_table.py`.

**Result** ($p = 0.5$, $\alpha = 0.1$, 60 minutes of distillation from run 2, then 7 hours against the `pros_a`
clone; evaluated against `pros_b`, a clone of the *other* half of the hands; 2 million hands per reading):

| Opponent | Run 2's table | Adapted |
|---|---|---|
| `pros_a` (trained against) | +42.2 | +57 to +60 |
| `pros_b` (held out) | +40.2 [+38.9, +41.4] | +47.5 to +48.9 |
| Run 2 | — | −2.4 to −4.2 |

The gain arrived within 30 minutes and then held. Its LBR exceeds that of the same MLP copy trained without the
opponent by about 15 bb/100 (Section 8.1), which is within the noise.

## 5. Playing the real game (`../scripts/blueprint_bot.py`)

### 5.1 Translation

A real bet of pot fraction $x$ may fall between two abstract sizes $A < x < B$. It is mapped to $A$ with the
**pseudo-harmonic** probability of Ganzfried and Sandholm (2013), and to $B$ otherwise:

$$f_{A,B}(x) = \frac{(B - x)(1 + A)}{(B - A)(1 + x)} .$$

Sizes outside the menu map to the nearest end. The draw is fixed per betting history, so one hand keeps one reading.

### 5.2 Purification

Actions with $\bar\sigma(I,a) < 0.05$ are removed and the rest renormalised. Rare actions are where an average
strategy is least converged.

### 5.3 Minimum-defence floor

Facing a post-flop bet $B$ into a pot $P$ with $B \ge 0.9P$, a bettor whose bluffs never get called profits with
any two cards. The **minimum defence frequency** is

$$\mathrm{MDF} = \frac{P}{P + B} .$$

Our range at the node is the blueprint's own reach-weighted range, $\pi_i^{\bar\sigma}(h) \propto \prod
\bar\sigma(I_k(h), a_k)$ over our earlier actions. If our holding lies in the top MDF of that range by strength, its
fold probability moves to call. This cut LBR's edge from +135 to +104.

### 5.4 Re-solving (`resolve.cpp`)

Unsafe re-solving solves the turn or river with the blueprint's ranges, which a deviating opponent exploits. LBR
measured it at +996, so it stays off.

**Safe** re-solving (Brown and Sandholm 2017; the re-solve gadget of Burch, Johanson and Bowling 2014) solves the
river subgame from the start of the street, with exact cards, by vector CFR+. Let $r_u$ and $r_o$ be the players'
blueprint reach at the subgame root. For each opponent holding $h$, the opponent's counterfactual best-response
value against the blueprint is

$$\mathrm{CBV}(h) = \max_{\sigma_o} v_o^{(\bar\sigma_u, \sigma_o)}(h).$$

Before the subgame, the opponent chooses per holding to enter or to take $\mathrm{CBV}(h)$. The re-solve minimises

$$\sum_h \tilde r_o(h) \max\big(v_o(h), \mathrm{CBV}(h)\big), \qquad \tilde r_o(h) = \max\big(r_o(h),\ 0.02 \max_{h'} r_o(h')\big).$$

The floor matters: an opponent who deviates earlier can arrive with holdings the blueprint never brings here, and
with zero weight the re-solve would ignore them. Without it, the worst single-hand gain is +718 (+808 in the first
measurement; re-run 2026-11-23 on the same 200 spots); with it, +1.4.

**Measured on 200 river spots** (1,000 iterations, exact best responses in the subgame):

| Strategy | Opponent's best response vs blueprint | Worst single-hand gain over CBV |
|---|---|---|
| Unsafe re-solve | −573 ± 46 | +1,179 |
| **Safe re-solve** | **−161 ± 17** | **+1.4** |

The subgame must start **before** the opponent's choices on the street. The first live version re-solved from the
node after the opponent's bet, which trusts the blueprint's betting range; LBR measured it at +176 [+98, +254].

**Live, against LBR** (run 2, 3,000 hands):

| Version | LBR's edge |
|---|---|
| Blueprint, no floor | +135 |
| Blueprint with the minimum-defence floor (Section 5.3) | +104 |
| Safe re-solve from the river's start, no floor | +146 [+76, +215] |
| **Safe re-solve with the floor** | **+105 [+42, +168]** |

The re-solve halved LBR's river all-ins that were answered by folds (328 hands to 159), but LBR's largest wins
moved to pot-size bets on the flop and turn, which a river subgame cannot reach. Turn re-solving is the next step.

## 6. Hand-reading models (`../scripts/opp_mlp_*.py`, `../scripts/human_clone.py`)

A player's next action is modelled as a categorical over seven classes: fold, check/call, raise $\le 0.4$,
$\le 0.8$, $\le 1.2$ or $> 1.2$ pots, all-in. It is given 15 features:

- street;
- $\mathrm{EHS}$ against one random hand, and $\mathrm{EHS}^{n-1}$ against the $n-1$ players still in;
- price, pot-to-stack and stack-to-pot ratios;
- raises this street, facing a bet, players in, position.

The model is a 2-layer MLP (128 GELU units), fitted by cross-entropy with early stopping on held-out hands. The
clones of professionals are these models fitted to each player's decisions in the 10,000 Pluribus hands, where all
hole cards are known.

## 7. Measurement

### 7.1 Local best response (`../scripts/lbr.py`)

LBR (Lisý and Bowling 2017) sees the bot's exact range at each decision, from Bayes on the bot's strategy. At each
decision it takes the action maximising immediate expected value, assuming calls to the end. It plays the **real**
game, so it also finds abstraction and translation leaks. Its winnings are a lower bound on $\varepsilon$.

Both players' cards are known, so luck is removed exactly. Each hand is scored by the equity-weighted pot at every
all-in, with a control variate pre-flop.

### 7.2 Best response inside the abstraction (`br_vec.cpp`)

For a table $\bar\sigma$ in $\hat G$, the best responder decides at every (node, bucket), aggregating over boards it
cannot see. Within a street a later decision must be settled before an earlier one. So information sets are
ordered by **level**, $8\cdot\text{street} + (\text{own decisions so far on the street})$, and decided from the
deepest level to the shallowest. Pass $\ell$ samples boards $\{b_k\}$, plays $\bar\sigma$ for the opponent and the
already-chosen response at deeper levels, and accumulates

$$\hat Q_\ell(I,a) = \sum_k \sum_{h:\, \beta(h, b_k) = I} \mathbf{v}_a[h; b_k] .$$

A plain argmax over $\hat Q$ overfits the boards seen. With 40 boards it scored +227 bb/100 in-sample and −53 out
of sample. So the argmax is **cross-validated**:

- $a^* = \arg\max_a \hat Q^{\text{even}}(I,a)$ is taken from even boards;
- it is kept only if the odd boards agree that it beats the table there,
  $\hat Q^{\text{odd}}(I,a^*) > \sum_a \bar\sigma(I,a)\, \hat Q^{\text{odd}}(I,a)$;
- otherwise the responder follows $\bar\sigma$ at $I$.

The resulting strategy is evaluated against $\bar\sigma$ on fresh boards. Its value is an unbiased estimate of a
lower bound on the abstract exploitability, and it tightens as the number of boards grows.

The table's value against itself in each seat is reported as a control: the two values must sum to about zero.

### 7.3 Head-to-head play

- **Against Slumbot:** 5,000 hands, luck-adjusted by AIVAT-lite. That is a control variate on all-in equity and
  pre-flop strength, plus $P \cdot \Delta\mathrm{equity}$ per street, cross-fitted.
- **Against our engines, language models and clones:** 24-hand sessions with a bootstrap over sessions.
- **Inside $\hat G$:** duplicate pairs (the same cards with seats swapped), 2 million hands.

**A calibration point.** The blueprint against an identical copy read $+19.9\ [+0.7, +38.8]$ over 1,500 sessions. A
rerun with the seat order reversed read $+3.0\ [-16, +22]$. Edges under about 20 bb/100 from that sample size are
noise.

## 8. Results (2026-11-09)

### 8.1 LBR (real game, purification and floor on)

| Strategy | LBR's edge (bb/100) |
|---|---|
| Run 2: 200 buckets, external sampling, 21.5 billion iterations | +104 [+44, +164] (+141 without the floor) |
| Run 3: 1,000 percentile buckets | +173 [+89, +258] |
| Run 4: 500 distribution-aware buckets (24 h) | +92 ± 5 |
| **Run 5: + DCFR + pruning (24 h)** | **+53 ± 7** |
| Run 6: + predictive CFR+ (24 h; +32 ± 5 over its first 6 h) | +60 ± 8 |
| **runvec: vector CFR + DCFR (24 h, 8 threads)** | **+59 ± 5** |
| Safe river re-solving on run 2 (Section 5.4), with the floor | +105 [+42, +168] |
| Phase B: run 2's MLP copy adapted to a clone (Section 4.5) | +182 ± 7 |
| Run 2's MLP copy without the opponent (`runmlp3`) | +167 ± 10 |
| MLP from scratch, weak magnet ($\alpha = 0.05$, $\tau = 0.002$) | stalled near +600 |
| **MLP from scratch, strong magnet ($\alpha = 0.3$, $\tau = 3\times 10^{-4}$), 2 h** | **+104** |

A single LBR point is about ±65. The 24-hour rows are the mean of the last 12 half-hourly checkpoints ± their
standard error. Those checkpoints share LBR's deals, so the error understates the uncertainty. After six hours
no run improved measurably. Run 6's early lead reversed (6-hour means 32, 54, 71, 60), while the DCFR runs stayed
level or improved slowly.

### 8.2 Speed

| Trainer | Throughput |
|---|---|
| External sampling | 13–22 thousand iterations a second per thread |
| Vector CFR | 0.45–0.5 million holding pairs a second per core; 0.78 million with per-bucket updates (same load) |
| GPU self-play | 250–310 thousand hands a second on one GPU (RTX 4090 or RTX PRO 6000) |

### 8.3 Opponents

| Opponent (heads-up, 200bb) | Blueprint run 2 (bb/100) |
|---|---|
| Slumbot (luck-adjusted, 5,000 hands) | +30.9 [−10, +72] |
| Our solver bot | +26.3 [+6, +46] |
| Simulated humans (maniac … nit) | +27.5 to +178.7 |
| Language models (Big-Tiger-Gemma 27B; Qwen2.5-VL 32B) | +78.5; +97.5 (n.s.) |

### 8.4 Clones of human professionals

The clones predict 77–82% of a professional's actions on held-out hands. Results against the pooled clone,
heads-up at 200bb:

| Our bot | bb/100 |
|---|---|
| Blueprint run 2 | +63.7 [+40, +89] |
| Run 6; runvec (3 h) | +43.4; +42.9 |
| MLP from scratch | +33.9 |
| Solver bot | +29.0 (n.s.) |
| Monte Carlo equity bot | +2.5 (n.s.) |

Lower exploitability did not mean higher profit against the clone. Exploiting a weak opponent is what phase B
(Section 4.5) is for.

## 9. Learning dynamics

Regret dynamics in zero-sum games are Hamiltonian-like: the last iterate can cycle or behave chaotically (Bailey and
Piliouras 2018; Sato, Akiyama and Farmer 2002). Three of our results agree:

- CFR's guarantee is for the **average** strategy;
- optimism (PCFR+, run 6) and the magnet (Section 4.2) both damp the cycling;
- the weak-magnet neural run is the example: within 30 minutes it came within 25 bb/100 of the blueprint, then
  drifted to a more deterministic, more exploitable policy (entropy $0.57 \to 0.35$, LBR $+165 \to +600$). The
  strong magnet converged steadily.

## 10. Limitations

1. **Every exploitability figure is a lower bound.** LBR uses a restricted response, and `br_vec` uses sampled and
   cross-validated information-set values.
2. **The abstraction is fixed.** Bet sizes outside the menu are handled only by translation. Card buckets merge
   holdings that a real best response would separate; this gap is LBR minus `br_vec`.
3. **The human clones played 6-max at 100bb.** At a heads-up 200bb table they extrapolate.
4. **Re-solving is not yet safe** (Section 5.4).
5. **Results against Slumbot** are within a ±40 bb/100 interval at 5,000 hands.

## 11. Reproducing

```
cd ~/research/blueprint && make -j            # pinned PokerHandEvaluator and hand-isomorphism into deps/
./abstraction out 200 && ./abstraction_pa out out_pa 500
./mccfr4 out_pa run6 --threads 16 --minutes 1440 --pcfr
./mccfr_vec out_pa runvec --threads 8 --minutes 1440 --dcfr
./mccfr_vec out runcyc --threads 8 --minutes 1440 --dcfr --flop-cycle --batch-boards 50   # Slumbot NL's sampling
BLUEPRINT_TREE=wide ./mccfr_vec out runwide --threads 8 --minutes 1440 --dcfr                # + 0.25x and 4x pot bets
./br_vec out run2 --boards 2000 --eval-boards 1000 --purify 0.05
./resolve_turn out run2 --test 100 --iters 200 --threads 6   # safe turn re-solve, every river card exact
cd .. && .venv-gpu/bin/python scripts/mlp_selfplay.py blueprint/runmlp2 --kl 0.3 --magnet-tau 0.0003 --entropy 0.01
sh blueprint/lbr_trend.sh blueprint/run6 3000 16
```

## References

- Bailey, J. and Piliouras, G. (2018). Multiplicative weights update in zero-sum games. *EC*.
- Brown, N. and Sandholm, T. (2017). Safe and nested subgame solving for imperfect-information games. *NeurIPS*.
- Brown, N. and Sandholm, T. (2018). Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. *Science*.
- Brown, N. and Sandholm, T. (2019). Solving imperfect-information games via discounted regret minimization. *AAAI*.
- Brown, N. and Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science* (Pluribus).
- Farina, G., Kroer, C. and Sandholm, T. (2021). Faster game solving via predictive Blackwell approachability. *AAAI*.
- Ganzfried, S. and Sandholm, T. (2013). Action translation in extensive-form games with large action spaces. *IJCAI*.
- Johanson, M., Zinkevich, M. and Bowling, M. (2008). Computing robust counter-strategies. *NeurIPS*.
- Johanson, M., Bard, N., Lanctot, M., Gibson, R. and Bowling, M. (2012). Efficient Nash equilibrium approximation through Monte Carlo counterfactual regret minimization. *AAMAS*.
- Johanson, M., Burch, N., Valenzano, R. and Bowling, M. (2013). Evaluating state-space abstractions in extensive-form games. *AAMAS*.
- Lanctot, M., Waugh, K., Zinkevich, M. and Bowling, M. (2009). Monte Carlo sampling for regret minimization in extensive games. *NeurIPS*.
- Lisý, V. and Bowling, M. (2017). Equilibrium approximation quality of current no-limit poker bots. *AAAI Workshop*.
- Perolat, J. et al. (2022). Mastering the game of Stratego with model-free multiagent reinforcement learning. *Science* (R-NaD).
- Sato, Y., Akiyama, E. and Farmer, J. D. (2002). Chaos in learning a simple two-person game. *PNAS*.
- Sokota, S. et al. (2023). A unified approach to reinforcement learning, quantal response equilibria, and two-player zero-sum games. *ICLR* (magnetic mirror descent).
- Waugh, K. (2013). A fast and optimal hand isomorphism algorithm. *AAAI Workshop*.
- Zinkevich, M., Johanson, M., Bowling, M. and Piccione, C. (2007). Regret minimization in games with incomplete information. *NeurIPS*.

Reference details are from memory; check the papers before citing fine points.
