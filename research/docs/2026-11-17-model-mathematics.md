# The models in mathematics, and their versions (2026-11-17, updated 2026-11-27)

What every 9-max network computes, how it is trained, how it is measured and how Kestrel searches with it, written
as formulas and checked against the code (`scripts/ninemax_selfplay.py`, `ninemax_engine.py`, `icm.py`,
`ninemax_eval.py`, `ninemax_ladder.py`, `ninemax_search.py`; Kestrel's `core/src/net.cpp`,
`core/src/ninemax_search.cpp`, `connectors/src/profile.cpp`). The version table at the end gives each run's settings
from its `config.json`.

Notation: $P = 9$ seats, $A$ menu slots ($A = S + 3$ with $S$ raise sizes; $A = 6$ for `mtt`), $BB = 100$ chips, $STACK = 100\,BB$. Seat $i$ is the player to act,
$\mathbb{1}[\cdot]$ an indicator, $\lVert\cdot\rVert$ the Euclidean norm.

## 1. The game and the menu

No-limit hold'em at a table of 2 to 9 players (`--seats` 6 for 6-max), seats 0 and 1 the blinds and the button
last; on a short table the early seats are empty. Every decision chooses one of $A = 6$ slots:

$$a \in \{\text{fold},\ \text{check/call},\ \text{size}_1,\ \text{size}_2,\ \text{size}_3,\ \text{all-in}\}$$

The sizes come from the menu (`ninemax_engine.MENUS`): `mtt` opens to 2, 2.5 or 3 bb before the flop and re-raises
to 3x (then 2.3x) the current bet; after the flop it bets 0.33, 0.75 or 1.5 pot and raises 1 pot (`medium` opens
2.5 bb only). A slot is legal when the rules allow it ($\ell \in \{0,1\}^A$); `--no-open-limp` makes calling the big
blind first in illegal.

**Wider menus (bet sizing, 2026-11-27).** `wide` ($S = 5$, $A = 8$): opens 2, 2.5 or 3.5 bb; re-raises 2.5x or 3.5x,
then 2.2x or 3x; heads-up bets 0.25, 0.5, 0.75, 1.25 or 2 pot and raises 0.75 or 1.5 pot; multiway bets 0.33 or
0.75 pot, raises 1 pot. `grid16` ($S = 16$): 16 sizes log-spaced over each range, $x_k = x_{lo}(x_{hi}/x_{lo})^{k/15}$
(opens 2-4 bb, re-raises 2-4x, bets and raises 0.2-3 pot). The random menu `size8` ($S = 8$) draws its sizes afresh
at every decision, $u_1 \le \dots \le u_8$ sorted uniforms and $x_k = x_{lo}(x_{hi}/x_{lo})^{u_k}$ over the same ranges,
once per decision so the legal moves, the inputs and the step see the same sizes. A post-flop size $x$ is a raise to
$\beta + x\,(p + \beta - c_i)$; a pre-flop open is $x$ big blinds and a re-raise $x\beta$.

**Per-seat menus.** The table keeps only chips and betting; the legal moves, the raise targets and the step depend
on the actor's menu alone. So networks trained on different menus share a table, each playing its own (nm48 on
`wide` against nm25 on `mtt` in every check, and in Kestrel's practice tables, `Table::set_menu`); rule players play
the table's menu, or `mtt` when a run trains on a random menu.

## 2. The cards: buckets and strength

A player's private information at street $t \in \{0,1,2,3\}$ is a bucket $b_t$: before the flop one of the 169
starting-hand classes, after it one of $B_t$ buckets (200 each since nm21; nm22-23 tried 100/200/500 and
1000/500/200), equal-population cuts of hand strength (EHS² on the flop and turn, HS on the river). A scalar
strength enters as well:

$$s = \begin{cases} \mathrm{pre}(b_0) & t = 0 \\ b_t / (B_t - 1) & t \ge 1 \end{cases}$$

## 3. The inputs $\phi(x)$

The public state $x$ is described by a vector $\phi(x) \in \mathbb{R}^{F}$ (`features`), all relative to seat $i$,
with the other seats in table order after it ($r_1, \dots, r_{P-1}$). With $c_j$ chips committed, pot
$p = \sum_j c_j + \text{dead}$, current bet $\beta$, chips $\text{avail}_j$ a player can put in, $\text{left} =
\text{avail}_i - c_i$:

| Block | Entries | Count |
|---|---|---|
| seat, street | one-hot $i$, one-hot $t$ | 9 + 4 |
| pot and price | $p/(P\cdot STACK)$, $\frac{\beta - c_i}{p + \beta - c_i}$, $c_i/STACK$, $\text{left}/STACK$, $\min(\text{left}/p, 20)/20$, raises/4, pre-flop raises/4, live/P, all-in/P, still to act/P | 10 |
| the others | for each $r_k$: live, all-in, $c_{r_k}/STACK$, acted, is the aggressor | 5 × 8 |
| legal moves and sizes | $\ell$; for each size slot $\min\!\big(\tfrac{\text{target}-\beta}{p}, 10\big)/10 \cdot \ell$ | 6 + 3 |
| **base, $F = 72$** | | |
| stack depths (`--stacks` or tournaments) | $\log(\text{stack}_i/BB)/\log 200$; each other's chips behind $/STACK$ clipped to 3; the effective stack | 1 + 8 + 1 |
| **with stacks, $F = 82$** | | |
| tournament (`--tourney`) | the prize ladder $9\,\pi_k$; my and the others' start equity $9\,E^0_j$; my equity per chip $\min(E^0_i / \sigma_i, 5)$ with $\sigma_i$ my chip share | 9 + 1 + 8 + 1 |
| **tournament, $F = 101$** | | |
| bounties (`--bounty`) | my and the others' bounty $9\,b_j$ (as shares of all the money) | 1 + 8 |
| **PKO, $F = 110$** | | |

The size slots scale with the menu: `wide` has $A + S = 13$ legal-and-size inputs instead of 9, so $F = 105$ in the
tournament game.

**The size-head layout** (plan B, `features(..., sizehead=True)`): the public inputs, then fold, call and all-in
legal, whether any raise is legal, the smallest and largest legal raise $x_{\min}/10$, $x_{\max}/10$, then each
offered size's pot fraction $x_k/10$ (0 when illegal) and its legal flag $\ell_k$: $F_{base} = 101 - 9 + 6 = 98$
inputs that do not depend on $S$, then $2S$.

## 4. The network $f_\theta$

One network for every seat (`NineNet`: 512 hidden, 4 residual blocks, 32-number embeddings). Each street's bucket
has a learned embedding $E_t[b_t] \in \mathbb{R}^{32}$, zeroed for streets not yet dealt:

$$h_0 = W_{in}\,\big[\phi(x);\ E_0[b_0]\,\mathbb{1}[t\ge0];\ \dots;\ E_3[b_3]\,\mathbb{1}[t\ge3];\ s\big] + c_{in}$$

$$h_{k+1} = h_k + W^{(k)}_2\,\mathrm{GELU}\big(W^{(k)}_1\,\mathrm{LN}(h_k) + c^{(k)}_1\big) + c^{(k)}_2, \qquad k = 0..3$$

$$\bar h = \mathrm{LN}(h_4), \qquad z = W_\pi \bar h + c_\pi \in \mathbb{R}^A, \qquad v = w_v^\top \bar h + c_v$$

The policy is the softmax over the legal slots, and $v$ estimates the result in big blinds:

$$\pi_\theta(a \mid x, b) = \frac{\ell_a\, e^{z_a}}{\sum_{a'} \ell_{a'}\, e^{z_{a'}}}$$

Kestrel runs the same arithmetic in C++ (`Net::forward`; `forward_batch` for many positions at once, checked equal
to it, worst difference 0).

**The size head** (`SizeHeadNet`, plan B, 2026-11-27). The trunk reads the $F_{base}$ inputs only, so it does not
depend on how many sizes are offered. Four type logits $t = W_\pi \bar h + c_\pi \in \mathbb{R}^4$ (fold, call, raise,
all-in), and a raise of pot fraction $x_k$ scores

$$g_k = w_3^\top \mathrm{GELU}\Big(W_2\,\mathrm{GELU}\big(W_h \bar h + W_z\,[x_k,\ \log(x_k + 0.01),\ \mathbb{1}[t = 0]] +
c_z\big) + c_2\Big) + c_3$$

a function of the size, so any size and any number of them can be scored. The joint logits over the menu's moves are

$$z = \Big[t_{fold},\ t_{call},\ t_{raise} + \log\frac{e^{g_k}}{\sum_{k' : \ell_{k'}} e^{g_{k'}}}\ (k = 1..S),\ t_{all\text{-}in}\Big]$$

so $\pi(\text{raise } k) = \pi(\text{raise}) \cdot \pi(k \mid \text{raise})$, and the chance of raising does not depend
on how many sizes are offered. Training offers `size8` (fresh random sizes every decision, so $g$ is learnt over the
whole range); evaluation and Kestrel offer `grid16`. NeuRD (section 6) is unchanged: each decision still has a finite
set of moves.

## 5. The reward

A hand ends with each player's chip change $u_j$ (`payoffs`).

**Cash** (nm1-6, nm9, nm11, nm28): $G_j = u_j / BB$.

**Tournament, ICM** (nm7-8, nm10, nm12-27): chips are worth their prize equity under the Malmuth-Harville model.
Place 1 goes to player $j$ with probability $s_j / \sum s$, each next place to a remaining player in proportion to
chips:

$$\Pr(\text{order } j_1, j_2, \dots) = \prod_{k} \frac{s_{j_k}}{\sum_{m \notin \{j_1..j_{k-1}\}} s_m}, \qquad
E_j(s) = \sum_{k} \pi_k \Pr(j \text{ finishes } k)$$

computed exactly over the $2^P$ subsets of placed players (`icm.icm`). The reward is the change in equity, in big
blinds at the table's average chip value:

$$G_j = \big(E_j(s + u) - E_j(s)\big) \cdot \frac{\sum_m s_m}{BB}$$

**Knockouts, PKO** (nm29-30): a share $\beta = 0.5$ of the money is in bounties $b_j$; a bust pays half its bounty
in cash to the winner(s) of the pot that took the player's last chips (the pot level at its contribution, among the
live players who put in at least as much, split on ties; until 2026-11-23 the hand's biggest winner) and adds the
other half to that winner's bounty. Each player's equity is ICM on
the regular prizes plus its chip share of the whole bounty pool (nm30's correction, 2026-11-16):

$$E^0_j = E_j^{ICM}(s) + \frac{s_j}{\sum s} \sum_m b_m, \qquad
E^1_j = E_j^{ICM}(s') + k_j + \frac{s'_j}{\sum s'} \sum_m b'_m$$

with $s'$ the stacks after the hand, $k_j$ the cash collected and $b'$ the bounties left. Then $\sum_j E^1_j =
\sum_j E^0_j$, so the reward sums to zero over the table. nm29 used $b_j\, s_j/\sum s$ (its own bounty only), which
created $+8.0$ bb a table a hand on average, and a copy of nm29 beat itself by $+90$ bb/100.

## 6. Training

Every decision $n$ of the learning network's seats in a batch of 16,384 tables gives $(\phi_n, b_n, \ell_n, a_n,
G_n)$. With the value baseline and a running scale,

$$A_n = \frac{G_n - v_n}{\hat\sigma}, \qquad \hat\sigma \leftarrow 0.99\,\hat\sigma + 0.01\,\overline{|G - v|}$$

**Policy term.** A2C / REINFORCE (nm1-3, nm5-8): $\mathcal{L}_{pg} = -A_n \log \pi_\theta(a_n)$.
NeuRD, Neural Replicator Dynamics (Hennes et al. 2020; nm4, nm9 onward) acts on the logits centred over the legal
slots, $y_a = z_a - \frac{1}{|\ell|}\sum_{a' \in \ell} z_{a'}$, with an importance-weighted value of the sampled
slot, and stops pushing a logit beyond $\pm\beta_N$:

$$\hat q_n = \frac{A_n}{\max(\pi_\theta(a_n), 0.05)}, \qquad
\mathcal{L}_{pg} = -\hat q_n\, y_{a_n}\; \mathbb{1}\big[(\hat q_n > 0 \wedge y_{a_n} < \beta_N) \vee (\hat q_n < 0 \wedge
y_{a_n} > -\beta_N)\big]$$

**The magnet and the loss.** A slowly moving copy $\theta_m$ pulls the policy back (regularised Nash dynamics, as in
R-NaD), with an entropy bonus and a value regression:

$$\mathrm{KL}_n = \sum_a \pi_\theta(a)\big(\log \pi_\theta(a) - \log \pi_{\theta_m}(a)\big), \qquad
H_n = -\sum_a \pi_\theta(a) \log \pi_\theta(a)$$

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_n \Big[\mathcal{L}_{pg,n} + \tfrac{1}{2}\,\tfrac{(v_n - G_n)^2}{25} +
\lambda_{KL}\,\mathrm{KL}_n - c_H H_n\Big]$$

Adam (learning rate $3\cdot10^{-4}$), gradients clipped to norm 1. After each step

$$\theta_m \leftarrow (1-\tau_m)\,\theta_m + \tau_m\,\theta, \qquad \bar\theta \leftarrow (1-\tau_a)\,\bar\theta +
\tau_a\,\theta$$

with $\tau_m = 3\cdot10^{-4}$, $\tau_a = 10^{-3}$. **The shipped model is the average $\bar\theta$**, not $\theta$.

**Who it plays.** Each batch is split: a share (`exploit_share`, 0.25) of tables seat an exploiter in one seat, a
fresh copy trained by plain policy gradient against the network and reset every hour; a share (`pool_share`, 0.4-0.5)
seats the network in $k \sim U\{1..8\}$ seats against one population member holding the rest (rule players `reg`,
`tag`, `maniac`, `station`, `random`, past snapshots); the rest is self-play. The network learns only from its own
seats.

**A warm start across menus** (`remap_menu`, nm48). Each new size slot $j$ takes the old slot $m(j)$ whose size is
nearest on a log scale, by vote over the contexts the slots are used in (opens, each re-raise level, heads-up and
multiway bets and raises; ties to the smaller); `mtt` to `wide` gives $m = (1, 1, 2, 3, 3)$. With $n_k = |\{j : m(j) =
k\}|$ copies of old slot $k$, the policy rows and the legal and size input columns are copied and corrected so the new
network starts where the old one was:

$$z^{new}_{2+j} = z^{old}_{2+m(j)} - \log n_{m(j)}, \qquad W^{new}_{in}[:, \ell_{2+j}] = W^{old}_{in}[:, \ell_{2+m(j)}] /
n_{m(j)}$$

The copy of nm25 on `wide` read +0.13 [−0.33, +0.59] bb/100 against nm25 (20,000 deals × 9).

**A size-head copy of nm25** (`ninemax_sizehead_init.py`). The trunk, the embeddings, the value and the fold, call and
all-in rows are copied (the raise type row from nm25's smallest size); then the copy, offered exactly nm25's legal
moves and sizes, is distilled on nm25's self-play decisions:

$$\mathcal{L} = \mathrm{KL}\big(\pi_{nm25}(\cdot \mid x) \,\|\, \pi_\theta(\cdot \mid x)\big) + \tfrac12 (v_\theta - v_{nm25})^2$$

After 300 steps (about 11 M decisions) the KL was 0.008; the copy read −0.38 [−1.29, +0.54] bb/100 against nm25 when
offered nm25's sizes and +1.03 [+0.10, +1.95] on `grid16` (5,000 deals × 9).

## 7. Measurement

**Duplicate match** (`duplicate`): the hero in each seat in turn against 8 copies of the opponent on the same $n$
deals,

$$d_k = \frac{1}{P}\sum_{\text{seat}} \frac{u_{\text{seat},k}}{BB}, \qquad \text{bb/100} = 100\,\bar d, \qquad
\text{CI}_{95} = 100\,\big(\bar d \pm 1.96\, \mathrm{sd}(d)/\sqrt n\big)$$

A copy against itself must read about 0: the zero-sum check that caught nm29.

**The version ladder** (`ninemax_ladder.py`): each pair both ways, $\Delta_{AB} = \tfrac12(\mathrm{d}(A\text{ vs }8B) -
\mathrm{d}(B\text{ vs }8A))$ with standard error $\mathrm{se}_{AB}$, and ratings by weighted least squares with the
anchor (nm17) at 0:

$$\hat r = \arg\min_r \sum_{A<B} \frac{(r_A - r_B - \Delta_{AB})^2}{\mathrm{se}_{AB}^2}, \qquad r_{nm17} = 0$$

**Exploitability.** A fresh copy trained for 30 minutes against 8 copies of the model (same loss as section 6 with
$\lambda_{KL} = 0$); its duplicate result is the hole it found. **LBR**: a best response by enumeration of the
opponent's range after the flop, heads-up, assuming a check-down.

**The pipeline's checks** (`pipeline.py`, from nm45): no-harm is a sequential test against nm25 (8 looks, at most
40,000 deals, $\alpha = 0.025$, margin 0.5 bb/100); other players are paired, the run and the baseline on the same
deals against each rule player, the 95% interval of the per-deal difference above −2.0; the exploiter check is the
mean of four 30-minute exploiters (limit +1.25, nm25's +0.75 plus 0.5).

**What sizes say about the hand** (`ninemax_sizes.py`, the `sizes` report). Per context (pre-flop open, each re-raise
level, each street's heads-up and multiway bets and raises), over the decisions that bet or raise, with $s$ the
player's strength and the groups the sizes chosen:

$$\eta^2 = \frac{\sum_{\text{size } k} n_k\,(\bar s_k - \bar s)^2}{\sum_n (s_n - \bar s)^2}$$

0 when the size says nothing about the hand, 1 when it names it; reported overall and by street (weighted by bets),
against nm25's. nm25's river reads 0.100; a 3-minute wide-menu network 0.034 (8,000 deals; a check of the tool).

**Search gain** (`search_paired.py`, Kestrel's `search_match`): per deal, the hero's result with search minus without,
on the same cards; the interval is over deals.

## 8. Search at decision time

At the hero's decision with known cards $c$ and the opponents' earlier decisions $\{(x_t, a_t, j_t)\}$, draw $N$
completions $\omega^{(n)}$ of the unknown cards (opponents' holdings, rest of the board) uniformly, and weight each by
how likely the network was to act as the opponents did with those cards:

$$w_n \propto \prod_t \pi_\theta\big(a_t \mid x_t, b_{j_t}(\omega^{(n)})\big), \qquad \textstyle\sum_n w_n = 1,
\qquad \mathrm{ESS} = 1 / \sum_n w_n^2$$

For each legal slot $a$, play every completion out with the network for every seat (sampling the policy) and score
the hero's reward $R(a, \omega)$ of section 5:

$$Q(a) = \sum_n w_n\, R(a, \omega^{(n)}), \qquad a^* = \arg\max_a Q(a)$$

**Robust** (after Pluribus): every completion is played four times, the opponents continuing with $\pi_\theta$ or
with it tilted towards folding, calling or raising, $\tilde\pi_c(a) \propto \pi_\theta(a)\, \rho^{\mathbb{1}[a \in
K_c]}$ with $\rho = 5$; the slot's value is its worst case, $Q(a) = \min_c \sum_n w_n R_c(a, \omega^{(n)})$.

$Q$ is a Monte Carlo estimate with error about $\mathrm{sd}(R)/\sqrt{\mathrm{ESS}}$, and $\arg\max$ picks the largest
error as readily as the largest value: at $N = 256$ search lost 5-6.5 bb/100, at $N = 1{,}024$ it gained +3.7 to
+6.2 (Kestrel, 2026-11-17).

## 9. Embeddings and similarity (Kestrel)

**A spot's embedding** projects the last hidden layer with a fixed Gaussian matrix $R \in \mathbb{R}^{128 \times
512}$, $R_{ij} \sim \mathcal{N}(0, 1)$ from a fixed seed (Johnson-Lindenstrauss: inner products kept to about
$\pm0.1$):

$$e(x) = \frac{R\,\bar h(x)}{\lVert R\,\bar h(x) \rVert}, \qquad \mathrm{sim}(x, y) = e(x)^\top e(y) = \cos\angle\big(e(x),
e(y)\big)$$

**An opponent's profile** uses six rates (VPIP, pre-flop raise, 3-bet, fold to c-bet, aggression frequency, went to
showdown) with $x_d$ events in $n_d$ chances, shrunk towards the reference types' mean $\mu_d$ with $k = 8$ chances
of prior, standardised by their spread $\sigma_d$ and weighted $w = (2, 2, 1.5, 1, 1, 1)$:

$$\hat r_d = \frac{x_d + k\,\mu_d}{n_d + k}, \qquad v_d = w_d\,\frac{\hat r_d - \mu_d}{\sigma_d}, \qquad
\text{type} = \arg\max_\tau \cos(v, v_\tau)$$

## 10. The heads-up blueprint (for completeness)

The heads-up model is not a network but a blueprint strategy from counterfactual regret minimisation over an abstract
game (card buckets and a betting menu). At iteration $T$ the regret of not always playing $a$ at information set $I$
is $R^T(I, a) = \sum_{t \le T} \big(v^t(I, a) - v^t(I)\big)$ weighted by the opponent's reach, and the strategy is
regret matching, $\sigma^{T+1}(I, a) \propto \max(R^T(I, a), 0)$; the average strategy converges to equilibrium.
Kestrel's river re-solve runs CFR+ on the river subgame from both blueprint ranges with a gadget that keeps every
opponent hand at or below its blueprint value (safe subgame solving). The turn re-solve (`resolve_turn.cpp`, 2026-11-24)
solves from the turn with every river card exact; `../blueprint/model.md` section 5.5 has its arithmetic and results.

## 11. The versions

All 9-max runs: $P = 9$, $A = 6$ (nm48: 8), 512 hidden, 4 blocks, $\lambda_{KL} = 0.3$ (nm19: 0.05), $\tau_m = 3\cdot10^{-4}$,
$\tau_a = 10^{-3}$, batch 16,384. "Init" is the warm start. Ratings are the ladder's (bb/100 against nm17 in nm25's
game, ±0.15-0.17), where played.

| Run | Game | $F$ | Policy update | $c_H$ | Pool / exploiter | Hours | Init | Change tried | Ladder |
|---|---|---|---|---|---|---|---|---|---|
| nm1 | cash 100bb, `medium` | 72 | A2C | 0.01 | self-play only | 24 | — | the first network | |
| nm2-3 | cash 100bb | 72 | A2C | 0.01 | tag, maniac, station, random; 0.5 | 22-24 | — | a population of rule players | |
| nm4 | cash 100bb | 72 | NeuRD $\beta_N$ 2 | 0.01 | same | 24 | — | NeuRD | |
| nm5 | cash 100bb | 72 | A2C | 0.01 | 0.4; exploiter 0.25 | 24 | — | an exploiter in training | |
| nm6 | cash 5-200bb | 82 | A2C | 0.01 | same | 24 | — | stack depths | |
| nm7-8 | tournament (ICM) | 101 | A2C | 0.02 | same | 24, 20 | — | ICM reward | |
| nm9 | cash 100bb | 72 | NeuRD 2 | 0.01 | same | 24 | — | NeuRD with exploiter | |
| nm10 | tournament, `mtt`, 10-25bb (and 5-10bb) | 101 | NeuRD 2 | 0.02 | same | 24 | — | short stacks, no open-limp | |
| nm11 | cash 20-50bb | 82 | NeuRD 2 | 0.01 | same | 24 | — | shallow cash | |
| nm12 | tournament 10-25bb | 101 | NeuRD 2 | 0.02 | same | 24 | — | the 10-25bb game | −1.1 |
| nm13 | same | 101 | NeuRD 2 | 0.02 | reg ×2, tag, maniac | 12 | nm12 | regular players in the pool | −1.1 |
| nm14 | same | 101 | NeuRD 2 | 0.02 | same | 12 | — | from scratch | −1.1 |
| nm15 | + big-blind ante | 101 | NeuRD 2 | 0.02 | same | 12 | nm13 | the ante | −0.36 |
| nm16 | + ante | 101 | NeuRD 2 | 0.02 | same | 12 | — | from scratch | −0.26 |
| **nm17** | + ante, half the final tables short-handed | 101 | NeuRD 2 | 0.02 | same | 12 | nm15 2h30 | short tables | **0 (anchor)** |
| nm18 | ante, no short tables | 101 | NeuRD 2 | 0.02 | same | 2 | nm16 12h | fine-tune | −0.43 |
| nm19 | same | 101 | NeuRD 2 | 0.02 | same | 2 | nm16 12h | $\lambda_{KL}$ 0.05 | −0.36 |
| **nm20** | same | 101 | NeuRD **6** | 0.02 | same | 2 | nm16 12h | $\beta_N$ 6 | **+1.04** |
| nm21-24 | ante, short 0.5 | 101 | NeuRD 2 | 0.02 | same | 6 | — | bucket layouts (200/200/200, 100/200/500, 1000/500/200) | nm21 −0.56; layouts tied |
| **nm25** | ante, short 0.5 | 101 | NeuRD **4** | **0.04** | reg ×2, tag, maniac, station | 6 | nm20 | the champion's settings | **+0.96** |
| nm26 | same | 101 | NeuRD 4 | 0.04 | same | 12 | nm25 | longer | +1.03 |
| nm27 | 5-25bb | 101 | NeuRD 4 | 0.04 | same | 12 | nm25 | shorter stacks | +1.03 |
| nm28 | **6-max cash** 50-250bb, short 0.2 | 82 | NeuRD 4 | 0.04 | same | 12 | — | 6-max cash | (cash; not on the ladder) |
| nm29 | **PKO**, bounty 0.5 | 110 | NeuRD 4 | 0.04 | same | 12 | — | knockouts (reward not zero-sum: void) | |
| nm30 | PKO, conserved bounty equity | 110 | NeuRD 4 | 0.04 | same | 8 | nm29 | the corrected reward | |
| nm45 | ante, short 0.5 | 101 | NeuRD 4 | 0.04 | + league 0.15, PFSP, 8 frozen, archetype styles | 6 | nm25 | the league on fixed seating | not adopted: station −16.5 [−17.2, −15.8] |
| nm46 | ante, short 0.5 | 101 | NeuRD 4 | 0.04 | maniac ×2, tag, station, reg; 0.6; exploiter 0.35, reset 45 min | 6 | nm25 | an aggressive pool (by hand, checks after) | passed nm45's checks; exploiter +1.10 |
| nm47 | same as nm46 | 101 | NeuRD 4 | 0.04 | same as nm46 | 6 | nm25 | nm46's recipe, pre-registered | running |
| nm48 | ante, short 0.5, **`wide` menu** | 105 | NeuRD 4 | 0.04 | nm25's | 6 | nm25 (remapped) | bet sizing A: 8 moves | queued after nm47 |

nm25 is the champion (Kestrel's default) over nm26 and nm27 because they tie it on the ladder and nm26 left a larger
hole for the exploiter (+1.1 against +0.2). The road from nm12 (−1.1) to nm25 (+0.96) is about 2 bb/100: the ante,
NeuRD's larger $\beta_N$ and more entropy did most of it; buckets, longer training and shorter stacks did not.
