// Safe re-solving on the turn, with every river card solved exactly (the baseline for value networks at the leaves;
// see research/docs/2026-11-24-slumbot-nl.md). Same gadget, CFR+ and card handling as resolve.cpp's river re-solve,
// extended across the river deal:
//
//   turn nodes      strategies over the 1,128 hands live on the 4-card board
//   the river deal  a public chance node: each of the 48 river cards gets its own copy of every river node below,
//                   over the 1,081 hands live on that 5-card board; a hand pair's value is the mean over the 44 river
//                   cards that clash with neither hand (weight 1/44, so turn folds and river results share one scale)
//   all-in on turn  a showdown node before the river: the same mean over river cards
//
// Arms (test mode), each judged by the opponent's exact best response over turn and river, given their blueprint
// range, in chips per hand pair (as resolve.cpp):
//   blueprint             no re-solve
//   unsafe re-solve       trusts the blueprint's opponent range
//   safe re-solve         the gadget: no opponent hand may gain over its blueprint value (CBV); rivers re-solved
//   safe, our river fixed the gadget, with OUR river play fixed to the blueprint's and the opponent's river play
//                         re-solved (a best response to it): blueprint values at the leaves, valued pessimistically,
//                         which is what a river value network would replace
//   safe, both rivers fixed both players' river play fixed to the blueprint's: the naive form, which trusts that the
//                         opponent will not deviate on the river
//
// Usage: resolve_turn ABS RUN --test SPOTS [--iters 200] [--purify 0.05] [--seed 1] [--threads 8] [--floor 0.02]
// Memory per spot: the largest turn subgame of the standard tree holds 5,476 river action slots, 2.3 GB over 48
// river cards (regrets and averages); the median holds 88 (37 MB). Keep --threads modest.
#define RESOLVE_LIB
#include "resolve.cpp"

static const float RIVER_W = 1.0f / 44.0f;

// the turn board and the 48 river boards, with each river hand's index in the turn's hand list
struct TurnBoards {
  Board tb;
  std::vector<Board> rb;
  std::vector<std::vector<int>> map;  // map[k][j]: turn index of river board k's hand j
  int nr = 0;
};

// a 4-card board: every hand that misses it, with its pre-flop, flop and turn buckets (river fields unused)
static void set_turn_board(Board& b, const uint8_t* cards) {
  bool used[52] = {false};
  for (int k = 0; k < 4; ++k) b.card[k] = cards[k], used[cards[k]] = true;
  b.card[4] = 0;
  b.n = 0;
  for (int h = 0; h < 1326; ++h) {
    int a = COMBO_A[h], c = COMBO_B[h];
    if (used[a] || used[c]) continue;
    int i = b.n++;
    b.hand[i] = h;
    uint8_t cs[7] = {(uint8_t)a, (uint8_t)c, cards[0], cards[1], cards[2], cards[3], 0};
    b.bucket[i][0] = (int)hand_index_last(&IX->pre, cs);
    b.bucket[i][1] = FB[hand_index_last(&IX->flop, cs)];
    b.bucket[i][2] = TB[hand_index_last(&IX->turn, cs)];
    b.bucket[i][3] = -1;
    b.rank[i] = 0;
  }
}

static void prep(TurnBoards& t, const uint8_t* cards4) {
  set_turn_board(t.tb, cards4);
  std::vector<int> idx(1326, -1);
  for (int i = 0; i < t.tb.n; ++i) idx[t.tb.hand[i]] = i;
  t.rb.resize(48);
  t.map.assign(48, {});
  t.nr = 0;
  for (int c = 0; c < 52; ++c) {
    if (c == cards4[0] || c == cards4[1] || c == cards4[2] || c == cards4[3]) continue;
    uint8_t five[5] = {cards4[0], cards4[1], cards4[2], cards4[3], (uint8_t)c};
    Board& r = t.rb[t.nr];
    set_board(r, five);
    t.map[t.nr].resize(r.n);
    for (int j = 0; j < r.n; ++j) t.map[t.nr][j] = idx[r.hand[j]];
    ++t.nr;
  }
}

// the re-solve's tables: turn nodes over turn hands, river nodes once per river card over that board's hands
struct TurnSol {
  std::unordered_map<int, size_t> at4, at5;
  std::vector<float> R4, S4;
  std::vector<std::vector<float>> R5, S5;
  int freeze = -1;  // river play fixed to the blueprint's for player 0 or 1, both (2), or neither (-1)
  bool frozen(const Node& nd) const { return nd.street == 3 && (freeze == 2 || freeze == nd.player); }
};

static void build_turn_sol(TurnSol& s, const TurnBoards& t, int root) {
  size_t tot4 = 0, tot5 = 0;
  std::vector<int> stack{root};
  while (!stack.empty()) {
    int id = stack.back();
    stack.pop_back();
    const Node& nd = T[id];
    if (nd.kind != 0) continue;
    if (nd.street == 2) s.at4[id] = tot4, tot4 += nd.children.size() * (size_t)t.tb.n;
    else s.at5[id] = tot5, tot5 += nd.children.size() * (size_t)1081;
    for (int c : nd.children) stack.push_back(c);
  }
  s.R4.assign(tot4, 0.0f), s.S4.assign(tot4, 0.0f);
  s.R5.assign(t.nr, std::vector<float>()), s.S5.assign(t.nr, std::vector<float>());
  if (s.freeze != 2)
    for (int k = 0; k < t.nr; ++k) s.R5[k].assign(tot5, 0.0f), s.S5[k].assign(tot5, 0.0f);
}

// strategy at node id for every hand of the space c (-1: turn, else river card k): sig[a * n + i]. Blueprint when
// sol is null or the river is frozen; else the re-solve's current (avg false) or average strategy.
static void node_strat(const TurnBoards& t, const TurnSol* sol, bool avg, int c, int id, float* sig) {
  const Node& nd = T[id];
  const Board& B = c < 0 ? t.tb : t.rb[c];
  const int n = B.n, na = (int)nd.children.size();
  auto bp = [&](int i, int a) { return SIG[nd.offset + (uint64_t)B.bucket[i][nd.street] * na + a]; };
  if (!sol || sol->frozen(nd)) {
    for (int a = 0; a < na; ++a)
      for (int i = 0; i < n; ++i) sig[a * n + i] = bp(i, a);
    return;
  }
  const float* x = c < 0 ? (avg ? sol->S4 : sol->R4).data() + sol->at4.at(id)
                         : (avg ? sol->S5[c] : sol->R5[c]).data() + sol->at5.at(id);
  for (int i = 0; i < n; ++i) {
    float tot = 0;
    for (int a = 0; a < na; ++a) tot += std::max(x[a * n + i], 0.0f);
    for (int a = 0; a < na; ++a)
      sig[a * n + i] = tot > 0 ? std::max(x[a * n + i], 0.0f) / tot : (avg ? bp(i, a) : 1.0f / na);
  }
}

// mode 0: CFR+ for `trav`; mode 1: best response of `trav` to the other player's average (or blueprint) strategy
static void walk(const TurnBoards& t, TurnSol* sol, int mode, int c, int id, int trav, const float* rt,
                 const float* ro, float* out, Arena& ar, float weight);

// the river deal from turn space: every river card in turn, values back on the turn's scale
static void deal_river(const TurnBoards& t, TurnSol* sol, int mode, int id, int trav, const float* rt, const float* ro,
                       float* out, Arena& ar, float weight) {
  const int n4 = t.tb.n;
  std::fill(out, out + n4, 0.0f);
  size_t mark = ar.top;
  float* rt5 = ar.get(1081);
  float* ro5 = ar.get(1081);
  float* o5 = ar.get(1081);
  for (int k = 0; k < t.nr; ++k) {
    const std::vector<int>& m = t.map[k];
    const int n5 = t.rb[k].n;
    bool live = false;
    for (int j = 0; j < n5; ++j) rt5[j] = rt ? rt[m[j]] : 0, ro5[j] = ro[m[j]], live |= ro5[j] > 0;
    if (!live) continue;
    walk(t, sol, mode, k, id, trav, rt ? rt5 : nullptr, ro5, o5, ar, weight);
    for (int j = 0; j < n5; ++j) out[m[j]] += RIVER_W * o5[j];
  }
  ar.top = mark;
}

static void walk(const TurnBoards& t, TurnSol* sol, int mode, int c, int id, int trav, const float* rt,
                 const float* ro, float* out, Arena& ar, float weight) {
  const Node& nd = T[id];
  if (c < 0 && (nd.street == 3 || nd.kind == 2)) return deal_river(t, sol, mode, id, trav, rt, ro, out, ar, weight);
  const Board& B = c < 0 ? t.tb : t.rb[c];
  const int n = B.n;
  if (nd.kind != 0) return terminal(nd, trav, B, ro, out);
  bool live = false;
  for (int i = 0; i < n && !live; ++i) live = ro[i] > 0;
  if (!live) {
    std::fill(out, out + n, 0.0f);
    return;
  }
  size_t mark = ar.top;
  const int na = (int)nd.children.size();
  float* sig = ar.get((size_t)na * n);
  float* child = ar.get(n);
  float* reach = ar.get(n);
  if (mode == 1) {  // best response
    if (nd.player == trav) {
      std::fill(out, out + n, -1e30f);
      for (int a = 0; a < na; ++a) {
        walk(t, sol, mode, c, nd.children[a], trav, nullptr, ro, child, ar, weight);
        for (int i = 0; i < n; ++i) out[i] = std::max(out[i], child[i]);
      }
    } else {
      node_strat(t, sol, true, c, id, sig);
      std::fill(out, out + n, 0.0f);
      for (int a = 0; a < na; ++a) {
        for (int i = 0; i < n; ++i) reach[i] = ro[i] * sig[a * n + i];
        walk(t, sol, mode, c, nd.children[a], trav, nullptr, reach, child, ar, weight);
        for (int i = 0; i < n; ++i) out[i] += child[i];
      }
    }
    ar.top = mark;
    return;
  }
  node_strat(t, sol, false, c, id, sig);
  std::fill(out, out + n, 0.0f);
  if (nd.player == trav) {
    float* vals = ar.get((size_t)na * n);
    for (int a = 0; a < na; ++a) {
      for (int i = 0; i < n; ++i) reach[i] = rt[i] * sig[a * n + i];
      walk(t, sol, mode, c, nd.children[a], trav, reach, ro, vals + (size_t)a * n, ar, weight);
      for (int i = 0; i < n; ++i) out[i] += sig[a * n + i] * vals[(size_t)a * n + i];
    }
    if (!sol->frozen(nd)) {
      float* R = c < 0 ? sol->R4.data() + sol->at4.at(id) : sol->R5[c].data() + sol->at5.at(id);
      float* S = c < 0 ? sol->S4.data() + sol->at4.at(id) : sol->S5[c].data() + sol->at5.at(id);
      for (int a = 0; a < na; ++a)
        for (int i = 0; i < n; ++i) {
          R[a * n + i] = std::max(R[a * n + i] + vals[(size_t)a * n + i] - out[i], 0.0f);  // CFR+
          S[a * n + i] += weight * rt[i] * sig[a * n + i];                                  // linear averaging
        }
    }
  } else {
    for (int a = 0; a < na; ++a) {
      for (int i = 0; i < n; ++i) reach[i] = ro[i] * sig[a * n + i];
      walk(t, sol, mode, c, nd.children[a], trav, rt, reach, child, ar, weight);
      for (int i = 0; i < n; ++i) out[i] += child[i];
    }
  }
  ar.top = mark;
}

// as resolve.cpp's solve(), in turn space
static void solve_turn(const TurnBoards& t, TurnSol& s, int root, int us, const float* r_us, const float* r_opp,
                       const float* cbv, bool safe, int iters, Arena& ar) {
  const int n = t.tb.n, opp = 1 - us;
  build_turn_sol(s, t, root);
  std::vector<float> g(2 * (size_t)n, 0.0f), ro(n), v(n), rg(r_opp, r_opp + n);
  if (safe) {
    float top = *std::max_element(rg.begin(), rg.end());
    for (auto& x : rg) x = std::max(x, FLOOR * top);
  }
  for (int it = 1; it <= iters; ++it) {
    auto enter = [&](int i) {
      float e = std::max(g[i], 0.0f), q = std::max(g[n + i], 0.0f);
      return e + q > 0 ? e / (e + q) : 0.5f;
    };
    for (int i = 0; i < n; ++i) ro[i] = rg[i] * (safe ? enter(i) : 1.0f);
    walk(t, &s, 0, -1, root, us, r_us, ro.data(), v.data(), ar, (float)it);
    walk(t, &s, 0, -1, root, opp, rg.data(), r_us, v.data(), ar, (float)it);
    if (safe)
      for (int i = 0; i < n; ++i) {
        float e = enter(i), mix = e * v[i] + (1 - e) * cbv[i];
        g[i] = std::max(g[i] + v[i] - mix, 0.0f);
        g[n + i] = std::max(g[n + i] + cbv[i] - mix, 0.0f);
      }
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: resolve_turn ABS RUN --test SPOTS [--iters 200] [--purify 0.05] [--seed 1] [--threads 8]\n");
    return 1;
  }
  std::string abs = argv[1], run = argv[2];
  int spots = 50, iters = 200, threads = 8;
  double purify = 0.05;
  uint64_t seed = 1;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--test") spots = atoi(argv[++i]);
    else if (a == "--iters") iters = atoi(argv[++i]);
    else if (a == "--purify") purify = atof(argv[++i]);
    else if (a == "--seed") seed = strtoull(argv[++i], nullptr, 10);
    else if (a == "--threads") threads = atoi(argv[++i]);
    else if (a == "--floor") FLOOR = (float)atof(argv[++i]);
  }
  if (!load_table(abs, run, purify)) return 1;
  const int M = 5;
  const char* name[M] = {"blueprint", "unsafe re-solve", "safe re-solve", "safe, our river fixed",
                         "safe, both rivers fixed"};
  std::vector<double> ex[M], worst[M], secs[M];
  for (int m = 0; m < M; ++m) ex[m].assign(spots, 0), worst[m].assign(spots, 0), secs[m].assign(spots, 0);
  std::vector<int> kept(spots, 0);
  double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (int k = 0; k < spots; ++k) {
    Rng rng(seed * 1000003ULL + k);
    TurnBoards* tbp = new TurnBoards;
    TurnBoards& tb = *tbp;
    Board* deal = new Board;
    Arena ar;
    ar.buf.resize((size_t)1326 * 80 * 64);
    // a turn decision node at the start of the turn, by blueprint self-play on a random deal, with both reaches
    int root = -1;
    std::vector<float> reach[2];
    for (int tries = 0; tries < 50 && root < 0; ++tries) {
      deal_board(rng, *deal);
      prep(tb, deal->card);
      const Board& b = tb.tb;
      const int n = b.n;
      reach[0].assign(n, 1.0f), reach[1].assign(n, 1.0f);
      int h0, h1;
      do h0 = (int)(rng() % n), h1 = (int)(rng() % n);
      while (COMBO_A[b.hand[h0]] == COMBO_A[b.hand[h1]] || COMBO_A[b.hand[h0]] == COMBO_B[b.hand[h1]] ||
             COMBO_B[b.hand[h0]] == COMBO_A[b.hand[h1]] || COMBO_B[b.hand[h0]] == COMBO_B[b.hand[h1]]);
      int id = 0;
      while (T[id].kind == 0 && T[id].street < 2) {
        const Node& nd = T[id];
        int na = (int)nd.children.size(), p = nd.player, mine = p == 0 ? h0 : h1;
        double u = (rng() >> 11) * (1.0 / 9007199254740992.0), acc = 0;
        int pick = na - 1;
        for (int a = 0; a < na; ++a) {
          acc += SIG[nd.offset + (uint64_t)b.bucket[mine][nd.street] * na + a];
          if (u < acc) {
            pick = a;
            break;
          }
        }
        for (int i = 0; i < n; ++i) reach[p][i] *= SIG[nd.offset + (uint64_t)b.bucket[i][nd.street] * na + pick];
        id = nd.children[pick];
      }
      if (T[id].kind == 0 && T[id].street == 2) root = id;
    }
    delete deal;
    if (root < 0) {
      delete tbp;
      continue;
    }
    const int n = tb.tb.n, us = T[root].player, opp = 1 - us;
    const float* r_us = reach[us].data();
    const float* r_opp = reach[opp].data();
    std::vector<float> comp(n), cbv(n), br(n);
    compatible(tb.tb, r_us, comp.data());
    double z = 0;
    for (int i = 0; i < n; ++i) z += r_opp[i] * comp[i];
    if (z <= 0) {
      delete tbp;
      continue;
    }
    walk(tb, nullptr, 1, -1, root, opp, nullptr, r_us, cbv.data(), ar, 1);  // the opponent's values vs the blueprint
    for (int m = 0; m < M; ++m) {
      double ts = omp_get_wtime();
      TurnSol s;
      TurnSol* use = nullptr;
      if (m > 0) {
        s.freeze = m == 3 ? us : m == 4 ? 2 : -1;
        solve_turn(tb, s, root, us, r_us, r_opp, cbv.data(), m >= 2, iters, ar);
        use = &s;
      }
      secs[m][k] = omp_get_wtime() - ts;
      walk(tb, use, 1, -1, root, opp, nullptr, r_us, br.data(), ar, 1);
      double e = 0, w = 0;
      for (int i = 0; i < n; ++i) {
        e += r_opp[i] * br[i];
        if (comp[i] > 0) w = std::max(w, (double)(br[i] - cbv[i]) / comp[i]);
      }
      ex[m][k] = e / z;
      worst[m][k] = w;
    }
    kept[k] = 1;
    delete tbp;
  }
  int nk = 0;
  for (int k = 0; k < spots; ++k) nk += kept[k];
  printf("%d turn spots, %d CFR+ iterations, %.0fs\n", nk, iters, omp_get_wtime() - t0);
  for (int m = 0; m < M; ++m) {
    double me = 0, mw = 0, d = 0, d2 = 0, ms = 0;
    for (int k = 0; k < spots; ++k)
      if (kept[k]) {
        me += ex[m][k], mw += worst[m][k], ms += secs[m][k];
        double x = ex[m][k] - ex[0][k];
        d += x, d2 += x * x;
      }
    me /= nk, mw /= nk, d /= nk, ms /= nk;
    double se = std::sqrt(std::max(d2 / nk - d * d, 0.0) / std::max(nk - 1, 1));
    printf("%-24s opponent best response %+8.1f chips per hand pair  (vs blueprint %+7.1f, se %.1f)  worst hand "
           "gain over blueprint %+8.1f  %.1fs a spot\n",
           name[m], me, d, se, mw, ms);
  }
  return 0;
}
