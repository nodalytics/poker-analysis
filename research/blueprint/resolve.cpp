// Safe re-solving on the river (Brown & Sandholm 2017; the re-solve gadget of Burch, Johanson & Bowling 2014), with
// exact cards: the blueprint plays each bucket one way, the re-solve plays each actual hand.
//
// The subgame is the blueprint tree below a river node, played with both players' blueprint ranges (each holding
// weighted by the product of the blueprint's probabilities of its actions so far). Vector CFR+ over every live hand
// solves it. Safe: at the root each opponent hand may take its counterfactual best-response value against the
// blueprint (CBV) instead of entering; the re-solve minimises what the opponent can get given that option, so no
// opponent hand gains more than against the blueprint, whatever range the opponent really arrived with. Unsafe (no
// gadget) trusts the blueprint's opponent range, which a deviating opponent exploits (LBR +996 with TexasSolver).
//
// Test mode samples river spots from blueprint self-play and, for the blueprint, the unsafe and the safe re-solve,
// computes in the subgame with exact best responses:
//   exploitability  the opponent's best-response value against our strategy, given their blueprint range
//   worst gain      the most any single opponent hand gains above its blueprint value (CBV), per hand pair:
//                   what an opponent who arrives with more of that hand wins extra; the safety property is 0
//
// Usage: resolve ABS RUN --test SPOTS [--iters 500] [--purify 0.05] [--seed 1]
#include <omp.h>

#include <cmath>
#include <unordered_map>

#include "vec.hpp"

static std::vector<float> SIG;  // blueprint strategy, normalised (and purified), layout offset + bucket * na + a
static float FLOOR = 0.02;      // safe re-solve: minimum opponent reach, as a share of the largest (--floor)

struct Sub {
  int root, us;                       // us: the player whose strategy is re-solved
  std::unordered_map<int, size_t> at; // node id -> start of its block in R, S (na * n floats, a * n + i)
  std::vector<float> R, S;
  int n;
};

static void build_sub(Sub& s, int root, int n) {
  s.root = root, s.n = n;
  size_t total = 0;
  std::vector<int> stack{root};
  while (!stack.empty()) {
    int id = stack.back();
    stack.pop_back();
    const Node& nd = T[id];
    if (nd.kind != 0) continue;
    s.at[id] = total;
    total += nd.children.size() * (size_t)n;
    for (int c : nd.children) stack.push_back(c);
  }
  s.R.assign(total, 0.0f);
  s.S.assign(total, 0.0f);
}

// strategy at node `id` for hand i: blueprint (sub == nullptr), the re-solve's current (avg = false) or average
static inline float strat(const Sub* sub, bool avg, const Board& b, int id, int i, int a) {
  const Node& nd = T[id];
  const int na = (int)nd.children.size();
  if (!sub) return SIG[nd.offset + (uint64_t)b.bucket[i][nd.street] * na + a];
  const float* x = (avg ? sub->S.data() : sub->R.data()) + sub->at.at(id);
  float tot = 0;
  for (int k = 0; k < na; ++k) tot += std::max(x[k * sub->n + i], 0.0f);
  if (tot <= 0) return avg ? SIG[nd.offset + (uint64_t)b.bucket[i][nd.street] * na + a] : 1.0f / na;
  return std::max(x[a * sub->n + i], 0.0f) / tot;
}

// counterfactual values for `trav` (fold / showdown with exact card removal, as in mccfr_vec)
static void terminal(const Node& nd, int trav, const Board& b, const float* ro, float* out) {
  if (nd.kind == 1) {
    float paid = (float)(nd.folder ? nd.c1 : nd.c0);
    compatible(b, ro, out);
    float sign = nd.folder == trav ? -1.0f : 1.0f;
    for (int i = 0; i < b.n; ++i) out[i] *= sign * paid;
  } else {
    showdown(b, ro, (float)std::min(nd.c0, nd.c1), out);
  }
}

// CFR+ on the subgame. Both players' strategies are re-solved (the opponent's is the subgame equilibrium response).
static void cfr(Sub& s, int id, int trav, const Board& b, const float* rt, const float* ro, float* out, Arena& ar,
                float weight) {
  const Node& nd = T[id];
  const int n = b.n;
  if (nd.kind != 0) return terminal(nd, trav, b, ro, out);
  size_t mark = ar.top;
  const int na = (int)nd.children.size();
  float* sig = ar.get((size_t)na * n);
  for (int a = 0; a < na; ++a)
    for (int i = 0; i < n; ++i) sig[a * n + i] = strat(&s, false, b, id, i, a);
  float* child = ar.get(n);
  float* reach = ar.get(n);
  std::fill(out, out + n, 0.0f);
  float* R = s.R.data() + s.at[id];
  float* S = s.S.data() + s.at[id];
  if (nd.player == trav) {
    float* vals = ar.get((size_t)na * n);
    for (int a = 0; a < na; ++a) {
      for (int i = 0; i < n; ++i) reach[i] = rt[i] * sig[a * n + i];
      cfr(s, nd.children[a], trav, b, reach, ro, vals + (size_t)a * n, ar, weight);
      for (int i = 0; i < n; ++i) out[i] += sig[a * n + i] * vals[(size_t)a * n + i];
    }
    for (int a = 0; a < na; ++a)
      for (int i = 0; i < n; ++i) {
        R[a * n + i] = std::max(R[a * n + i] + vals[(size_t)a * n + i] - out[i], 0.0f);  // CFR+
        S[a * n + i] += weight * rt[i] * sig[a * n + i];                                  // linear averaging
      }
  } else {
    for (int a = 0; a < na; ++a) {
      for (int i = 0; i < n; ++i) reach[i] = ro[i] * sig[a * n + i];
      cfr(s, nd.children[a], trav, b, rt, reach, child, ar, weight);
      for (int i = 0; i < n; ++i) out[i] += child[i];
    }
  }
  ar.top = mark;
}

// best response of player `br` (max per hand at its nodes) to the other player's fixed strategy: blueprint
// (sub == nullptr) or the re-solve's average
static void best_response(const Sub* sub, int id, int br, const Board& b, const float* ro, float* out, Arena& ar) {
  const Node& nd = T[id];
  const int n = b.n;
  if (nd.kind != 0) return terminal(nd, br, b, ro, out);
  size_t mark = ar.top;
  const int na = (int)nd.children.size();
  float* child = ar.get(n);
  float* reach = ar.get(n);
  if (nd.player == br) {
    std::fill(out, out + n, -1e30f);
    for (int a = 0; a < na; ++a) {
      best_response(sub, nd.children[a], br, b, ro, child, ar);
      for (int i = 0; i < n; ++i) out[i] = std::max(out[i], child[i]);
    }
  } else {
    std::fill(out, out + n, 0.0f);
    for (int a = 0; a < na; ++a) {
      for (int i = 0; i < n; ++i) reach[i] = ro[i] * strat(sub, true, b, id, i, a);
      best_response(sub, nd.children[a], br, b, reach, child, ar);
      for (int i = 0; i < n; ++i) out[i] += child[i];
    }
  }
  ar.top = mark;
}

// re-solve the subgame at `root` for player `us` with ranges r_us, r_opp; safe adds the gadget with values cbv
static void solve(Sub& s, int root, int us, const Board& b, const float* r_us, const float* r_opp, const float* cbv,
                  bool safe, int iters, Arena& ar) {
  const int n = b.n, opp = 1 - us;
  build_sub(s, root, n);
  s.us = us;
  std::vector<float> g(2 * (size_t)n, 0.0f), ro(n), v(n);  // gadget regrets: enter (0) or take cbv (1)
  // safe: every opponent hand gets at least FLOOR x the largest reach, since an opponent who deviates earlier can
  // arrive with hands the blueprint never brings here; with zero weight the re-solve would ignore them
  std::vector<float> rg(r_opp, r_opp + n);
  if (safe) {
    float top = *std::max_element(rg.begin(), rg.end());
    for (auto& x : rg) x = std::max(x, FLOOR * top);
  }
  for (int t = 1; t <= iters; ++t) {
    auto enter = [&](int i) {
      float e = std::max(g[i], 0.0f), q = std::max(g[n + i], 0.0f);
      return e + q > 0 ? e / (e + q) : 0.5f;
    };
    // us: the opponent enters the subgame with their range times the gadget's probability of entering
    for (int i = 0; i < n; ++i) ro[i] = rg[i] * (safe ? enter(i) : 1.0f);
    cfr(s, root, us, b, r_us, ro.data(), v.data(), ar, (float)t);
    // the opponent: subgame values, then the gadget's choice per hand against cbv
    cfr(s, root, opp, b, rg.data(), r_us, v.data(), ar, (float)t);
    if (safe)
      for (int i = 0; i < n; ++i) {
        float e = enter(i), mix = e * v[i] + (1 - e) * cbv[i];
        g[i] = std::max(g[i] + v[i] - mix, 0.0f);
        g[n + i] = std::max(g[n + i] + cbv[i] - mix, 0.0f);
      }
  }
}

// the blueprint table for abstraction ABS from RUN/avg.f32, normalised and purified into SIG; false on error
static bool load_table(const std::string& abs, const std::string& run, double purify) {
  uint64_t total = vec_init(abs);
  if (!total) return false;
  SIG.resize(total);
  FILE* f = fopen((run + "/avg.f32").c_str(), "rb");
  if (!f || fread(SIG.data(), sizeof(float), total, f) != total) {
    fprintf(stderr, "%s/avg.f32 missing or the wrong size\n", run.c_str());
    if (f) fclose(f);
    return false;
  }
  fclose(f);
  for (auto& nd : T)
    if (nd.kind == 0) {
      int na = (int)nd.children.size();
      for (int bk = 0; bk < BUCKETS[nd.street]; ++bk) {
        float* x = &SIG[nd.offset + (uint64_t)bk * na];
        float sum = 0;
        for (int a = 0; a < na; ++a) sum += std::max(x[a], 0.0f);
        for (int a = 0; a < na; ++a) x[a] = sum > 0 ? std::max(x[a], 0.0f) / sum : 1.0f / na;
        float keep = 0;
        for (int a = 0; a < na; ++a) keep += x[a] >= purify ? x[a] : 0;
        if (keep > 0)
          for (int a = 0; a < na; ++a) x[a] = x[a] >= purify ? x[a] / keep : 0;
      }
    }
  return true;
}

// ---- library interface for blueprint_bot.py (libresolve.so, built with -DRESOLVE_LIB) ----
extern "C" int rs_init(const char* abs, const char* run, double purify, double floor) {
  if (!SIG.empty()) return 0;
  FLOOR = (float)floor;
  return load_table(abs, run, purify) ? 0 : 1;
}

// Re-solve the river subgame from `root`, the first river node (its start: the opponent's later choices on the street,
// bluffs included, are inside the solve, which is what makes the gadget safe), on board[5] (card = rank * 4 + suit),
// with each abstract player's reach at the root over the 1,326 combos (combo order: a < b over cards). `target` is
// the node we act at now (root or below it); the player there is the one protected by the gadget. out[combo * 6 + a]
// receives the re-solved average strategy at `target` for every live combo. Returns 0, or 1 on a bad spot.
extern "C" int rs_solve(int root, int target, const int* board, const float* reach0, const float* reach1, int iters,
                        int safe, float* out) {
  if (root < 0 || root >= (int)T.size() || T[root].kind != 0 || T[root].street != 3) return 1;
  if (target < root || target >= (int)T.size() || T[target].kind != 0 || T[target].street != 3) return 1;
  Board* b = new Board;
  uint8_t cards[5];
  for (int k = 0; k < 5; ++k) cards[k] = (uint8_t)board[k];
  set_board(*b, cards);
  const int n = b->n, us = T[target].player;
  std::vector<float> r[2] = {std::vector<float>(n), std::vector<float>(n)};
  for (int i = 0; i < n; ++i) r[0][i] = reach0[b->hand[i]], r[1][i] = reach1[b->hand[i]];
  Arena ar;
  ar.buf.resize((size_t)1326 * 80 * 64);
  std::vector<float> cbv(n);
  best_response(nullptr, root, 1 - us, *b, r[us].data(), cbv.data(), ar);
  Sub s;
  solve(s, root, us, *b, r[us].data(), r[1 - us].data(), cbv.data(), safe != 0, iters, ar);
  std::fill(out, out + 1326 * 6, 0.0f);
  if (!s.at.count(target)) {  // not below root
    delete b;
    return 1;
  }
  const int na = (int)T[target].children.size();
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < na; ++a) out[b->hand[i] * 6 + a] = strat(&s, true, *b, target, i, a);
  delete b;
  return 0;
}

#ifndef RESOLVE_LIB
int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: resolve ABS RUN --test SPOTS [--iters 500] [--purify 0.05] [--seed 1]\n");
    return 1;
  }
  std::string abs = argv[1], run = argv[2];
  int spots = 50, iters = 500, threads = 16;
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
  // results per spot: exploitability (opponent BR per hand pair) and worst per-hand gain over cbv
  std::vector<double> ex[3], worst[3];
  for (auto& v : ex) v.assign(spots, 0);
  for (auto& v : worst) v.assign(spots, 0);
  std::vector<int> kept(spots, 0);
  double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (int k = 0; k < spots; ++k) {
    Rng rng(seed * 1000003ULL + k);
    Board* b = new Board;
    Arena ar;
    ar.buf.resize((size_t)1326 * 80 * 64);
    // sample a river decision node by blueprint self-play on a random deal, keeping both players' reach
    int root = -1, h0 = 0, h1 = 0;
    std::vector<float> reach[2];
    for (int tries = 0; tries < 50 && root < 0; ++tries) {
      deal_board(rng, *b);
      const int n = b->n;
      reach[0].assign(n, 1.0f), reach[1].assign(n, 1.0f);
      do h0 = (int)(rng() % n), h1 = (int)(rng() % n);
      while (COMBO_A[b->hand[h0]] == COMBO_A[b->hand[h1]] || COMBO_A[b->hand[h0]] == COMBO_B[b->hand[h1]] ||
             COMBO_B[b->hand[h0]] == COMBO_A[b->hand[h1]] || COMBO_B[b->hand[h0]] == COMBO_B[b->hand[h1]]);
      int id = 0;
      while (T[id].kind == 0 && T[id].street < 3) {
        const Node& nd = T[id];
        int na = (int)nd.children.size(), p = nd.player, mine = p == 0 ? h0 : h1;
        double u = (rng() >> 11) * (1.0 / 9007199254740992.0), acc = 0;
        int pick = na - 1;
        for (int a = 0; a < na; ++a) {
          acc += SIG[nd.offset + (uint64_t)b->bucket[mine][nd.street] * na + a];
          if (u < acc) {
            pick = a;
            break;
          }
        }
        for (int i = 0; i < n; ++i) reach[p][i] *= SIG[nd.offset + (uint64_t)b->bucket[i][nd.street] * na + pick];
        id = nd.children[pick];
      }
      if (T[id].kind == 0 && T[id].street == 3) root = id;
    }
    if (root < 0) {
      delete b;
      continue;
    }
    const int n = b->n, us = T[root].player, opp = 1 - us;
    const float* r_us = reach[us].data();
    const float* r_opp = reach[opp].data();
    // per hand pair normaliser: sum over opponent hands of their reach times our compatible reach
    std::vector<float> comp(n), cbv(n), br(n);
    compatible(*b, r_us, comp.data());
    double z = 0;
    for (int i = 0; i < n; ++i) z += r_opp[i] * comp[i];
    if (z <= 0) {
      delete b;
      continue;
    }
    best_response(nullptr, root, opp, *b, r_us, cbv.data(), ar);  // the opponent's values against the blueprint
    for (int m = 0; m < 3; ++m) {
      const Sub* use = nullptr;
      Sub s;
      if (m > 0) {
        solve(s, root, us, *b, r_us, r_opp, cbv.data(), m == 2, iters, ar);
        use = &s;
      }
      // the opponent's best response to our strategy (the re-solve's average where we act)
      best_response(use, root, opp, *b, r_us, br.data(), ar);
      double e = 0, w = 0;
      for (int i = 0; i < n; ++i) {
        e += r_opp[i] * br[i];
        if (comp[i] > 0) w = std::max(w, (double)(br[i] - cbv[i]) / comp[i]);  // per hand pair, any opponent hand
      }
      ex[m][k] = e / z;
      worst[m][k] = w;
    }
    kept[k] = 1;
    delete b;
  }
  const char* name[3] = {"blueprint", "unsafe re-solve", "safe re-solve"};
  int nk = 0;
  for (int k = 0; k < spots; ++k) nk += kept[k];
  printf("%d river spots, %d CFR+ iterations, %.0fs\n", nk, iters, omp_get_wtime() - t0);
  for (int m = 0; m < 3; ++m) {
    double me = 0, mw = 0, d = 0, d2 = 0;
    for (int k = 0; k < spots; ++k)
      if (kept[k]) {
        me += ex[m][k], mw += worst[m][k];
        double x = ex[m][k] - ex[0][k];
        d += x, d2 += x * x;
      }
    me /= nk, mw /= nk, d /= nk;
    double se = std::sqrt(std::max(d2 / nk - d * d, 0.0) / std::max(nk - 1, 1));
    printf("%-16s opponent best response %+8.1f chips per hand pair  (vs blueprint %+7.1f, se %.1f)  worst hand "
           "gain over blueprint %+8.1f\n",
           name[m], me, d, se, mw);
  }
  return 0;
}
#endif
