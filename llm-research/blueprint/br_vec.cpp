// Best response to a blueprint table inside its own simplified game: how exploitable is the strategy, given the same
// buckets and bet sizes? A sharper yardstick than LBR (which plays the real game but is noisy, about +-65 bb/100):
// this one reads to within a few bb/100. It does not see abstraction or bet-mapping leaks; LBR does. Use both.
//
// The best responder decides at each (betting node, its bucket), aggregated over every board it cannot see. It is
// found street by street from the river back: pass s samples boards, plays the table for the opponent and the
// already-chosen best response on later streets, and adds up each street-s (node, bucket, action)'s counterfactual
// value; the best action is the argmax. Then an independent evaluation pass plays that best response against the
// table on fresh boards. Its value is an unbiased estimate of a lower bound on the true best response (a strategy
// chosen from noisy estimates can only be worse than the true argmax), and it tightens as --boards grows.
//
// Exploitability = (BR value as player 0 + BR value as player 1) / 2, in chips per hand = bb/100 (big blind 100).
//
// Usage: br_vec ABS RUN [--boards 2000] [--eval-boards 3000] [--threads 16] [--purify 0.05] [--out FILE]
//   RUN holds avg.f32 for this abstraction's tree (a blueprint run, or an MLP run exported by mlp_selfplay.py).
//   --purify P zeroes actions below P and renormalises, as blueprint_bot.py plays (its defence floor is not modelled).
#include <omp.h>

#include <cmath>
#include <fstream>

#include "vec.hpp"

static std::vector<float> SIG;    // the table's strategy, normalised, layout offset + bucket * na + a
static std::vector<double> Q[2];  // action values of the level being decided, from even and from odd boards
static std::vector<uint8_t> BEST; // best response action per (node, bucket), at offset + bucket * na; FOLLOW = play the table
static const uint8_t FOLLOW = 255;
// decision level of each node for each player: street * 8 + how many times that player has acted on this street.
// Within a street a later decision must be settled before an earlier one, so levels are decided deepest first.
static std::vector<int> LVL[2];

enum Mode { DECIDE, EVAL, SELF };  // SELF: the table plays both seats (the baseline a best response must beat)

// counterfactual values for BR player `br` for every live hand (zeros above the street being decided in DECIDE mode)
static void walk(int id, int br, int level, Mode mode, const Board& b, const float* ro, float* out, Arena& ar,
                 int half = 0) {
  const Node& nd = T[id];
  const int n = b.n;
  bool live = false;
  for (int i = 0; i < n && !live; ++i) live = ro[i] > 0;
  if (!live) {
    std::fill(out, out + n, 0.0f);
    return;
  }
  if (nd.kind == 1) {
    float paid = (float)(nd.folder ? nd.c1 : nd.c0);
    compatible(b, ro, out);
    float sign = nd.folder == br ? -1.0f : 1.0f;
    for (int i = 0; i < n; ++i) out[i] *= sign * paid;
    return;
  }
  if (nd.kind == 2) {
    showdown(b, ro, (float)std::min(nd.c0, nd.c1), out);
    return;
  }
  size_t mark = ar.top;
  const int na = (int)nd.children.size(), st = nd.street;
  float* child = ar.get(n);
  std::fill(out, out + n, 0.0f);
  if (mode == SELF && nd.player == br) {  // the table in the best responder's seat: values weighted by its strategy
    for (int a = 0; a < na; ++a) {
      walk(nd.children[a], br, level, mode, b, ro, child, ar, half);
      for (int i = 0; i < n; ++i) out[i] += SIG[nd.offset + (uint64_t)b.bucket[i][st] * na + a] * child[i];
    }
  } else if (nd.player != br) {  // the table plays: split the opponent's reach by its strategy
    float* reach = ar.get(n);
    for (int a = 0; a < na; ++a) {
      bool any = false;
      for (int i = 0; i < n; ++i) {
        reach[i] = ro[i] * SIG[nd.offset + (uint64_t)b.bucket[i][st] * na + a];
        any |= reach[i] > 0;
      }
      if (!any) continue;
      walk(nd.children[a], br, level, mode, b, reach, child, ar, half);
      for (int i = 0; i < n; ++i) out[i] += child[i];
    }
  } else if (mode == EVAL || LVL[br][id] > level) {  // the best response already chosen (or the table, where FOLLOW)
    float* wgt = ar.get(n);
    for (int a = 0; a < na; ++a) {
      bool any = false;
      for (int i = 0; i < n; ++i) {
        uint64_t base = nd.offset + (uint64_t)b.bucket[i][st] * na;
        wgt[i] = BEST[base] == FOLLOW ? SIG[base + a] : (BEST[base] == a ? 1.0f : 0.0f);
        any |= wgt[i] > 0;
      }
      if (!any) continue;
      walk(nd.children[a], br, level, mode, b, ro, child, ar, half);
      for (int i = 0; i < n; ++i) out[i] += wgt[i] * child[i];
    }
  } else {  // explore every action; on the level being decided, add up each action's value
    for (int a = 0; a < na; ++a) {
      walk(nd.children[a], br, level, mode, b, ro, child, ar, half);
      if (LVL[br][id] == level)
        for (int i = 0; i < n; ++i) {
          double* q = &Q[half][nd.offset + (uint64_t)b.bucket[i][st] * na + a];
#pragma omp atomic
          *q += child[i];
        }
    }
    std::fill(out, out + n, 0.0f);  // values above the decided level are not needed in this pass
  }
  ar.top = mark;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: br_vec ABS RUN [--boards N] [--eval-boards M] [--threads T] [--purify P] [--out FILE]\n");
    return 1;
  }
  std::string abs = argv[1], run = argv[2], outfile;
  int boards = 2000, eval_boards = 3000, threads = 16;
  double purify = 0;
  bool insample = false;  // debugging: decide and evaluate on the same boards (must beat the table by a wide margin)
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--boards") boards = atoi(argv[++i]);
    else if (a == "--eval-boards") eval_boards = atoi(argv[++i]);
    else if (a == "--threads") threads = atoi(argv[++i]);
    else if (a == "--purify") purify = atof(argv[++i]);
    else if (a == "--out") outfile = argv[++i];
    else if (a == "--insample") insample = true;
  }
  uint64_t total = vec_init(abs);
  if (!total) return 1;
  for (int p = 0; p < 2; ++p) {  // parents precede children in T (built depth first)
    LVL[p].assign(T.size(), 0);
    std::vector<int> acted(T.size(), 0);
    for (size_t i = 0; i < T.size(); ++i) {
      const Node& nd = T[i];
      LVL[p][i] = nd.street * 8 + acted[i];
      for (int c : nd.children)
        acted[c] = T[c].street != nd.street ? 0 : acted[i] + (nd.kind == 0 && nd.player == p ? 1 : 0);
    }
  }
  SIG.resize(total);
  {
    FILE* f = fopen((run + "/avg.f32").c_str(), "rb");
    if (!f || fread(SIG.data(), sizeof(float), total, f) != total) {
      fprintf(stderr, "%s/avg.f32 missing or not %lu floats (wrong abstraction?)\n", run.c_str(), total);
      return 1;
    }
    fclose(f);
  }
  BEST.assign(total, 0);
  for (auto& nd : T)  // normalise (and purify) the table; the best response starts by following it
    if (nd.kind == 0) {
      int na = (int)nd.children.size();
      for (int bk = 0; bk < BUCKETS[nd.street]; ++bk) {
        float* s = &SIG[nd.offset + (uint64_t)bk * na];
        float sum = 0;
        for (int a = 0; a < na; ++a) sum += s[a] > 0 ? s[a] : 0;
        for (int a = 0; a < na; ++a) s[a] = sum > 0 ? std::max(s[a], 0.0f) / sum : 1.0f / na;
        if (purify > 0) {
          float t = 0;
          for (int a = 0; a < na; ++a) t += s[a] >= purify ? s[a] : 0;
          if (t > 0)
            for (int a = 0; a < na; ++a) s[a] = s[a] >= purify ? s[a] / t : 0;
        }
        BEST[nd.offset + (uint64_t)bk * na] = FOLLOW;
      }
    }
  const size_t ARENA = (size_t)1326 * 80 * 64;
  double value[2], se[2], self[2], self_se[2];
  auto evaluate = [&](int br, Mode mode, double& m, double& sd) {  // mean value per hand over fresh boards
    std::vector<double> per(eval_boards);
#pragma omp parallel num_threads(threads)
    {
      Board* b = new Board;
      Arena ar;
      ar.buf.resize((size_t)1326 * 80 * 64);
      std::vector<float> ones(1326, 1.0f), res(1326);
#pragma omp for schedule(dynamic, 4)
      for (int k = 0; k < eval_boards; ++k) {
        Rng rng((insample ? 0x5eed0000ULL : 0xe7a10000ULL) + (uint64_t)br * 1000003 + k);
        deal_board(rng, *b);
        walk(0, br, -1, mode, *b, ones.data(), res.data(), ar);
        double s = 0;
        for (int i = 0; i < b->n; ++i) s += res[i];
        per[k] = s / b->n / 990.0;
      }
      delete b;
    }
    m = 0;
    double v = 0;
    for (double x : per) m += x;
    m /= eval_boards;
    for (double x : per) v += (x - m) * (x - m);
    sd = std::sqrt(v / (eval_boards - 1) / eval_boards);
  };
  double t0 = omp_get_wtime();
  for (int br = 0; br < 2; ++br) {
    int top = 0;
    for (size_t i = 0; i < T.size(); ++i)
      if (T[i].kind == 0 && T[i].player == br) top = std::max(top, LVL[br][i]);
    for (int level = top; level >= 0; --level) {
      bool any_node = false;
      for (size_t i = 0; i < T.size() && !any_node; ++i)
        any_node = T[i].kind == 0 && T[i].player == br && LVL[br][i] == level;
      if (!any_node) continue;
      Q[0].assign(total, 0.0);
      Q[1].assign(total, 0.0);
#pragma omp parallel num_threads(threads)
      {
        Board* b = new Board;
        Arena ar;
        ar.buf.resize(ARENA);
        std::vector<float> ones(1326, 1.0f), res(1326);
#pragma omp for schedule(dynamic, 4)
        for (int k = 0; k < boards; ++k) {
          Rng rng(0x5eed0000ULL + (uint64_t)br * 1000003 + k);  // the same boards for every level
          deal_board(rng, *b);
          walk(0, br, level, DECIDE, *b, ones.data(), res.data(), ar, k & 1);
        }
        delete b;
      }
      for (auto& nd : T)
        if (nd.kind == 0 && nd.player == br && LVL[br][&nd - &T[0]] == level) {
          int na = (int)nd.children.size();
          for (int bk = 0; bk < BUCKETS[nd.street]; ++bk) {
            // cross-validated: the best action on even boards is kept only if odd boards agree that it beats the
            // table's own mix there; otherwise follow the table. Unvalidated argmaxes overfit to the boards seen.
            uint64_t base = nd.offset + (uint64_t)bk * na;
            const double* qa = &Q[0][base];
            const double* qb = &Q[1][base];
            bool seen = false;
            int best = 0;
            double mix = 0;
            for (int a = 0; a < na; ++a) {
              seen |= qa[a] != 0 || qb[a] != 0;
              if (qa[a] > qa[best]) best = a;
              mix += SIG[base + a] * qb[a];
            }
            BEST[base] = seen && qb[best] > mix + 1e-9 ? (uint8_t)best : FOLLOW;
          }
        }
      fprintf(stderr, "player %d level %d decided (%.0fs)\n", br, level, omp_get_wtime() - t0);
    }
    // evaluation on fresh boards (the same boards for both): the best response, and the table in its seat
    evaluate(br, EVAL, value[br], se[br]);
    evaluate(br, SELF, self[br], self_se[br]);
    double m = value[br];
    fprintf(stderr, "player %d: best response %+.2f (se %.2f), the table itself %+.2f (se %.2f) bb/100, %.0fs\n", br, m,
            se[br], self[br], self_se[br], omp_get_wtime() - t0);
  }
  double ex = (value[0] + value[1]) / 2, ex_se = std::sqrt(se[0] * se[0] + se[1] * se[1]) / 2;
  printf("exploitability in the abstraction: %+.2f bb/100 [%+.2f, %+.2f]  (BR as player 0 %+.2f, as player 1 %+.2f; "
         "the table's own values %+.2f and %+.2f should sum to about 0; %d boards a street, %d to evaluate, purify "
         "%.2f)\n",
         ex, ex - 1.96 * ex_se, ex + 1.96 * ex_se, value[0], value[1], self[0], self[1], boards, eval_boards, purify);
  if (!outfile.empty())
    std::ofstream(outfile) << "{\"run\": \"" << run << "\", \"exploitability_bb100\": " << ex << ", \"ci95\": ["
                           << ex - 1.96 * ex_se << ", " << ex + 1.96 * ex_se << "], \"br_player0\": " << value[0]
                           << ", \"br_player1\": " << value[1] << ", \"boards\": " << boards
                           << ", \"eval_boards\": " << eval_boards << ", \"purify\": " << purify << "}\n";
  return 0;
}
