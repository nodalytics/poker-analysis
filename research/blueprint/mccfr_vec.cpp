// Vector Monte Carlo CFR for the blueprint (public chance sampling, Johanson et al. 2012; the vector form solvers such
// as TexasSolver use): each iteration deals one board, then updates EVERY hand of both players at once through the
// whole betting tree, as tight array arithmetic. One iteration carries about 1,081 x 1,081 hand pairs, against one
// pair for external sampling. Same tree (tree.hpp), same card abstraction and table layout as mccfr.cpp, so runs are
// read by blueprint_bot.py and inspect_blueprint.py unchanged.
//
// Terminal values with card removal: folds by inclusion-exclusion over each hand's two cards; showdowns by one sorted
// pass from the weakest hand up and one from the strongest down, keeping per-card sums.
//
// Usage: mccfr_vec ABSTRACTION_DIR OUTDIR [--threads N] [--minutes M] [--checkpoint-minutes C] [--dcfr]
//        [--dcfr-seconds S] [--prune-after-minutes P] [--prune-threshold C] [--bench SECONDS]
// --bench runs single-threaded for SECONDS and prints hand pairs per second per core, then exits.
#include <omp.h>
#include <sys/mman.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include "vec.hpp"

static float* R;
static float* A;
// regret-based pruning (as mccfr.cpp): after a warm-up, a traverser action whose regret is below -PRUNE_C in every
// bucket on the board has probability 0 for every hand; 95% of the time its subtree is skipped and its regrets left
static std::atomic<bool> PRUNE_ON(false);
static float PRUNE_C = 1e5f;

// returns the traverser's counterfactual values in `out` for every live hand
static void cfr(int id, int trav, const Board& b, const float* rt, const float* ro, float* out, Arena& ar) {
  const Node& nd = T[id];
  const int n = b.n;
  // the opponent never reaches this node with any hand: every value is 0. External sampling never visits such a
  // node either, so skipping it keeps the two trainers' maths the same.
  bool live = false;
  for (int i = 0; i < n && !live; ++i) live = ro[i] > 0;
  if (!live) {
    std::fill(out, out + n, 0.0f);
    return;
  }
  if (nd.kind == 1) {
    float paid = (float)(nd.folder ? nd.c1 : nd.c0);
    compatible(b, ro, out);
    float sign = nd.folder == trav ? -1.0f : 1.0f;
    for (int i = 0; i < n; ++i) out[i] *= sign * paid;
    return;
  }
  if (nd.kind == 2) {
    showdown(b, ro, (float)std::min(nd.c0, nd.c1), out);
    return;
  }
  size_t mark = ar.top;
  const int na = (int)nd.children.size(), st = nd.street;
  // regret matching once per distinct bucket on this board, then gathered per hand: sig[a*n + i]
  const int nu = b.nub[st];
  float* sigu = ar.get((size_t)nu * na);
  for (int u = 0; u < nu; ++u) {
    const float* r = R + nd.offset + (uint64_t)b.ub[st][u] * na;
    float sum = 0;
    for (int a = 0; a < na; ++a) sum += r[a] > 0 ? r[a] : 0;
    for (int a = 0; a < na; ++a) sigu[u * na + a] = sum > 0 ? (r[a] > 0 ? r[a] / sum : 0) : 1.0f / na;
  }
  float* sig = ar.get((size_t)na * n);
  for (int i = 0; i < n; ++i) {
    const float* su = sigu + (size_t)b.uidx[i][st] * na;
    for (int a = 0; a < na; ++a) sig[a * n + i] = su[a];
  }
  float* child = ar.get(n);
  float* reach = ar.get(n);
  if (nd.player == trav) {
    float* vals = ar.get((size_t)na * n);
    std::fill(out, out + n, 0.0f);
    bool skip[16] = {false};
    static thread_local uint64_t prng = 0x9E3779B97F4A7C15ULL ^ (uint64_t)omp_get_thread_num();
    const bool prune = PRUNE_ON.load(std::memory_order_relaxed) && st < 3 &&
                       ((prng = prng * 6364136223846793005ULL + 1442695040888963407ULL) >> 33) % 100 < 95;
    for (int a = 0; a < na; ++a) {
      if (prune) {
        bool all_low = true;
        for (int u = 0; u < nu && all_low; ++u) all_low = R[nd.offset + (uint64_t)b.ub[st][u] * na + a] < -PRUNE_C;
        if (all_low) {  // probability 0 for every hand: out is unchanged, the regrets stay as they are
          skip[a] = true;
          continue;
        }
      }
      for (int i = 0; i < n; ++i) reach[i] = rt[i] * sig[a * n + i];
      cfr(nd.children[a], trav, b, reach, ro, vals + (size_t)a * n, ar);
      for (int i = 0; i < n; ++i) out[i] += sig[a * n + i] * vals[(size_t)a * n + i];
    }
    // sum each bucket's regret and average-strategy increments over its hands, then one write per bucket
    float* dr = ar.get((size_t)nu * na);
    float* da = ar.get((size_t)nu * na);
    std::fill(dr, dr + (size_t)nu * na, 0.0f);
    std::fill(da, da + (size_t)nu * na, 0.0f);
    for (int i = 0; i < n; ++i) {
      float* r = dr + (size_t)b.uidx[i][st] * na;
      float* av = da + (size_t)b.uidx[i][st] * na;
      for (int a = 0; a < na; ++a) {
        if (!skip[a]) r[a] += vals[(size_t)a * n + i] - out[i];
        av[a] += rt[i] * sig[a * n + i];
      }
    }
    for (int u = 0; u < nu; ++u) {
      float* r = R + nd.offset + (uint64_t)b.ub[st][u] * na;
      float* av = A + nd.offset + (uint64_t)b.ub[st][u] * na;
      for (int a = 0; a < na; ++a) {
        r[a] += dr[u * na + a];
        av[a] += da[u * na + a];
      }
    }
  } else {
    std::fill(out, out + n, 0.0f);
    for (int a = 0; a < na; ++a) {
      for (int i = 0; i < n; ++i) reach[i] = ro[i] * sig[a * n + i];
      cfr(nd.children[a], trav, b, rt, reach, child, ar);
      for (int i = 0; i < n; ++i) out[i] += child[i];
    }
  }
  ar.top = mark;
}

static bool save(const std::string& p, const float* x, uint64_t n) {
  std::string tmp = p + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  bool ok = fwrite(x, sizeof(float), n, f) == n;
  fclose(f);
  return ok && rename(tmp.c_str(), p.c_str()) == 0;
}

static bool load(const std::string& p, float* x, uint64_t n) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return false;
  bool ok = fread(x, sizeof(float), n, f) == n;
  fclose(f);
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: mccfr_vec ABSTRACTION_DIR OUTDIR [options]\n");
    return 1;
  }
  std::string abs = argv[1], out = argv[2];
  int threads = 48;
  double minutes = 60, ckpt = 30, dcfr_seconds = 60, bench = 0, prune_after = -1;
  bool dcfr = false;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--threads") threads = atoi(argv[++i]);
    else if (a == "--minutes") minutes = atof(argv[++i]);
    else if (a == "--checkpoint-minutes") ckpt = atof(argv[++i]);
    else if (a == "--dcfr") dcfr = true;
    else if (a == "--dcfr-seconds") dcfr_seconds = atof(argv[++i]);
    else if (a == "--bench") bench = atof(argv[++i]);
    else if (a == "--prune-after-minutes") prune_after = atof(argv[++i]);
    else if (a == "--prune-threshold") PRUNE_C = (float)atof(argv[++i]);
  }
  uint64_t total = vec_init(abs);
  if (!total) return 1;
  {
    std::ofstream t(out + "/tree.tsv");
    t << "id\tkind\tplayer\tstreet\tc0\tc1\toffset\tlabels\tchildren\n";
    for (size_t i = 0; i < T.size(); ++i) {
      auto& n = T[i];
      t << i << '\t' << (int)n.kind << '\t' << (int)n.player << '\t' << (int)n.street << '\t' << n.c0 << '\t' << n.c1
        << '\t' << n.offset << '\t';
      for (size_t k = 0; k < n.labels.size(); ++k) t << (k ? "," : "") << n.labels[k];
      t << '\t';
      for (size_t k = 0; k < n.children.size(); ++k) t << (k ? "," : "") << n.children[k];
      t << '\n';
    }
  }
  auto big = [](uint64_t n) {
    size_t bytes = ((n * sizeof(float) + (2u << 20) - 1) / (2u << 20)) * (2u << 20);
    float* x = (float*)aligned_alloc(2u << 20, bytes);
    if (x) {
      madvise(x, bytes, MADV_HUGEPAGE);
      memset(x, 0, bytes);
    }
    return x;
  };
  R = big(total);
  A = big(total);
  if (!R || !A) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  uint64_t done = 0;
  {
    std::ifstream pf(out + "/iterations.txt");
    if (pf && load(out + "/regret.f32", R, total) && load(out + "/avg.f32", A, total)) pf >> done;
  }
  const size_t ARENA = (size_t)1326 * 80 * 64;  // vectors per depth x depth, generous
  if (bench > 0) {
    if (prune_after >= 0) PRUNE_ON.store(true);  // benchmark pruning from a loaded checkpoint
    Rng rng(1);
    Board* b = new Board;
    Arena ar;
    ar.buf.resize(ARENA);
    std::vector<float> ones(1326, 1.0f), res(1326);
    {  // self-test: fold and showdown values against an O(n^2) brute force on random reaches
      deal_board(rng, *b);
      std::vector<float> opp(b->n), f(b->n), s(b->n);
      for (int i = 0; i < b->n; ++i) opp[i] = (float)(rng() % 1000) / 1000.0f;
      compatible(*b, opp.data(), f.data());
      showdown(*b, opp.data(), 3.0f, s.data());
      double ef = 0, es = 0;
      for (int i = 0; i < b->n; ++i) {
        double bf = 0, bs = 0;
        int a1 = COMBO_A[b->hand[i]], a2 = COMBO_B[b->hand[i]];
        for (int j = 0; j < b->n; ++j) {
          int c1 = COMBO_A[b->hand[j]], c2 = COMBO_B[b->hand[j]];
          if (c1 == a1 || c1 == a2 || c2 == a1 || c2 == a2) continue;
          bf += opp[j];
          bs += opp[j] * (b->rank[i] < b->rank[j] ? 3 : b->rank[i] > b->rank[j] ? -3 : 0);
        }
        ef = std::max(ef, std::fabs(bf - f[i]));
        es = std::max(es, std::fabs(bs - s[i]));
      }
      printf("self-test: max error fold %.4f showdown %.4f (%d live hands)\n", ef, es, b->n);
      if (ef > 0.05 || es > 0.05) return 1;
    }
    double t0 = omp_get_wtime();
    uint64_t pairs = 0, its = 0;
    while (omp_get_wtime() - t0 < bench) {
      deal_board(rng, *b);
      for (int trav = 0; trav < 2; ++trav) {
        cfr(0, trav, *b, ones.data(), ones.data(), res.data(), ar);
        pairs += (uint64_t)b->n * b->n;
        ++its;
      }
    }
    double s = omp_get_wtime() - t0;
    printf("single core: %.2f vector iterations/s, %.0f hand pairs/s (%.1fM), %lu iterations in %.1fs\n", its / s,
           pairs / s, pairs / s / 1e6, its, s);
    return 0;
  }
  std::atomic<uint64_t> iters(done);
  // Workers update R and A lock-free (Hogwild, as before). The discount and the checkpoint rescale or read every entry,
  // so they run only while every worker is parked: the master raises `sync`, each thread finishes its iteration and
  // meets the others at a barrier, the threads discount their own slices, the master saves, and all resume. Normal
  // iterations pay one relaxed load of the flag.
  std::atomic<bool> sync(false), stop(false);
  bool do_disc = false, do_ckpt = false;  // set by the master before it raises `sync`
  double t0 = omp_get_wtime(), last_ckpt = t0, last_disc = t0;
  int disc_round = 0;
  float fp = 1, fn = 1, fa = 1;
  FILE* log = fopen((out + "/progress.log").c_str(), "a");
#pragma omp parallel num_threads(threads)
  {
    const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
    Rng rng(std::random_device{}() ^ (uint64_t)tid * 0x9E3779B97F4A7C15ULL);
    Board* b = new Board;
    Arena ar;
    ar.buf.resize(ARENA);
    std::vector<float> ones(1326, 1.0f), res(1326);
    for (;;) {
      if (sync.load(std::memory_order_acquire)) {
#pragma omp barrier
        if (do_disc) {  // every thread rescales its own slice: no worker is writing
          uint64_t lo = total * (uint64_t)tid / nt, hi = total * (uint64_t)(tid + 1) / nt;
          for (uint64_t i = lo; i < hi; ++i) {
            R[i] *= R[i] > 0 ? fp : fn;
            A[i] *= fa;
          }
        }
#pragma omp barrier
        if (tid == 0) {
          double now = omp_get_wtime();
          if (do_ckpt) {  // a consistent snapshot: the tables and the iteration count of the same moment
            uint64_t it = iters.load();
            if (save(out + "/regret.f32", R, total) && save(out + "/avg.f32", A, total))
              std::ofstream(out + "/iterations.txt") << it << "\n";
            fprintf(log, "%.1f min %lu vector iterations %.1f it/s\n", (now - t0) / 60, it, (it - done) / (now - t0));
            fflush(log);
          }
          do_disc = do_ckpt = false;
          sync.store(false, std::memory_order_release);
        }
#pragma omp barrier
        if (stop.load()) break;
        continue;
      }
      deal_board(rng, *b);
      for (int trav = 0; trav < 2; ++trav) cfr(0, trav, *b, ones.data(), ones.data(), res.data(), ar);
      iters.fetch_add(2);
      if (tid == 0) {
        double now = omp_get_wtime();
        if ((now - t0) / 60 >= minutes) stop.store(true);
        if (prune_after >= 0 && (now - t0) / 60 >= prune_after) PRUNE_ON.store(true);
        if (dcfr && now - last_disc >= dcfr_seconds) {
          ++disc_round;
          double t = disc_round, pa = std::pow(t, 1.5);
          fp = (float)(pa / (pa + 1)), fn = 0.5f, fa = (float)std::pow(t / (t + 1), 2.0);
          do_disc = true;
          last_disc = now;
        }
        if (now - last_ckpt >= ckpt * 60 || stop.load()) {
          do_ckpt = true;
          last_ckpt = now;
        }
        if (do_disc || do_ckpt) sync.store(true, std::memory_order_release);
      }
    }
  }
  fprintf(stderr, "stopped after %lu vector iterations\n", iters.load());
  return 0;
}
