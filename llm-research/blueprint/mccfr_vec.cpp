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
//        [--dcfr-seconds S] [--bench SECONDS]
// --bench runs single-threaded for SECONDS and prints hand pairs per second per core, then exits.
#include <omp.h>
#include <sys/mman.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include "common.hpp"
#include "tree.hpp"

static std::vector<uint16_t> FB, TB, RB;
static Indexers* IX;
static float* R;
static float* A;
static int COMBO_A[1326], COMBO_B[1326];

struct Board {
  int n;                 // live hands (not touching the board)
  int hand[1326];        // combo index of each live hand
  int bucket[1326][4];   // per street
  int rank[1326];        // showdown rank, lower is stronger
  int order[1326];       // live positions sorted strongest first
  uint8_t card[5];
};

struct Rng {
  uint64_t s[4];
  explicit Rng(uint64_t seed) {
    for (auto& x : s) {
      seed += 0x9E3779B97F4A7C15ULL;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      x = z ^ (z >> 31);
    }
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t operator()() {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0], s[3] ^= s[1], s[1] ^= s[2], s[0] ^= s[3], s[2] ^= t, s[3] = rotl(s[3], 45);
    return r;
  }
};

static void deal_board(Rng& rng, Board& b) {
  bool used[52] = {false};
  for (int k = 0; k < 5; ++k) {
    int c;
    do c = (int)(rng() % 52);
    while (used[c]);
    used[c] = true;
    b.card[k] = (uint8_t)c;
  }
  b.n = 0;
  for (int h = 0; h < 1326; ++h) {
    int a = COMBO_A[h], c = COMBO_B[h];
    if (used[a] || used[c]) continue;
    int i = b.n++;
    b.hand[i] = h;
    uint8_t cs[7] = {(uint8_t)a, (uint8_t)c, b.card[0], b.card[1], b.card[2], b.card[3], b.card[4]};
    b.bucket[i][0] = (int)hand_index_last(&IX->pre, cs);
    b.bucket[i][1] = FB[hand_index_last(&IX->flop, cs)];
    b.bucket[i][2] = TB[hand_index_last(&IX->turn, cs)];
    b.bucket[i][3] = RB[hand_index_last(&IX->river, cs)];
    b.rank[i] = evaluate_7cards(cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6]);
  }
  for (int i = 0; i < b.n; ++i) b.order[i] = i;
  std::sort(b.order, b.order + b.n, [&](int x, int y) { return b.rank[x] < b.rank[y]; });
}

// opponent reach compatible with each of our hands (shares no card), by inclusion-exclusion
static void compatible(const Board& b, const float* opp, float* out) {
  double total = 0, per[52] = {0};
  for (int i = 0; i < b.n; ++i) {
    total += opp[i];
    per[COMBO_A[b.hand[i]]] += opp[i];
    per[COMBO_B[b.hand[i]]] += opp[i];
  }
  for (int i = 0; i < b.n; ++i)
    out[i] = (float)(total - per[COMBO_A[b.hand[i]]] - per[COMBO_B[b.hand[i]]] + opp[i]);
}

// showdown: value per our hand = paid x (compatible opponent reach we beat - reach that beats us)
static void showdown(const Board& b, const float* opp, float paid, float* out) {
  // from weakest up: reach strictly weaker than each hand
  double acc = 0, per[52] = {0};
  int i = b.n - 1;
  while (i >= 0) {
    int j = i;
    while (j >= 0 && b.rank[b.order[j]] == b.rank[b.order[i]]) --j;  // tie group (i..j+1]
    for (int k = i; k > j; --k) {
      int h = b.order[k];
      out[h] = (float)(acc - per[COMBO_A[b.hand[h]]] - per[COMBO_B[b.hand[h]]]);
    }
    for (int k = i; k > j; --k) {
      int h = b.order[k];
      acc += opp[h];
      per[COMBO_A[b.hand[h]]] += opp[h];
      per[COMBO_B[b.hand[h]]] += opp[h];
    }
    i = j;
  }
  // from strongest down: subtract reach strictly stronger
  acc = 0;
  std::fill(per, per + 52, 0.0);
  i = 0;
  while (i < b.n) {
    int j = i;
    while (j < b.n && b.rank[b.order[j]] == b.rank[b.order[i]]) ++j;
    for (int k = i; k < j; ++k) {
      int h = b.order[k];
      out[h] -= (float)(acc - per[COMBO_A[b.hand[h]]] - per[COMBO_B[b.hand[h]]]);
      out[h] *= paid;
    }
    for (int k = i; k < j; ++k) {
      int h = b.order[k];
      acc += opp[h];
      per[COMBO_A[b.hand[h]]] += opp[h];
      per[COMBO_B[b.hand[h]]] += opp[h];
    }
    i = j;
  }
}

// per-thread scratch: one block of vectors per tree depth
struct Arena {
  std::vector<float> buf;
  size_t top = 0;
  float* get(size_t n) {
    float* p = buf.data() + top;
    top += n;
    if (top > buf.size()) {
      fprintf(stderr, "arena overflow\n");
      exit(1);
    }
    return p;
  }
};

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
  float* sig = ar.get((size_t)na * n);  // sig[a*n + i]
  for (int i = 0; i < n; ++i) {
    const float* r = R + nd.offset + (uint64_t)b.bucket[i][st] * na;
    float sum = 0;
    for (int a = 0; a < na; ++a) sum += r[a] > 0 ? r[a] : 0;
    for (int a = 0; a < na; ++a) sig[a * n + i] = sum > 0 ? (r[a] > 0 ? r[a] / sum : 0) : 1.0f / na;
  }
  float* child = ar.get(n);
  float* reach = ar.get(n);
  if (nd.player == trav) {
    float* vals = ar.get((size_t)na * n);
    std::fill(out, out + n, 0.0f);
    for (int a = 0; a < na; ++a) {
      for (int i = 0; i < n; ++i) reach[i] = rt[i] * sig[a * n + i];
      cfr(nd.children[a], trav, b, reach, ro, vals + (size_t)a * n, ar);
      for (int i = 0; i < n; ++i) out[i] += sig[a * n + i] * vals[(size_t)a * n + i];
    }
    for (int i = 0; i < n; ++i) {
      float* r = R + nd.offset + (uint64_t)b.bucket[i][st] * na;
      float* av = A + nd.offset + (uint64_t)b.bucket[i][st] * na;
      for (int a = 0; a < na; ++a) {
        r[a] += vals[(size_t)a * n + i] - out[i];
        av[a] += rt[i] * sig[a * n + i];
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
  double minutes = 60, ckpt = 30, dcfr_seconds = 60, bench = 0;
  bool dcfr = false;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--threads") threads = atoi(argv[++i]);
    else if (a == "--minutes") minutes = atof(argv[++i]);
    else if (a == "--checkpoint-minutes") ckpt = atof(argv[++i]);
    else if (a == "--dcfr") dcfr = true;
    else if (a == "--dcfr-seconds") dcfr_seconds = atof(argv[++i]);
    else if (a == "--bench") bench = atof(argv[++i]);
  }
  BUCKETS[1] = BUCKETS[2] = BUCKETS[3] = bucket_count(abs);
  T.reserve(1 << 20);
  build({0, 0, SB, BB, 0, 0, BB});
  uint64_t total = 0;
  int depth = 0;
  for (auto& n : T)
    if (n.kind == 0) {
      n.offset = total;
      total += (uint64_t)BUCKETS[n.street] * n.children.size();
    }
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
  (void)depth;
  int k = 0;
  for (int a = 0; a < 52; ++a)
    for (int c = a + 1; c < 52; ++c) COMBO_A[k] = a, COMBO_B[k] = c, ++k;
  Indexers ix;
  IX = &ix;
  FB = load_buckets(abs, "flop");
  TB = load_buckets(abs, "turn");
  RB = load_buckets(abs, "river");
  if (FB.empty() || TB.empty() || RB.empty()) {
    fprintf(stderr, "bucket tables missing in %s\n", abs.c_str());
    return 1;
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
  std::atomic<bool> stop(false);
  double t0 = omp_get_wtime(), last_ckpt = t0, last_disc = t0;
  int disc_round = 0;
  FILE* log = fopen((out + "/progress.log").c_str(), "a");
#pragma omp parallel num_threads(threads)
  {
    Rng rng(std::random_device{}() ^ (uint64_t)omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
    Board* b = new Board;
    Arena ar;
    ar.buf.resize(ARENA);
    std::vector<float> ones(1326, 1.0f), res(1326);
    while (!stop) {
      deal_board(rng, *b);
      for (int trav = 0; trav < 2; ++trav) cfr(0, trav, *b, ones.data(), ones.data(), res.data(), ar);
      iters.fetch_add(2);
#pragma omp master
      {
        double now = omp_get_wtime();
        if ((now - t0) / 60 >= minutes) stop.store(true);
        if (dcfr && now - last_disc >= dcfr_seconds) {
          ++disc_round;
          double t = disc_round, pa = std::pow(t, 1.5);
          float fp = (float)(pa / (pa + 1)), fn = 0.5f, fa = (float)std::pow(t / (t + 1), 2.0);
          for (uint64_t i = 0; i < total; ++i) {
            R[i] *= R[i] > 0 ? fp : fn;
            A[i] *= fa;
          }
          last_disc = now;
        }
        if (now - last_ckpt >= ckpt * 60 || stop.load()) {
          uint64_t it = iters.load();
          if (save(out + "/regret.f32", R, total) && save(out + "/avg.f32", A, total))
            std::ofstream(out + "/iterations.txt") << it << "\n";
          fprintf(log, "%.1f min %lu vector iterations %.1f it/s\n", (now - t0) / 60, it, (it - done) / (now - t0));
          fflush(log);
          last_ckpt = now;
        }
      }
    }
  }
  fprintf(stderr, "stopped after %lu vector iterations\n", iters.load());
  return 0;
}
