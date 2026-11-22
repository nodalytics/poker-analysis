// Heads-up no-limit hold'em blueprint by external-sampling Monte Carlo CFR (the Libratus / Pluribus recipe).
//
// Game: 200 big blinds each (Slumbot's game), blinds 50/100, player 0 = small blind / button (acts first pre-flop,
// last after it). Card abstraction from `abstraction`: 169 pre-flop classes, 200 buckets on flop, turn and river.
// Action abstraction (raise sizes are raise-to amounts; any raise leaving under a third of the stack goes all-in):
//   pre-flop   open to 2, 2.5 or 3bb; vs a limp raise to 3 or 4bb; 3-bet to 3x or 4x; 4-bet to 2.3x; all-in; <= 4 raises
//   post-flop  bet 1/3, 2/3, 1 or 2 pots (2 only as the first bet); raise to pot, or 2.5x the bet (first raise);
//              all-in; at most 3 raises per street
// Each iteration deals real cards, looks up both players' buckets, updates the traverser's regrets at every one of
// its decisions and samples the opponent's actions from regret matching (adding to its average strategy). Threads
// share the tables without locks, as Pluribus did. Linear-CFR discounting for the first phase; checkpoints.
//
// Usage: mccfr ABSTRACTION_DIR OUTDIR [--dry-run] [--threads N] [--minutes M] [--checkpoint-minutes C]
//        [--discount-minutes D]
// Writes OUTDIR/tree.tsv (one line per decision node), regret.f32, avg.f32 (resumable), progress.log.
#include <omp.h>
#include <sys/mman.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"

#include "tree.hpp"

// ---------- cards ----------
// xoshiro256** (Blackman & Vigna): several times faster than mt19937_64, plenty for sampling
struct Rng {
  using result_type = uint64_t;
  uint64_t s[4];
  static constexpr uint64_t min() { return 0; }
  static constexpr uint64_t max() { return ~0ULL; }
  explicit Rng(uint64_t seed) {
    for (auto& x : s) {  // splitmix64 seeding
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
  double uniform() { return (double)((*this)() >> 11) * 0x1.0p-53; }
};

// a dealt hand; buckets and showdown ranks are computed only when the traversal reaches them (lazy)
struct Deal {
  uint8_t c[9];  // hole p0, hole p1, board 5
  int bucket[2][4];
  int rank[2];  // 7-card ranks, lower is stronger
  uint8_t have_bucket[2] = {0, 0}, have_rank = 0;
};

static std::vector<uint16_t> FB, TB, RB;
static Indexers* IX;

static bool STRATIFIED = false;  // cycle each thread through all 1,326 starting hands in shuffled order

struct Strata {
  std::vector<uint16_t> order;  // a*52+b pairs
  size_t pos = 0;
};

static void deal(Rng& rng, Deal& d, Strata* st, int which) {
  uint8_t deck[52];
  for (int i = 0; i < 52; ++i) deck[i] = (uint8_t)i;
  int start = 0;
  if (st) {
    if (st->pos >= st->order.size()) {
      for (size_t i = st->order.size() - 1; i > 0; --i) std::swap(st->order[i], st->order[rng() % (i + 1)]);
      st->pos = 0;
    }
    int pair = st->order[st->pos++], a = pair / 52, b = pair % 52;
    // put the stratified hand in player `which`'s slots, the rest of the deck stays random
    int sa = 2 * which, sb = 2 * which + 1;
    std::swap(deck[sa], deck[a]);
    int pb = -1;
    for (int i = 0; i < 52; ++i)
      if (deck[i] == b) pb = i;
    std::swap(deck[sb], deck[pb]);
    for (int i = 0; i < 9; ++i) {
      if (i == sa || i == sb) continue;
      int j;
      do j = i + (int)(rng() % (52 - i));
      while (j == sa || j == sb);
      std::swap(deck[i], deck[j]);
    }
    (void)start;
  } else {
    for (int i = 0; i < 9; ++i) {
      int j = i + (int)(rng() % (52 - i));
      std::swap(deck[i], deck[j]);
    }
  }
  for (int i = 0; i < 9; ++i) d.c[i] = deck[i];
  d.have_bucket[0] = d.have_bucket[1] = 0;
  d.have_rank = 0;
}

static inline int bucket_of(Deal& d, int p, int street) {
  if (!(d.have_bucket[p] >> street & 1)) {
    uint8_t c[7] = {d.c[2 * p], d.c[2 * p + 1], d.c[4], d.c[5], d.c[6], d.c[7], d.c[8]};
    int b = street == 0   ? (int)hand_index_last(&IX->pre, c)
            : street == 1 ? FB[hand_index_last(&IX->flop, c)]
            : street == 2 ? TB[hand_index_last(&IX->turn, c)]
                          : RB[hand_index_last(&IX->river, c)];
    d.bucket[p][street] = b;
    d.have_bucket[p] |= (uint8_t)(1 << street);
  }
  return d.bucket[p][street];
}

static inline void ranks_of(Deal& d) {
  if (!d.have_rank) {
    for (int p = 0; p < 2; ++p)
      d.rank[p] = evaluate_7cards(d.c[2 * p], d.c[2 * p + 1], d.c[4], d.c[5], d.c[6], d.c[7], d.c[8]);
    d.have_rank = 1;
  }
}

// ---------- CFR ----------
// pruning (Pluribus): after a warm-up, 95% of the time skip the traverser's actions whose regret is below -PRUNE_C
static std::atomic<bool> PRUNE_ON(false);
static float PRUNE_C = 1e6f;
static bool PCFR = false;  // predictive CFR+ (Farina, Kroer & Sandholm 2021): strategy from max(R + last regret, 0)
static float* P = nullptr;  // PCFR+: the last instantaneous regret (the prediction)
static float* R;  // regrets
static float* A;  // average strategy sums

static inline void strategy(const float* r, int n, float* s) {
  float sum = 0;
  for (int a = 0; a < n; ++a) sum += r[a] > 0 ? r[a] : 0;
  for (int a = 0; a < n; ++a) s[a] = sum > 0 ? (r[a] > 0 ? r[a] / sum : 0) : 1.0f / n;
}

static float traverse(int id, int trav, Deal& d, Rng& rng) {
  const Node& n = T[id];
  if (n.kind == 1) {
    int loser_paid = n.folder ? n.c1 : n.c0;
    return n.folder == trav ? -(float)loser_paid : (float)loser_paid;
  }
  if (n.kind == 2) {
    ranks_of(d);
    int paid = std::min(n.c0, n.c1);  // the smaller stack is the most either can win
    if (d.rank[0] == d.rank[1]) return 0;
    bool trav_wins = d.rank[trav] < d.rank[1 - trav];
    return trav_wins ? (float)paid : -(float)paid;
  }
  int na = (int)n.children.size();
  uint64_t base = n.offset + (uint64_t)bucket_of(d, n.player, n.street) * na;
  float s[10];
  if (PCFR) {
    float q[10];
    for (int a = 0; a < na; ++a) q[a] = R[base + a] + P[base + a];
    strategy(q, na, s);
  } else {
    strategy(R + base, na, s);
  }
  if (n.player == trav) {
    float v[10], ev = 0;
    bool skip[10] = {false};
    bool prune = PRUNE_ON.load(std::memory_order_relaxed) && n.street < 3 && (rng() % 100) < 95;
    for (int a = 0; a < na; ++a) {
      // a pruned action has negative regret, so regret matching gives it probability 0: skipping it leaves ev exact
      if (prune && R[base + a] < -PRUNE_C) {
        skip[a] = true;
        v[a] = 0;
        continue;
      }
      v[a] = traverse(n.children[a], trav, d, rng);
      ev += s[a] * v[a];
    }
    if (PCFR) {
      for (int a = 0; a < na; ++a) {
        float inst = v[a] - ev;
        float r = R[base + a] + inst;
        R[base + a] = r > 0 ? r : 0;  // CFR+ floor
        P[base + a] = inst;
      }
      return ev;
    }
    for (int a = 0; a < na; ++a)
      if (!skip[a]) R[base + a] += v[a] - ev;
    return ev;
  }
  for (int a = 0; a < na; ++a) A[base + a] += s[a];
  double u = rng.uniform(), acc = 0;
  int pick = na - 1;
  for (int a = 0; a < na; ++a) {
    acc += s[a];
    if (u < acc) {
      pick = a;
      break;
    }
  }
  return traverse(n.children[pick], trav, d, rng);
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
    fprintf(stderr, "usage: mccfr ABSTRACTION_DIR OUTDIR [--dry-run] [--threads N] [--minutes M] ...\n");
    return 1;
  }
  std::string abs = argv[1], out = argv[2];
  bool dry = false;
  int threads = 48;
  double minutes = 60, ckpt = 30, discount_minutes = 400;
  bool dcfr = false;            // discounted CFR (Brown & Sandholm 2019): alpha 1.5, beta 0, gamma 2 per period
  double dcfr_seconds = 60;     // one discounting period
  double prune_after = -1;      // minutes before pruning starts (-1: never)
  std::string init_from, init_abs;  // multigrid start: a solved run on a coarser abstraction, and that abstraction
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--dry-run") dry = true;
    else if (a == "--threads") threads = atoi(argv[++i]);
    else if (a == "--minutes") minutes = atof(argv[++i]);
    else if (a == "--checkpoint-minutes") ckpt = atof(argv[++i]);
    else if (a == "--discount-minutes") discount_minutes = atof(argv[++i]);
    else if (a == "--dcfr") dcfr = true;
    else if (a == "--dcfr-seconds") dcfr_seconds = atof(argv[++i]);
    else if (a == "--prune-after-minutes") prune_after = atof(argv[++i]);
    else if (a == "--prune-threshold") PRUNE_C = (float)atof(argv[++i]);
    else if (a == "--pcfr") PCFR = true;
    else if (a == "--stratified") STRATIFIED = true;
    else if (a == "--init-from") init_from = argv[++i];
    else if (a == "--init-abs") init_abs = argv[++i];
  }
  BUCKETS[1] = BUCKETS[2] = BUCKETS[3] = bucket_count(abs);
  T.reserve(1 << 20);
  build({0, 0, SB, BB, 0, 0, BB});
  uint64_t total = 0, decisions = 0;
  for (auto& n : T)
    if (n.kind == 0) {
      n.offset = total;
      total += (uint64_t)BUCKETS[n.street] * n.children.size();
      ++decisions;
    }
  fprintf(stderr, "tree: %zu nodes, %lu decision nodes, %lu regret entries, %.2f GB for regrets + averages\n",
          T.size(), decisions, total, 2.0 * total * 4 / 1e9);
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
  if (dry) return 0;
  Indexers ix;
  IX = &ix;
  FB = load_buckets(abs, "flop");
  TB = load_buckets(abs, "turn");
  RB = load_buckets(abs, "river");
  if (FB.empty() || TB.empty() || RB.empty()) {
    fprintf(stderr, "bucket tables missing in %s\n", abs.c_str());
    return 1;
  }
  // 2 MB pages for the big tables: far fewer TLB misses on random access (transparent huge pages)
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
  if (PCFR) P = big(total);
  if (!R || !A || (PCFR && !P)) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  if (PCFR) prune_after = -1;  // CFR+ floors regrets at 0, so nothing would ever be pruned
  uint64_t done = 0;
  {
    std::ifstream pf(out + "/iterations.txt");
    if (pf && load(out + "/regret.f32", R, total) && load(out + "/avg.f32", A, total)) {
      pf >> done;
      fprintf(stderr, "resumed at %lu iterations\n", done);
    }
  }
  if (!done && !init_from.empty()) {
    // multigrid: map each fine bucket to the coarse bucket most of its states fall in, then copy the coarse run's
    // regrets and average strategy into the fine tables (same betting tree, different card abstraction)
    int CB = bucket_count(init_abs);
    std::vector<uint16_t> cf = load_buckets(init_abs, "flop"), ct = load_buckets(init_abs, "turn"),
                          cr = load_buckets(init_abs, "river");
    std::vector<std::vector<int>> map(4);
    const std::vector<uint16_t>* fine[4] = {nullptr, &FB, &TB, &RB};
    const std::vector<uint16_t>* coarse[4] = {nullptr, &cf, &ct, &cr};
    map[0].resize(169);
    for (int b = 0; b < 169; ++b) map[0][b] = b;
    Rng mr(5);
    for (int s = 1; s < 4; ++s) {
      std::vector<uint32_t> co((size_t)BUCKETS[s] * CB, 0);
      for (int k = 0; k < 4000000; ++k) {
        uint64_t i = mr() % fine[s]->size();
        ++co[(size_t)(*fine[s])[i] * CB + (*coarse[s])[i]];
      }
      map[s].resize(BUCKETS[s]);
      for (int b = 0; b < BUCKETS[s]; ++b) {
        int best = 0;
        for (int c = 1; c < CB; ++c)
          if (co[(size_t)b * CB + c] > co[(size_t)b * CB + best]) best = c;
        map[s][b] = best;
      }
    }
    uint64_t ctotal = 0;
    std::vector<uint64_t> coff(T.size(), 0);
    for (size_t i = 0; i < T.size(); ++i)
      if (T[i].kind == 0) {
        coff[i] = ctotal;
        ctotal += (uint64_t)(T[i].street ? CB : 169) * T[i].children.size();
      }
    std::vector<float> CR(ctotal), CA(ctotal);
    if (!load(init_from + "/regret.f32", CR.data(), ctotal) || !load(init_from + "/avg.f32", CA.data(), ctotal)) {
      fprintf(stderr, "cannot read the coarse run %s (sizes must match its tree)\n", init_from.c_str());
      return 1;
    }
    for (size_t i = 0; i < T.size(); ++i)
      if (T[i].kind == 0) {
        int na = (int)T[i].children.size(), s = T[i].street;
        for (int b = 0; b < BUCKETS[s]; ++b)
          for (int a = 0; a < na; ++a) {
            R[T[i].offset + (uint64_t)b * na + a] = CR[coff[i] + (uint64_t)map[s][b] * na + a];
            A[T[i].offset + (uint64_t)b * na + a] = CA[coff[i] + (uint64_t)map[s][b] * na + a];
          }
      }
    fprintf(stderr, "multigrid start from %s (%d coarse buckets)\n", init_from.c_str(), CB);
  }
  std::atomic<uint64_t> iters(done);
  double t0 = omp_get_wtime(), last_ckpt = t0, last_disc = t0;
  int disc_round = 0;
  FILE* log = fopen((out + "/progress.log").c_str(), "a");
  std::atomic<bool> stop(false);
#pragma omp parallel num_threads(threads)
  {
    Rng rng(std::random_device{}() ^ (uint64_t)omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
    Deal d;
    Strata st;
    if (STRATIFIED) {
      for (int a = 0; a < 52; ++a)
        for (int b = a + 1; b < 52; ++b) st.order.push_back((uint16_t)(a * 52 + b));
      st.pos = st.order.size();  // shuffled on first use
    }
    uint64_t local = 0;
    while (!stop) {
      for (int k = 0; k < 500; ++k) {
        // one deal, both players traverse it (two iterations; the deal's bucket lookups are shared)
        deal(rng, d, STRATIFIED ? &st : nullptr, (int)(local & 1));
        traverse(0, 0, d, rng);
        traverse(0, 1, d, rng);
        ++local;
      }
      iters.fetch_add(1000);
#pragma omp master
      {
        double now = omp_get_wtime();
        if ((now - t0) / 60 >= minutes) stop.store(true);
        if (prune_after >= 0 && (now - t0) / 60 >= prune_after && !PRUNE_ON.load()) {
          PRUNE_ON.store(true);
          fprintf(log, "%.1f min pruning on\n", (now - t0) / 60);
        }
        if (PCFR && now - last_disc >= dcfr_seconds) {
          // PCFR+: quadratic averaging of the strategy (weights t^2), regrets are already floored at 0
          ++disc_round;
          double t = disc_round;
          float fa = (float)std::pow(t / (t + 1), 2.0);
#pragma omp parallel for num_threads(8) schedule(static)
          for (uint64_t i = 0; i < total; ++i) A[i] *= fa;
          last_disc = now;
        }
        if (!PCFR && dcfr && now - last_disc >= dcfr_seconds) {
          // DCFR, one period = dcfr_seconds of sampling: positive regrets x t^1.5/(t^1.5+1), negative x 1/2,
          // average strategy x (t/(t+1))^2
          ++disc_round;
          double t = disc_round, pa = std::pow(t, 1.5);
          float fp = (float)(pa / (pa + 1)), fn = 0.5f, fa = (float)std::pow(t / (t + 1), 2.0);
#pragma omp parallel for num_threads(8) schedule(static)
          for (uint64_t i = 0; i < total; ++i) {
            R[i] *= R[i] > 0 ? fp : fn;
            A[i] *= fa;
          }
          last_disc = now;
        }
        // linear-CFR style discounting (Pluribus: every 10 minutes for the first 400) - approximate, done in place
        if (!PCFR && !dcfr && (now - t0) / 60 < discount_minutes && now - last_disc >= 600) {
          ++disc_round;
          float f = (float)disc_round / (disc_round + 1);
          for (uint64_t i = 0; i < total; ++i) R[i] *= f, A[i] *= f;
          last_disc = now;
        }
        if (now - last_ckpt >= ckpt * 60 || stop.load()) {
          uint64_t it = iters.load();
          if (save(out + "/regret.f32", R, total) && save(out + "/avg.f32", A, total)) {
            std::ofstream(out + "/iterations.txt") << it << "\n";
          }
          fprintf(log, "%.1f min %lu iterations %.0f it/s\n", (now - t0) / 60, it, (it - done) / (now - t0));
          fflush(log);
          last_ckpt = now;
        }
      }
    }
  }
  fprintf(stderr, "stopped after %lu iterations\n", iters.load());
  return 0;
}
