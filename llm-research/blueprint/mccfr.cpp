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

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"

static const int STACK = 20000, SB = 50, BB = 100;
static int BUCKETS[4] = {169, 200, 200, 200};  // flop..river set from the abstraction's meta.txt

struct Node {
  int8_t kind = 0;  // 0 decision, 1 fold (player `folder` folded), 2 showdown
  int8_t player = 0, street = 0, folder = -1;
  int c0 = 0, c1 = 0;  // chips put in by player 0 and 1
  std::vector<std::string> labels;
  std::vector<int> children;
  uint64_t offset = 0;  // into the regret / average arrays
};

static std::vector<Node> T;

struct Bet {
  int street, player, c0, c1, raises, acted;  // acted: decisions taken this street
  int last_raise;                             // size of the last raise increment
};

static int add_node(const Node& n) {
  T.push_back(n);
  return (int)T.size() - 1;
}

static int build(Bet b);

static int next_street(int c0, int c1, int street) {
  if (street == 3 || c0 >= STACK || c1 >= STACK) {
    Node n;
    n.kind = 2;
    n.street = street;
    n.c0 = c0, n.c1 = c1;
    return add_node(n);
  }
  return build({street + 1, 1, c0, c1, 0, 0, BB});
}

static int build(Bet b) {
  Node n;
  n.player = b.player;
  n.street = b.street;
  n.c0 = b.c0, n.c1 = b.c1;
  int id = add_node(n);
  int mine = b.player ? b.c1 : b.c0, theirs = b.player ? b.c0 : b.c1;
  int to_call = theirs - mine, pot = b.c0 + b.c1;
  std::vector<std::pair<std::string, int>> acts;  // label, raise-to (or -1 fold, -2 check/call)
  if (to_call > 0) acts.push_back({"fold", -1});
  acts.push_back({to_call > 0 ? "call" : "check", -2});
  int cap = b.street == 0 ? 4 : 3;
  if (b.raises < cap && theirs < STACK) {
    std::vector<std::pair<std::string, int>> sizes;
    if (b.street == 0) {
      if (b.raises == 0 && b.player == 0) {
        sizes.push_back({"open2", 200});
        sizes.push_back({"open2.5", 250});
        sizes.push_back({"open3", 300});
      } else if (b.raises == 0) {
        sizes.push_back({"raise3", 300});
        sizes.push_back({"raise4", 400});
      } else if (b.raises == 1) {
        sizes.push_back({"3bet3x", 3 * theirs});
        sizes.push_back({"3bet4x", 4 * theirs});
      } else if (b.raises == 2) {
        sizes.push_back({"4bet2.3x", (int)(2.3 * theirs)});
      }
    } else if (to_call == 0) {
      sizes.push_back({"bet0.33", mine + pot / 3});
      sizes.push_back({"bet0.66", mine + 2 * pot / 3});
      sizes.push_back({"bet1", mine + pot});
      if (b.raises == 0) sizes.push_back({"bet2", mine + 2 * pot});
    } else {
      sizes.push_back({"raise1", theirs + (pot + to_call)});
      if (b.raises == 1) sizes.push_back({"raise2.5x", theirs + (int)(1.5 * to_call) + to_call});
    }
    int min_to = theirs + std::max(b.last_raise, BB);
    for (auto& s : sizes) {
      int to = std::max(s.second, min_to);
      if (to >= STACK - STACK / 3 || to >= STACK) continue;  // close to all-in: use all-in instead
      acts.push_back({s.first, to});
    }
    acts.push_back({"allin", STACK});
  }
  std::vector<int> kids;
  std::vector<std::string> labels;
  for (auto& a : acts) {
    int child;
    if (a.second == -1) {
      Node f;
      f.kind = 1;
      f.folder = b.player;
      f.street = b.street;
      f.c0 = b.c0, f.c1 = b.c1;
      child = add_node(f);
    } else if (a.second == -2) {
      int c0 = b.c0, c1 = b.c1;
      if (b.player) c1 = theirs; else c0 = theirs;
      // the street closes on a call, or on a check by the second player to act (pre-flop: the big blind)
      bool closes = (to_call > 0 && !(b.street == 0 && b.raises == 0 && b.player == 0)) || (to_call == 0 && b.acted >= 1);
      if (b.street == 0 && b.raises == 0 && b.player == 0 && to_call > 0)
        child = build({0, 1, c0, c1, 0, b.acted + 1, b.last_raise});  // the small blind limps: the big blind acts
      else if (closes)
        child = next_street(c0, c1, b.street);
      else
        child = build({b.street, 1 - b.player, c0, c1, b.raises, b.acted + 1, b.last_raise});
    } else {
      int to = std::min(a.second, STACK);
      int c0 = b.c0, c1 = b.c1;
      if (b.player) c1 = to; else c0 = to;
      child = build({b.street, 1 - b.player, c0, c1, b.raises + 1, b.acted + 1, std::max(to - theirs, b.last_raise)});
    }
    kids.push_back(child);
    labels.push_back(a.first);
  }
  T[id].children = kids;
  T[id].labels = labels;
  return id;
}

// ---------- cards ----------
struct Deal {
  int bucket[2][4];
  int rank[2];  // 7-card ranks, lower is stronger
};

static std::vector<uint16_t> FB, TB, RB;
static Indexers* IX;

static void deal(std::mt19937_64& rng, Deal& d) {
  uint8_t deck[52];
  for (int i = 0; i < 52; ++i) deck[i] = (uint8_t)i;
  for (int i = 0; i < 9; ++i) {
    int j = i + (int)(rng() % (52 - i));
    std::swap(deck[i], deck[j]);
  }
  for (int p = 0; p < 2; ++p) {
    uint8_t c[7] = {deck[2 * p], deck[2 * p + 1], deck[4], deck[5], deck[6], deck[7], deck[8]};
    d.bucket[p][0] = (int)hand_index_last(&IX->pre, c);
    d.bucket[p][1] = FB[hand_index_last(&IX->flop, c)];
    d.bucket[p][2] = TB[hand_index_last(&IX->turn, c)];
    d.bucket[p][3] = RB[hand_index_last(&IX->river, c)];
    d.rank[p] = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
  }
}

// ---------- CFR ----------
static float* R;  // regrets
static float* A;  // average strategy sums

static inline void strategy(const float* r, int n, float* s) {
  float sum = 0;
  for (int a = 0; a < n; ++a) sum += r[a] > 0 ? r[a] : 0;
  for (int a = 0; a < n; ++a) s[a] = sum > 0 ? (r[a] > 0 ? r[a] / sum : 0) : 1.0f / n;
}

static float traverse(int id, int trav, const Deal& d, std::mt19937_64& rng) {
  const Node& n = T[id];
  if (n.kind == 1) {
    int loser_paid = n.folder ? n.c1 : n.c0;
    return n.folder == trav ? -(float)loser_paid : (float)loser_paid;
  }
  if (n.kind == 2) {
    int paid = std::min(n.c0, n.c1);  // the smaller stack is the most either can win
    if (d.rank[0] == d.rank[1]) return 0;
    bool trav_wins = d.rank[trav] < d.rank[1 - trav];
    return trav_wins ? (float)paid : -(float)paid;
  }
  int na = (int)n.children.size();
  uint64_t base = n.offset + (uint64_t)d.bucket[n.player][n.street] * na;
  float s[10];
  strategy(R + base, na, s);
  if (n.player == trav) {
    float v[10], ev = 0;
    for (int a = 0; a < na; ++a) {
      v[a] = traverse(n.children[a], trav, d, rng);
      ev += s[a] * v[a];
    }
    for (int a = 0; a < na; ++a) R[base + a] += v[a] - ev;
    return ev;
  }
  for (int a = 0; a < na; ++a) A[base + a] += s[a];
  double u = std::uniform_real_distribution<double>(0, 1)(rng), acc = 0;
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
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--dry-run") dry = true;
    else if (a == "--threads") threads = atoi(argv[++i]);
    else if (a == "--minutes") minutes = atof(argv[++i]);
    else if (a == "--checkpoint-minutes") ckpt = atof(argv[++i]);
    else if (a == "--discount-minutes") discount_minutes = atof(argv[++i]);
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
  R = (float*)calloc(total, sizeof(float));
  A = (float*)calloc(total, sizeof(float));
  if (!R || !A) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  uint64_t done = 0;
  {
    std::ifstream pf(out + "/iterations.txt");
    if (pf && load(out + "/regret.f32", R, total) && load(out + "/avg.f32", A, total)) {
      pf >> done;
      fprintf(stderr, "resumed at %lu iterations\n", done);
    }
  }
  std::atomic<uint64_t> iters(done);
  double t0 = omp_get_wtime(), last_ckpt = t0, last_disc = t0;
  int disc_round = 0;
  FILE* log = fopen((out + "/progress.log").c_str(), "a");
  std::atomic<bool> stop(false);
#pragma omp parallel num_threads(threads)
  {
    std::mt19937_64 rng(std::random_device{}() ^ (uint64_t)omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
    Deal d;
    while (!stop) {
      for (int k = 0; k < 1000; ++k) {
        deal(rng, d);
        uint64_t it = iters.fetch_add(1);
        traverse(0, (int)(it & 1), d, rng);
      }
#pragma omp master
      {
        double now = omp_get_wtime();
        if ((now - t0) / 60 >= minutes) stop.store(true);
        // linear-CFR style discounting (Pluribus: every 10 minutes for the first 400) - approximate, done in place
        if ((now - t0) / 60 < discount_minutes && now - last_disc >= 600) {
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
