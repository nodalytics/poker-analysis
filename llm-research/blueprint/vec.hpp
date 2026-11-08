// Shared by the vector trainer (mccfr_vec.cpp) and the vector best response (br_vec.cpp): one dealt board with every
// live hand's buckets and showdown rank, fold and showdown values with exact card removal, scratch memory, and setup.
#pragma once
#include <algorithm>
#include <cstring>

#include "common.hpp"
#include "tree.hpp"

static std::vector<uint16_t> FB, TB, RB;
static Indexers* IX;
static int COMBO_A[1326], COMBO_B[1326];

struct Board {
  int n;                 // live hands (not touching the board)
  int hand[1326];        // combo index of each live hand
  int bucket[1326][4];   // per street
  int rank[1326];        // showdown rank, lower is stronger
  int order[1326];       // live positions sorted strongest first
  uint8_t card[5];
  // the distinct buckets on this board per street, and each hand's index into them: hands sharing a bucket share a
  // strategy, so the trainer works per bucket (a few hundred) rather than per hand (1,081)
  int nub[4];
  int ub[4][1326];
  int uidx[1326][4];
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
  static thread_local std::vector<int> slot(1 << 16, -1);
  for (int st = 0; st < 4; ++st) {
    b.nub[st] = 0;
    for (int i = 0; i < b.n; ++i) {
      int bk = b.bucket[i][st];
      if (slot[bk] < 0) slot[bk] = b.nub[st], b.ub[st][b.nub[st]++] = bk;
      b.uidx[i][st] = slot[bk];
    }
    for (int u = 0; u < b.nub[st]; ++u) slot[b.ub[st][u]] = -1;
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

// Card tables, the tree and its table offsets for abstraction ABS. Returns the table length (floats), 0 on error.
static uint64_t vec_init(const std::string& abs) {
  BUCKETS[1] = BUCKETS[2] = BUCKETS[3] = bucket_count(abs);
  T.reserve(1 << 20);
  build({0, 0, SB, BB, 0, 0, BB});
  uint64_t total = 0;
  for (auto& n : T)
    if (n.kind == 0) {
      n.offset = total;
      total += (uint64_t)BUCKETS[n.street] * n.children.size();
    }
  int k = 0;
  for (int a = 0; a < 52; ++a)
    for (int c = a + 1; c < 52; ++c) COMBO_A[k] = a, COMBO_B[k] = c, ++k;
  IX = new Indexers();
  FB = load_buckets(abs, "flop");
  TB = load_buckets(abs, "turn");
  RB = load_buckets(abs, "river");
  if (FB.empty() || TB.empty() || RB.empty()) {
    fprintf(stderr, "bucket tables missing in %s\n", abs.c_str());
    return 0;
  }
  return total;
}
