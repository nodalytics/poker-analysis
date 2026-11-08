// Shared library for Python: the blueprint's card buckets (same indexers and tables as mccfr).
//   int bp_init(const char* abstraction_dir)            0 on success
//   int bp_bucket(const int* cards, int n)              cards = hole[2] + board[0,3,4,5], rank*4+suit (suit c,d,h,s)
// Returns the pre-flop class (0..168) with no board, else the street's bucket (0..199); -1 on error.
//   void bp_deal(int n, uint64_t seed, int32_t* buckets, int32_t* ranks)
// Deals n heads-up hands for GPU self-play (mlp_selfplay.py): buckets[n][2][4] (pre-flop class, flop, turn,
// river bucket per player) and ranks[n][2] (7-card rank, lower is stronger). Multithreaded with OpenMP.
#ifdef _OPENMP  // bp_deal's threads; the Windows build for the advisor goes without
#include <omp.h>
#endif

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

static Indexers* IX = nullptr;
static std::vector<uint16_t> FB, TB, RB;
// Without the 123 MB river table (the advisor's data pack), river buckets come from hand strength: river buckets are
// equal-population cuts of HS, so bucket = the first b with HS <= cut[b] (river_cuts.f32, one upper edge per bucket).
static std::vector<float> RCUT;

static int river_bucket_from_hs(const uint8_t* c) {
  bool used[52] = {false};
  for (int k = 0; k < 7; ++k) used[c[k]] = true;
  int me = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
  double win = 0;
  int n = 0;
  for (int a = 0; a < 52; ++a) {
    if (used[a]) continue;
    for (int b = a + 1; b < 52; ++b) {
      if (used[b]) continue;
      int op = evaluate_7cards(a, b, c[2], c[3], c[4], c[5], c[6]);
      win += me < op ? 1.0 : me == op ? 0.5 : 0.0;
      ++n;
    }
  }
  float hs = (float)(win / n);
  int bk = (int)(std::lower_bound(RCUT.begin(), RCUT.end(), hs) - RCUT.begin());
  return std::min(bk, (int)RCUT.size() - 1);
}

extern "C" int bp_init(const char* dir) {
  if (IX) return 0;
  std::string d(dir);
  FB = load_buckets(d, "flop");
  TB = load_buckets(d, "turn");
  RB = load_buckets(d, "river");
  if (RB.empty()) {
    FILE* f = fopen((d + "/river_cuts.f32").c_str(), "rb");
    if (f) {
      fseek(f, 0, SEEK_END);
      RCUT.resize(ftell(f) / 4);
      fseek(f, 0, SEEK_SET);
      if (fread(RCUT.data(), 4, RCUT.size(), f) != RCUT.size()) RCUT.clear();
      fclose(f);
    }
  }
  if (FB.empty() || TB.empty() || (RB.empty() && RCUT.empty())) return 1;
  IX = new Indexers();
  return 0;
}

extern "C" int bp_bucket(const int* cards, int n) {
  if (!IX || (n != 2 && n != 5 && n != 6 && n != 7)) return -1;
  uint8_t c[7];
  bool seen[52] = {false};
  for (int i = 0; i < n; ++i) {
    if (cards[i] < 0 || cards[i] > 51 || seen[cards[i]]) return -1;  // the indexer aborts on duplicates
    seen[cards[i]] = true;
    c[i] = (uint8_t)cards[i];
  }
  if (n == 2) return (int)hand_index_last(&IX->pre, c);
  if (n == 5) return FB[hand_index_last(&IX->flop, c)];
  if (n == 6) return TB[hand_index_last(&IX->turn, c)];
  return RB.empty() ? river_bucket_from_hs(c) : RB[hand_index_last(&IX->river, c)];
}

static inline uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

extern "C" void bp_deal(int n, uint64_t seed, int32_t* bk, int32_t* rk) {
  if (!IX) return;
#pragma omp parallel for schedule(static)
  for (int g = 0; g < n; ++g) {
    uint64_t st = seed * 0x100000001B3ULL + (uint64_t)g;
    uint8_t deck[52];
    for (int k = 0; k < 52; ++k) deck[k] = (uint8_t)k;
    for (int k = 0; k < 9; ++k) {  // partial Fisher-Yates: 2 + 2 hole cards, 5 board cards
      int j = k + (int)(splitmix(st) % (uint64_t)(52 - k));
      std::swap(deck[k], deck[j]);
    }
    for (int p = 0; p < 2; ++p) {
      uint8_t c[7] = {deck[2 * p], deck[2 * p + 1], deck[4], deck[5], deck[6], deck[7], deck[8]};
      int32_t* b = bk + (size_t)g * 8 + p * 4;
      b[0] = (int32_t)hand_index_last(&IX->pre, c);
      b[1] = FB[hand_index_last(&IX->flop, c)];
      b[2] = TB[hand_index_last(&IX->turn, c)];
      b[3] = RB.empty() ? river_bucket_from_hs(c) : RB[hand_index_last(&IX->river, c)];
      rk[(size_t)g * 2 + p] = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    }
  }
}
