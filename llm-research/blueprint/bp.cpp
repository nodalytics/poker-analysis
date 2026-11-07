// Shared library for Python: the blueprint's card buckets (same indexers and tables as mccfr).
//   int bp_init(const char* abstraction_dir)            0 on success
//   int bp_bucket(const int* cards, int n)              cards = hole[2] + board[0,3,4,5], rank*4+suit (suit c,d,h,s)
// Returns the pre-flop class (0..168) with no board, else the street's bucket (0..199); -1 on error.
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

static Indexers* IX = nullptr;
static std::vector<uint16_t> FB, TB, RB;

extern "C" int bp_init(const char* dir) {
  if (IX) return 0;
  std::string d(dir);
  FB = load_buckets(d, "flop");
  TB = load_buckets(d, "turn");
  RB = load_buckets(d, "river");
  if (FB.empty() || TB.empty() || RB.empty()) return 1;
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
  return RB[hand_index_last(&IX->river, c)];
}
