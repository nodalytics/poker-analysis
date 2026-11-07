// Cross-check: N random river states -> cards and hand strength, for comparison with an independent evaluator.
// Usage: check N   (prints "index c1 c2 b1 b2 b3 b4 b5 HS" per line)
#include <random>

#include "common.hpp"

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 200;
  Indexers ix;
  std::mt19937_64 rng(7);
  uint64_t NR = ix.size(3);
  printf("river_states %lu\n", NR);
  for (int k = 0; k < n; ++k) {
    uint64_t i = rng() % NR;
    uint8_t c[7];
    hand_unindex(&ix.river, 1, i, c);
    bool used[52] = {false};
    for (int j = 0; j < 7; ++j) used[c[j]] = true;
    int me = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]), win = 0, tie = 0, m = 0;
    for (int a = 0; a < 52; ++a)
      for (int b = a + 1; b < 52; ++b) {
        if (used[a] || used[b]) continue;
        int op = evaluate_7cards(a, b, c[2], c[3], c[4], c[5], c[6]);
        win += me < op, tie += me == op, ++m;
      }
    printf("%lu", i);
    for (int j = 0; j < 7; ++j) printf(" %s", card_str(c[j]).c_str());
    printf(" %.6f\n", (win + 0.5 * tie) / m);
    // the index must round-trip
    if (hand_index_last(&ix.river, c) != i) printf("ROUNDTRIP FAIL %lu\n", i);
  }
  return 0;
}
