// Prints the pre-flop indexer's 169 classes in index order (index, class like AKs/77/T9o), for reading strategies.
#include "common.hpp"

int main() {
  Indexers ix;
  static const char* R = "23456789TJQKA";
  for (int i = 0; i < 169; ++i) {
    uint8_t c[2];
    hand_unindex(&ix.pre, 0, i, c);
    int r0 = c[0] >> 2, r1 = c[1] >> 2;
    int hi = r0 > r1 ? r0 : r1, lo = r0 > r1 ? r1 : r0;
    printf("%d\t%c%c%s\n", i, R[hi], R[lo], hi == lo ? "" : ((c[0] & 3) == (c[1] & 3) ? "s" : "o"));
  }
  return 0;
}
