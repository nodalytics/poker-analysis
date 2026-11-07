// Shared helpers: hand indexers per street and card utilities. Cards are rank*4+suit, rank 0 = deuce .. 12 = ace
// (the convention of both PokerHandEvaluator and hand-isomorphism).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define _Bool bool  // hand_index.h is C99
extern "C" {
#include "hand_index.h"
int evaluate_7cards(int a, int b, int c, int d, int e, int f, int g);  // lower is stronger
}

struct Indexers {
  hand_indexer_t pre, flop, turn, river;
  Indexers() {
    // board order does not matter within a street's evaluation, so the board is one round: hole cards, then all
    // board cards (flop 1,286,792 / turn 13,960,050 / river 123,156,254 states)
    const uint8_t c1[] = {2}, c2[] = {2, 3}, c3[] = {2, 4}, c4[] = {2, 5};
    if (!hand_indexer_init(1, c1, &pre) || !hand_indexer_init(2, c2, &flop) || !hand_indexer_init(2, c3, &turn) ||
        !hand_indexer_init(2, c4, &river)) {
      fprintf(stderr, "hand indexer init failed\n");
      exit(1);
    }
  }
  uint64_t size(int street) const {
    const hand_indexer_t* h[] = {&pre, &flop, &turn, &river};
    return hand_indexer_size(h[street], street ? 1 : 0);
  }
  const hand_indexer_t* get(int street) const {
    const hand_indexer_t* h[] = {&pre, &flop, &turn, &river};
    return h[street];
  }
};

inline std::string card_str(int c) {
  static const char* R = "23456789TJQKA";
  static const char* S = "cdhs";
  return std::string(1, R[c >> 2]) + S[c & 3];
}

// Bucket tables: 16-bit (OUT/<street>_buckets.u16, written when there are more than 255 buckets) or 8-bit (.u8).
inline std::vector<uint16_t> load_buckets(const std::string& dir, const char* street) {
  std::vector<uint16_t> out;
  std::string p16 = dir + "/" + street + "_buckets.u16", p8 = dir + "/" + street + "_buckets.u8";
  FILE* f = fopen(p16.c_str(), "rb");
  bool wide = f != nullptr;
  if (!f) f = fopen(p8.c_str(), "rb");
  if (!f) return out;
  fseek(f, 0, SEEK_END);
  size_t n = ftell(f) / (wide ? 2 : 1);
  fseek(f, 0, SEEK_SET);
  out.resize(n);
  if (wide) {
    if (fread(out.data(), 2, n, f) != n) out.clear();
  } else {
    std::vector<uint8_t> b(n);
    if (fread(b.data(), 1, n, f) != n) out.clear();
    else
      for (size_t i = 0; i < n; ++i) out[i] = b[i];
  }
  fclose(f);
  return out;
}

inline int bucket_count(const std::string& dir) {
  FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
  int b = 200;
  if (f) {
    char key[64];
    int v;
    while (fscanf(f, "%63s %d", key, &v) == 2)
      if (std::string(key) == "buckets") b = v;
    fclose(f);
  }
  return b;
}

