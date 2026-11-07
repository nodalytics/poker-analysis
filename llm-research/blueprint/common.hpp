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
