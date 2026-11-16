// Search at decision time for the 9-max networks: a port of llm-research/scripts/ninemax_search.py, which beat plain
// nm25 by +2 to +6 bb/100 against four kinds of opponent (paired, 4,000 deals each; docs/2026-11-15-what-the-models-
// learn.md).
//
// At our decision: deal the unknown cards many times (the opponents' hands and the rest of the board), weight each deal
// by how likely the network was to take every action each opponent took with those cards (their ranges as the network
// sees them), try each legal action and play every deal to the end with the network for everyone (all of them in
// lockstep, one batched forward pass a step), and score the results as the network trains: the change in ICM equity.
// Greedy: the action with the best weighted value. Robust (after Pluribus): every deal is played four times, the
// opponents continuing as the network or the network tilted towards folding, calling or raising, and an action is
// worth its worst case of the four.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "kestrel/net.hpp"

namespace kestrel {

struct SearchConfig {
    int samples = 1024;   // deals of the unknown cards, 16 to 4,096 (the Python tests used 2,048)
    bool robust = false;  // the opponents may switch continuation strategy
    double bias = 5.0;    // how strongly a continuation strategy is tilted
    uint64_t seed = 1;
};

// one opponent decision earlier in the hand, as the network saw it: the table's features, the legal moves, the street,
// and the move taken (as a slot of the network's menu)
struct OppDecision {
    int seat = 0, street = 0, slot = 0;
    std::vector<float> feat;
    std::vector<uint8_t> legal;
};

struct SearchResult {
    std::vector<float> policy;  // over the menu's slots
    std::vector<double> q;      // each legal slot's value, big blinds of ICM equity (NaN: not legal)
    double ess = 0;             // effective number of deals after the weights
    int samples = 0;
};

// throws std::invalid_argument when the cards cannot be dealt (repeated cards, a board over 5, the hero not seated)
SearchResult ninemax_search(const Net& net, const Table& T, const TourneyCtx& ctx, const std::vector<OppDecision>& hist,
                            int hero, const std::array<int, 2>& hole, const std::vector<int>& board,
                            const SearchConfig& cfg);

}  // namespace kestrel
