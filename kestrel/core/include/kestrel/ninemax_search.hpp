// Search at decision time for the 9-max networks: a port of research/scripts/ninemax_search.py, which beat plain
// nm25 by +2 to +6 bb/100 against four kinds of opponent (paired, 4,000 deals each; docs/2026-11-15-what-the-models-
// learn.md).
//
// At our decision: deal the unknown cards many times (the opponents' hands and the rest of the board), weight each deal
// by how likely the network was to take every action each opponent took with those cards (their ranges as the network
// sees them), try each legal action and play every deal to the end with the network for everyone (all of them in
// lockstep, one batched forward pass a step, the same random draws for every action), and score the results as the
// network trains: the change in prize equity (ICM, and the knockout term in a bounty game: end_equity).
// Greedy: the action with the best weighted value. Robust (after Pluribus): every deal is played four times, the
// opponents continuing as the network or the network tilted towards folding, calling or raising, and an action is
// worth its worst case of the four.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "kestrel/net.hpp"

namespace kestrel {

// how one opponent plays, as a tilt of the network's own policy: its fold, call and raise probabilities multiplied by
// these (then renormalised); 1, 1, 1 is the network itself
struct OpponentTilt {
    double fold = 1, call = 1, raise = 1;
};

struct SearchConfig {
    int samples = 1024;   // deals of the unknown cards, 16 to 4,096; Kestrel offers 1,024 and up (256 lost to the
                          // plain network by 5-6.5 bb/100, 1,024 gained +6.2 against raise-biased opponents)
    // candidate deals per sample (2026-11-27): oversample x samples deals are weighted, and samples of them are
    // resampled in proportion to their weights (systematic resampling) before the rollouts. The rollouts then rest on
    // many more distinct deals: in research (ninemax_belief_eval.py, nm25, 363 spots, 256 rollouts) the move values'
    // error against a search 64 times larger fell from 1.39 to 0.66 bb. But in Kestrel dealing, bucketing and weighting
    // the extra deals costs more than the rollouts: at 1,024 samples, 8 x made a decision about 7 times slower (about
    // 20 s against 2.9 s; kestrel_search_match, nm25, 2026-11-27), and Kestrel's own search already rests on 447 of
    // 1,024 effective deals on average. So it is off by default until a cheaper version is measured. 1: off.
    int oversample = 1;
    bool robust = false;  // the opponents may switch continuation strategy
    double bias = 5.0;    // how strongly a continuation strategy is tilted
    // safe exploitation: each seat's tilt (Kestrel's seats; empty: none). Every deal is also played out with the
    // opponents tilted so, and a move's value is (1 - exploit_weight) x its usual value + exploit_weight x its value
    // against them. The tilts come already shrunk towards 1 by the evidence behind them (connectors: opponent_tilt),
    // so an opponent seen for a few hands changes nothing.
    std::vector<OpponentTilt> tilts;
    double exploit_weight = 0.0;
    // common random numbers: every rollout step draws one uniform per deal, shared by every candidate move and
    // continuation, so Q(a) - Q(b) is a paired comparison (false: one draw per rollout, as before 2026-11-23)
    bool crn = true;
    // the ESS floor, as a share of the samples: below it the values rest on too few deals, and the result is the
    // network's own policy (SearchResult::fell_back); 0 turns it off. As ninemax_search.py's MIN_ESS.
    double min_ess = 0.02;
    uint64_t seed = 1;           // the deals (and, without crn, the rollouts)
    uint64_t rollout_seed = 0;   // the rollouts' own draws (0: from seed, and without crn the deals' stream as
                                 // before); another value keeps the deals and draws new rollouts
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
    double ess = 0;             // effective number of deals after the weights (at most samples)
    int samples = 0;
    bool fell_back = false;     // the ESS was below the floor: policy is the network's own
};

// throws std::invalid_argument when the cards cannot be dealt (repeated cards, a board over 5, the hero not seated)
SearchResult ninemax_search(const Net& net, const Table& T, const TourneyCtx& ctx, const std::vector<OppDecision>& hist,
                            int hero, const std::array<int, 2>& hole, const std::vector<int>& board,
                            const SearchConfig& cfg);

}  // namespace kestrel
