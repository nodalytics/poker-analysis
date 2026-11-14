// Luck-adjusted results for heads-up sessions where both hands are shown at the end (Slumbot shows them): AIVAT's
// chance-node corrections in the unbiased form of llm-research/scripts/aivat_lite.py, with exact equities.
//   all-in      both all-in before the river: the result becomes its expectation, equity x pot - our chips
//   pre-flop    a control variate on our pre-flop equity against their actual hand (mean exactly 0.5 whatever
//               anyone plays), slope per position fitted on the other half of the hands
//   board       for each flop, turn and river reached: pot at that street x (equity after - equity before), a
//               martingale over the cards; its weight cross-fitted the same way
#pragma once
#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kestrel {

// our equity (win 1, tie 1/2) against their hand on a partial board: every runout after the flop, 20,000 sampled
// boards before it (it only feeds the control variate, where sampling noise cannot bias the estimate)
double equity(const std::array<int, 2>& mine, const std::array<int, 2>& theirs, const std::vector<int>& board);

struct LuckHand {
    double raw = 0;        // result in big blinds
    double adjusted = 0;   // after the all-in correction
    double pre = 0.5;      // pre-flop equity
    double board = 0;      // board-card luck term (big blinds)
    int pos = 0;
    bool allin = false;
};

// one hand from a log line (Kestrel: result_bb, their_hole; the Python Slumbot client: winnings_chips,
// slumbot_hole), in an ACPC heads-up game; false when their cards are not shown
bool luck_hand(const nlohmann::json& line, int64_t big, int64_t stack, LuckHand& out);

struct LuckEstimate {
    int hands = 0, allin = 0;
    double raw = 0, raw_se = 0;            // bb/100 and its standard error
    double adjusted = 0, adjusted_se = 0;  // all three corrections
    double variance_reduction = 1;
    nlohmann::json to_json() const;
};

LuckEstimate luck_estimate(const std::vector<LuckHand>& hands, uint64_t seed = 1);

}  // namespace kestrel
