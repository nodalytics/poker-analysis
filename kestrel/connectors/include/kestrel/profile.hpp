// Opponent profiles: an opponent's statistics as a vector, compared by cosine similarity with reference player types
// (nit, TAG, LAG, maniac, calling station, loose-passive) and with the other players Kestrel has met.
//
// The vector has six rates: VPIP, pre-flop raise, 3-bet, fold to a continuation bet, post-flop aggression frequency
// (bets and raises over bets, raises and calls) and went to showdown. Each rate is pulled towards the average of the
// types in proportion to how few chances it had (k = 8 chances of prior), so a player seen for 10 hands is not
// declared a maniac on two raises; then centred on that average and divided by the spread of the types, with the
// pre-flop rates weighted double (3-bet x1.5), as they rest on many more chances. Cosine similarity then compares the
// direction of a player's deviations from the average: looser and more passive than average reads as a calling
// station whether it is a little or a lot.
//
// The reference rates are approximate, from common HUD guides for 6-max and full-ring online games; they name the
// style, they do not measure it. Kestrel does not change its play on a profile (bounded exploitation comes later:
// docs/design.md, Status).
#pragma once
#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kestrel {

struct OpponentStats;

constexpr int kProfileDim = 6;

struct PlayerType {
    std::string name, about;
    std::array<double, kProfileDim> rates;  // vpip, pfr, 3-bet, fold to c-bet, aggression frequency, went to showdown
};

const std::vector<PlayerType>& player_types();

// the player's rates, each pulled towards the types' average by its number of chances
std::array<double, kProfileDim> profile_rates(const OpponentStats& s);
// the vector compared: rates centred on the types' average and scaled by their spread
std::array<double, kProfileDim> profile_vector(const std::array<double, kProfileDim>& rates);

// {type, similarity, runner_up, runner_up_similarity, hands, settled (enough hands to say), rates}
nlohmann::json profile_json(const OpponentStats& s);

}  // namespace kestrel
