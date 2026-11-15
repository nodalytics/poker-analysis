#pragma once
#include <vector>

namespace kestrel {

// prize equity from chip stacks (payouts for places 1..). Malmuth-Harville: places from the top, the next one drawn
// in proportion to chips. The networks read this one, as they were trained on it.
std::vector<double> icm(const std::vector<double>& stacks, const std::vector<double>& payouts);
// Malmuth-Weitzman: places from the bottom, the next to bust drawn in proportion to 1/stack (an empty seat or a busted
// player takes the lowest place left). An analysis option only (llm-research/docs/2026-11-15-tournament-equity-models.md)
std::vector<double> icm_mw(const std::vector<double>& stacks, const std::vector<double>& payouts);

std::vector<double> end_stacks(const std::vector<double>& start, const std::vector<double>& payoff);

}
