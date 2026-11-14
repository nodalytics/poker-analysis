// ICM (Malmuth-Harville): each place goes to one of the players not yet placed, in proportion to their chips.
// A port of scripts/icm.py (icm, end_stacks).
#pragma once
#include <vector>

namespace kestrel {

// stacks (any non-negative units, 0 = no chips), payouts for places 1..n: each player's prize equity
std::vector<double> icm(const std::vector<double>& stacks, const std::vector<double>& payouts);

// stacks after a hand; players who bust keep a vanishing stack in proportion to their start, so they place below
// everyone else and in the order of their starting stacks
std::vector<double> end_stacks(const std::vector<double>& start, const std::vector<double>& payoff);

}  // namespace kestrel
