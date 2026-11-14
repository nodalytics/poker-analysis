#pragma once
#include <vector>

namespace kestrel {

std::vector<double> icm(const std::vector<double>& stacks, const std::vector<double>& payouts);

std::vector<double> end_stacks(const std::vector<double>& start, const std::vector<double>& payoff);

}
