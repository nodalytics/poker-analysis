#include "kestrel/icm.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace kestrel {

std::vector<double> icm(const std::vector<double>& stacks, const std::vector<double>& payouts) {
    const int P = int(stacks.size());
    if (P > 20) throw std::invalid_argument("icm: at most 20 players");
    const unsigned M = 1u << P;
    std::vector<double> s(P);
    double total = 0;
    for (int j = 0; j < P; ++j) total += s[j] = std::max(stacks[j], 0.0);
    std::vector<double> msum(M, 0.0), prob(M, 0.0), eq(P, 0.0);
    for (unsigned m = 1; m < M; ++m) {
        int j = std::countr_zero(m);
        msum[m] = msum[m & (m - 1)] + s[j];
    }
    prob[0] = 1.0;
    for (int k = 0; k < P; ++k) {
        double pay = k < int(payouts.size()) ? payouts[k] : 0.0;
        for (unsigned m = 0; m < M; ++m) {
            if (std::popcount(m) != k || prob[m] == 0.0) continue;
            double rem = std::max(total - msum[m], 1e-300);
            for (int j = 0; j < P; ++j) {
                if (m >> j & 1u) continue;
                double w = prob[m] * s[j] / rem;
                eq[j] += w * pay;
                if (k + 1 < P) prob[m | (1u << j)] += w;
            }
        }
    }
    return eq;
}

std::vector<double> end_stacks(const std::vector<double>& start, const std::vector<double>& payoff) {
    std::vector<double> r(start.size());
    for (size_t j = 0; j < start.size(); ++j) {
        double after = start[j] + payoff[j];
        r[j] = after <= 0.5 ? start[j] * 1e-9 : after;
    }
    return r;
}

}
