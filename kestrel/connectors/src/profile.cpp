#include "kestrel/profile.hpp"

#include <algorithm>
#include <cmath>

#include "kestrel/opponents.hpp"

namespace kestrel {

using json = nlohmann::json;

namespace {

constexpr double kPrior = 8;     // chances of prior weight on each rate
constexpr int kSettled = 30;     // hands before a type is shown

struct Norm {
    std::array<double, kProfileDim> mean{}, spread{};
};

const Norm& norm() {
    static const Norm n = [] {
        Norm r;
        const auto& T = player_types();
        for (int d = 0; d < kProfileDim; ++d) {
            double m = 0, v = 0;
            for (auto& t : T) m += t.rates[d];
            m /= double(T.size());
            for (auto& t : T) v += (t.rates[d] - m) * (t.rates[d] - m);
            r.mean[d] = m, r.spread[d] = std::max(std::sqrt(v / double(T.size())), 0.02);
        }
        return r;
    }();
    return n;
}

double cos6(const std::array<double, kProfileDim>& a, const std::array<double, kProfileDim>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (int d = 0; d < kProfileDim; ++d) ab += a[d] * b[d], aa += a[d] * a[d], bb += b[d] * b[d];
    return aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0;
}

}  // namespace

const std::vector<PlayerType>& player_types() {
    //                               vpip  pfr   3bet  f-cb  aggr  wtsd
    static const std::vector<PlayerType> types = {
        {"nit", "plays few hands, raises them, folds a lot after the flop", {0.13, 0.10, 0.03, 0.60, 0.45, 0.22}},
        {"tag", "tight-aggressive: a selective range played with raises", {0.20, 0.16, 0.06, 0.50, 0.55, 0.26}},
        {"lag", "loose-aggressive: many hands, mostly raised, keeps betting", {0.32, 0.26, 0.10, 0.40, 0.60, 0.28}},
        {"maniac", "plays most hands and raises them; bets and raises after the flop", {0.60, 0.45, 0.20, 0.25, 0.75, 0.32}},
        {"station", "calling station: plays many hands, rarely raises, calls down", {0.50, 0.05, 0.01, 0.20, 0.15, 0.42}},
        {"loose-passive", "limps and calls many hands, folds when it misses", {0.40, 0.10, 0.03, 0.45, 0.30, 0.30}},
    };
    return types;
}

std::array<double, kProfileDim> profile_rates(const OpponentStats& s) {
    const Norm& n = norm();
    const double x[kProfileDim] = {double(s.vpip), double(s.pfr), double(s.threebet), double(s.cbet_folds),
                                   double(s.aggressive), double(s.showdown)};
    const double c[kProfileDim] = {double(s.hands), double(s.hands), double(s.threebet_chance), double(s.cbet_faced),
                                   double(s.aggressive + s.passive), double(s.saw_flop)};
    std::array<double, kProfileDim> r{};
    for (int d = 0; d < kProfileDim; ++d) r[d] = (x[d] + kPrior * n.mean[d]) / (c[d] + kPrior);
    return r;
}

std::array<double, kProfileDim> profile_vector(const std::array<double, kProfileDim>& rates) {
    // the pre-flop rates count double (3-bet 1.5): they have many more chances behind them than the post-flop ones,
    // and are what HUD guides type players by. Unweighted, the test platform's tight-raising "tag" bots read as
    // calling stations from their post-flop calls; weighted, 7 of its 8 bots get the type their rules describe.
    static const double weight[kProfileDim] = {2.0, 2.0, 1.5, 1.0, 1.0, 1.0};
    const Norm& n = norm();
    std::array<double, kProfileDim> v{};
    for (int d = 0; d < kProfileDim; ++d) v[d] = weight[d] * (rates[d] - n.mean[d]) / n.spread[d];
    return v;
}

json profile_json(const OpponentStats& s) {
    const auto rates = profile_rates(s);
    const auto v = profile_vector(rates);
    std::vector<std::pair<double, std::string>> sims;
    for (auto& t : player_types()) sims.push_back({cos6(v, profile_vector(t.rates)), t.name});
    std::sort(sims.begin(), sims.end(), std::greater<>());
    auto r3 = [](double x) { return std::round(x * 1000.0) / 1000.0; };
    json rj = json::array();
    for (double x : rates) rj.push_back(r3(x));
    return {{"type", sims[0].second}, {"similarity", r3(sims[0].first)}, {"runner_up", sims[1].second},
            {"runner_up_similarity", r3(sims[1].first)}, {"hands", s.hands}, {"settled", s.hands >= kSettled},
            {"rates", rj}};
}

}  // namespace kestrel
