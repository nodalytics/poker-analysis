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

void profile_counts(const OpponentStats& s, bool recent, double x[kProfileDim], double n[kProfileDim]) {
    if (recent) {
        for (int d = 0; d < kProfileDim; ++d) x[d] = s.recent_x[d], n[d] = s.recent_n[d];
        return;
    }
    const double xs[kProfileDim] = {double(s.vpip), double(s.pfr), double(s.threebet), double(s.cbet_folds),
                                    double(s.aggressive), double(s.showdown)};
    const double ns[kProfileDim] = {double(s.hands), double(s.hands), double(s.threebet_chance), double(s.cbet_faced),
                                    double(s.aggressive + s.passive), double(s.saw_flop)};
    for (int d = 0; d < kProfileDim; ++d) x[d] = xs[d], n[d] = ns[d];
}

RatePrior population_prior(const std::vector<const OpponentStats*>& players) {
    RatePrior p;
    const Norm& nm = norm();
    for (int d = 0; d < kProfileDim; ++d) p.mean[d] = nm.mean[d], p.strength[d] = kPrior;
    int fitted = 0;
    for (int d = 0; d < kProfileDim; ++d) {
        double sx = 0, sn = 0;
        std::vector<std::pair<double, double>> r;  // (rate, chances) of players with enough chances
        for (auto* s : players) {
            double x[kProfileDim], n[kProfileDim];
            profile_counts(*s, false, x, n);
            if (n[d] >= 10) r.push_back({x[d] / n[d], n[d]}), sx += x[d], sn += n[d];
        }
        if (r.size() < 5) continue;
        const double m = std::clamp(sx / sn, 0.01, 0.99);
        double var = 0, noise = 0;
        for (auto& [rate, n] : r) var += (rate - m) * (rate - m), noise += m * (1 - m) / n;
        var /= double(r.size()), noise /= double(r.size());
        const double between = std::max(var - noise, 1e-4);  // how much players truly differ
        p.mean[d] = m;
        p.strength[d] = std::clamp(m * (1 - m) / between - 1, 2.0, 200.0);  // the Beta prior's a + b
        fitted = std::max(fitted, int(r.size()));
    }
    p.players = fitted;
    return p;
}

std::array<double, kProfileDim> profile_rates(const OpponentStats& s) {
    double x[kProfileDim], n[kProfileDim];
    profile_counts(s, false, x, n);
    const Norm& nm = norm();
    std::array<double, kProfileDim> r{};
    for (int d = 0; d < kProfileDim; ++d) r[d] = (x[d] + kPrior * nm.mean[d]) / (n[d] + kPrior);
    return r;
}

namespace {
// the posterior mean of each rate under a prior (Beta-binomial)
std::array<double, kProfileDim> shrunk(const OpponentStats& s, const RatePrior& p, bool recent) {
    double x[kProfileDim], n[kProfileDim];
    profile_counts(s, recent, x, n);
    std::array<double, kProfileDim> r{};
    for (int d = 0; d < kProfileDim; ++d) r[d] = (x[d] + p.strength[d] * p.mean[d]) / (n[d] + p.strength[d]);
    return r;
}
}  // namespace

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

json profile_json(const OpponentStats& s, const RatePrior* prior) {
    RatePrior types;  // without a population, the reference types' average with 8 chances
    for (int d = 0; d < kProfileDim; ++d) types.mean[d] = norm().mean[d], types.strength[d] = kPrior;
    const RatePrior& p = prior ? *prior : types;
    auto r3 = [](double x) { return std::round(x * 1000.0) / 1000.0; };
    auto classify = [&](const std::array<double, kProfileDim>& rates) {
        const auto v = profile_vector(rates);
        std::vector<std::pair<double, std::string>> sims;
        for (auto& t : player_types()) sims.push_back({cos6(v, profile_vector(t.rates)), t.name});
        std::sort(sims.begin(), sims.end(), std::greater<>());
        return sims;
    };
    const auto rates = shrunk(s, p, false);
    const auto life = classify(rates);
    const auto recent = classify(shrunk(s, p, true));
    const double recent_hands = s.recent_n[0];  // the forgetting count of hands (chances for VPIP)
    const bool settled = s.hands >= kSettled, recent_settled = recent_hands >= 20;
    // evidence of a change: the largest z of a pre-flop rate's recent value against the lifetime one (binomial error at
    // the recent sample size); a different type label alone is not enough (labels flip near a boundary)
    double change_z = 0;
    for (int d = 0; d < 3; ++d) {
        if (s.recent_n[d] < 10) continue;
        const double p = std::clamp(rates[d], 0.01, 0.99), q = s.recent_x[d] / s.recent_n[d];
        change_z = std::max(change_z, std::fabs(q - p) / std::sqrt(p * (1 - p) / s.recent_n[d]));
    }
    json rj = json::array(), pm = json::array(), ps = json::array();
    for (double x : rates) rj.push_back(r3(x));
    for (int d = 0; d < kProfileDim; ++d) pm.push_back(r3(p.mean[d])), ps.push_back(std::round(p.strength[d] * 10) / 10);
    return {{"type", life[0].second}, {"similarity", r3(life[0].first)}, {"runner_up", life[1].second},
            {"runner_up_similarity", r3(life[1].first)}, {"hands", s.hands}, {"settled", settled}, {"rates", rj},
            {"prior", {{"mean", pm}, {"strength", ps}, {"players", p.players}}},
            {"recent", {{"type", recent[0].second}, {"similarity", r3(recent[0].first)}, {"hands", r3(recent_hands)},
                        {"settled", recent_settled}}},
            {"change_z", r3(change_z)},
            {"changed", settled && recent_settled && recent[0].second != life[0].second && change_z >= 3}};
}

OpponentTilt opponent_tilt(const json& st) {
    OpponentTilt t;
    if (!st.is_object() || !st.contains("profile")) return t;
    const json& pr = st["profile"];
    if (!pr.contains("rates") || !pr.contains("prior")) return t;
    auto z = [&](int d) {  // the shrunk rate against the population, in prior standard deviations
        const double r = pr["rates"][d].get<double>(), m = std::clamp(pr["prior"]["mean"][d].get<double>(), 0.01, 0.99);
        const double k = pr["prior"]["strength"][d].get<double>();
        return std::clamp((r - m) / std::sqrt(m * (1 - m) / (k + 1)), -2.0, 2.0);
    };
    const double hands = st.value("hands", 0.0), w = 0.5 * hands / (hands + 100.0);
    // rates: vpip, pfr, 3-bet, fold to c-bet, aggression frequency, went to showdown
    t.fold = std::exp(w * z(3));
    t.call = std::exp(w * (z(5) - z(4)) / 2);
    t.raise = std::exp(w * (z(4) + z(1)) / 2);
    return t;
}

}  // namespace kestrel
