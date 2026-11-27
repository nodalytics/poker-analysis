// Kestrel's 9-max search against the network alone, paired: every deal is played twice on the same cards, the hero
// (one seat, rotating) once with the network's own policy and once with search, the other eight seats the network
// (optionally tilted towards folding, calling or raising, as ninemax_search.py's --opp-bias), the hero in every seat
// of every deal in turn. The gain is scored deal by deal, so the cards' luck cancels. Tables are drawn as the network's training drew them (icm.scenarios: final
// tables, bubbles and in-the-money tables, --depths average stacks, --short of the final tables short-handed).
//
//   kestrel_search_match --model DIR --pack DIR [--deals 2000] [--samples 1024] [--robust] [--opp-bias net|fold|call|raise]
//                        [--depths 10,25] [--short 0.5] [--seed 1] [--no-crn] [--min-ess 0.02] [--oversample 1]
//                        [--per-deal FILE]
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kestrel/buckets.hpp"
#include "kestrel/icm.hpp"
#include "kestrel/net.hpp"
#include "kestrel/ninemax_search.hpp"

extern "C" int evaluate_7cards(int a, int b, int c, int d, int e, int f, int g);

using namespace kestrel;

namespace {

struct Game {
    std::vector<int64_t> stacks;
    TourneyCtx ctx;
};

// icm.scenarios for one table
Game scenario(std::mt19937_64& rng, int P, double lo, double hi, double short_share) {
    std::uniform_real_distribution<double> u(0, 1);
    std::normal_distribution<double> nz(0, 1);
    const double r = u(rng), spread = 0.7;
    const int kind = r < 0.4 ? 0 : r < 0.8 ? 1 : 2;
    const double avg = lo * std::pow(hi / lo, u(rng));
    Game g;
    g.stacks.assign(P, 0);
    for (int s = 0; s < P; ++s)
        g.stacks[s] = std::max<int64_t>(2 * BB, std::llround(avg * std::exp(spread * nz(rng) - spread * spread / 2) * BB));
    const double alpha = 0.6 + 0.8 * u(rng);
    const int busts = 1 + int(rng() % 3);
    const double m = 0.3 + 0.7 * u(rng);
    std::vector<double> pay(P);
    for (int k = 0; k < P; ++k) {
        const double place = k + 1, fin = std::pow(place, -alpha);
        pay[k] = kind == 0 ? fin : kind == 1 ? (place > P - busts ? 0.0 : fin) : 1 + m * (P - place) / P;
    }
    if (short_share > 0 && kind == 0 && u(rng) < short_share) {
        const int n = 2 + int(rng() % (P - 2));
        for (int s = 2; s < P - (n - 2); ++s) g.stacks[s] = 0;
        for (int k = n; k < P; ++k) pay[k] = 0;
    }
    const double z = std::accumulate(pay.begin(), pay.end(), 0.0);
    for (auto& x : pay) x /= z;
    g.ctx.pay = pay;
    g.ctx.eq0 = start_equity(std::vector<double>(g.stacks.begin(), g.stacks.end()), g.ctx);
    return g;
}

void tilt(std::vector<float>& p, int kind, double factor) {
    if (kind == 0) return;
    double z = 0;
    for (size_t a = 0; a < p.size(); ++a) {
        if ((kind == 1 && a == 0) || (kind == 2 && a == 1) || (kind == 3 && a >= 2)) p[a] = float(p[a] * factor);
        z += p[a];
    }
    for (auto& x : p) x = float(x / z);
}

struct Played {
    double result = 0;  // the hero's ICM equity change, big blinds
    int searched = 0, fallen = 0;
    double ms = 0, ess = 0;
};

// one hand: the hero searches or not; the opponents' moves come from rng (the same seed in both runs)
Played play(const Net& net, const Game& g, const std::vector<int>& deck, int hero, bool search, const SearchConfig& sc,
            int opp_kind, uint64_t seed) {
    const NetInfo& info = net.info();
    const int P = int(g.stacks.size());
    Table T(g.stacks, Menu::named(info.menu), info.rules);
    std::vector<std::array<int, 4>> bk(P);
    std::vector<int> rank(P, 0);
    const int d = 2 * P;
    for (int s = 0; s < P; ++s) {
        int c[7] = {deck[2 * s], deck[2 * s + 1], deck[d], deck[d + 1], deck[d + 2], deck[d + 3], deck[d + 4]};
        bk[s] = {bucket_of(c, 2), bucket_of(c, 5), bucket_of(c, 6), bucket_of(c, 7)};
        rank[s] = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<OppDecision> hist;
    Played out;
    const int nb[4] = {0, 3, 4, 5};
    while (!T.done()) {
        const int i = T.actor(), st = T.street();
        auto L = T.legal();
        auto f = features(T, info, &g.ctx);
        std::vector<float> lg;
        float v;
        net.forward(f, bk[i], st, strength(info, st, bk[i][st]), lg, v);
        auto p = net.policy(lg, L);
        const double x = u(rng);  // drawn on every decision, so both runs stay in step until they differ
        int act = -1;
        if (i == hero && search) {
            int live = 0;
            for (int s = 0; s < P; ++s) live += T.seated[s] && !T.folded[s];
            if (st == 0 || live <= 3) {
                std::vector<int> board(deck.begin() + d, deck.begin() + d + nb[st]);
                SearchConfig c = sc;
                c.seed = seed * 31 + out.searched;
                auto t0 = std::chrono::steady_clock::now();
                auto r = ninemax_search(net, T, g.ctx, hist, hero, {deck[2 * hero], deck[2 * hero + 1]}, board, c);
                out.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                out.ess += r.ess, ++out.searched;
                if (r.fell_back) {  // below the ESS floor: the network's policy, with this decision's draw
                    double acc = 0;
                    for (int a = 0; a < int(r.policy.size()) && act < 0; ++a)
                        if (L[a] && (acc += r.policy[a]) >= x) act = a;
                    for (int a = 0; a < int(r.policy.size()) && act < 0; ++a)
                        if (L[a]) act = a;
                    ++out.fallen;
                } else {
                    act = int(std::max_element(r.policy.begin(), r.policy.end()) - r.policy.begin());
                }
            }
        }
        if (act < 0) {
            if (i != hero) tilt(p, opp_kind, 5.0);
            double acc = 0;
            for (int a = 0; a < int(p.size()) && act < 0; ++a)
                if (L[a] && (acc += p[a]) >= x) act = a;
            if (act < 0)
                for (int a = 0; a < int(p.size()); ++a)
                    if (L[a]) act = a;
        }
        if (i != hero) hist.push_back({i, st, act, f, L});
        T.step(act);
    }
    auto pay = T.payoffs(rank);
    std::vector<double> st0(g.stacks.begin(), g.stacks.end());
    auto eq1 = icm(end_stacks(st0, pay), g.ctx.pay);
    const double chips = std::accumulate(st0.begin(), st0.end(), 0.0);
    out.result = (eq1[hero] - g.ctx.eq0[hero]) * chips / double(BB);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, pack, per_deal;
    int deals = 2000, opp = 0;
    double lo = 10, hi = 25, short_share = 0.5;
    uint64_t seed = 1;
    SearchConfig sc;
    for (int k = 1; k < argc; ++k) {
        std::string a = argv[k], v = k + 1 < argc ? argv[k + 1] : "";
        if (a == "--robust") {
            sc.robust = true;
            continue;
        }
        if (a == "--no-crn") {  // independent rollout draws per move (the search before 2026-11-23)
            sc.crn = false;
            continue;
        }
        ++k;
        if (a == "--model") model = v;
        else if (a == "--pack") pack = v;
        else if (a == "--deals") deals = std::stoi(v);
        else if (a == "--samples") sc.samples = std::stoi(v);
        else if (a == "--oversample") sc.oversample = std::stoi(v);
        else if (a == "--per-deal") per_deal = v;
        else if (a == "--min-ess") sc.min_ess = std::stod(v);
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--short") short_share = std::stod(v);
        else if (a == "--depths") lo = std::stod(v.substr(0, v.find(','))), hi = std::stod(v.substr(v.find(',') + 1));
        else if (a == "--opp-bias") opp = v == "fold" ? 1 : v == "call" ? 2 : v == "raise" ? 3 : 0;
        else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
    }
    if (model.empty() || pack.empty()) return std::fprintf(stderr, "--model DIR --pack DIR are needed\n"), 2;
    buckets_init(pack);
    Net net(model);
    const int P = net.info().players;
    std::vector<double> plain(deals, 0.0), with(deals, 0.0), ms(deals, 0.0), ess(deals, 0.0);
    std::vector<int> searched(deals), fallen(deals);
    auto t0 = std::chrono::steady_clock::now();
    int done = 0;
#pragma omp parallel for schedule(dynamic, 1)
    for (int n = 0; n < deals; ++n) {
        std::mt19937_64 rng(seed * 1000003 + uint64_t(n));
        Game g = scenario(rng, P, lo, hi, short_share);
        std::vector<int> deck(52);
        std::iota(deck.begin(), deck.end(), 0);
        std::shuffle(deck.begin(), deck.end(), rng);
        // the hero in every seated seat in turn (as ninemax_search.py): the deal's result is the average over seats,
        // so the seats' luck cancels within the deal
        const uint64_t s = rng();
        int seats = 0;
        for (int hero = 0; hero < P; ++hero) {
            if (g.stacks[hero] == 0) continue;
            ++seats;
            plain[n] += play(net, g, deck, hero, false, sc, opp, s + hero).result;
            auto r = play(net, g, deck, hero, true, sc, opp, s + hero);
            with[n] += r.result, searched[n] += r.searched, ms[n] += r.ms, ess[n] += r.ess, fallen[n] += r.fallen;
        }
        plain[n] /= seats, with[n] /= seats;
#pragma omp critical
        if (++done % 200 == 0)
            std::fprintf(stderr, "%d deals, %.0f s\n", done,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    auto stat = [&](const std::vector<double>& x) {
        double m = std::accumulate(x.begin(), x.end(), 0.0) / x.size(), v = 0;
        for (double y : x) v += (y - m) * (y - m);
        const double se = std::sqrt(v / (x.size() - 1) / x.size());
        return std::array<double, 3>{100 * m, 100 * (m - 1.96 * se), 100 * (m + 1.96 * se)};
    };
    std::vector<double> gain(deals);
    for (int n = 0; n < deals; ++n) gain[n] = with[n] - plain[n];
    if (!per_deal.empty()) {  // each deal's result without and with search (big blinds): two runs with the same seed
        FILE* f = std::fopen(per_deal.c_str(), "w");  // play the same deals, so their difference is paired deal by deal
        if (f) {
            for (int n = 0; n < deals; ++n) std::fprintf(f, "%.6f %.6f\n", plain[n], with[n]);
            std::fclose(f);
        }
    }
    const int S = std::accumulate(searched.begin(), searched.end(), 0);
    const int FB = std::accumulate(fallen.begin(), fallen.end(), 0);
    const double MS = std::accumulate(ms.begin(), ms.end(), 0.0), ES = std::accumulate(ess.begin(), ess.end(), 0.0);
    auto a = stat(plain), b = stat(with), c = stat(gain);
    static const char* kinds[] = {"net", "fold", "call", "raise"};
    std::printf("{\"model\": \"%s\", \"deals\": %d, \"samples\": %d, \"robust\": %s, \"opp_bias\": \"%s\", "
                "\"plain_bb100\": [%.2f, %.2f, %.2f], \"search_bb100\": [%.2f, %.2f, %.2f], "
                "\"gain_bb100\": [%.2f, %.2f, %.2f], \"searched\": %d, \"ms_per_search\": %.1f, \"ess_mean\": %.1f, "
                "\"crn\": %s, \"min_ess\": %g, \"ess_fallback\": %d, \"threads\": %d}\n",
                model.c_str(), deals, sc.samples, sc.robust ? "true" : "false", kinds[opp], a[0], a[1], a[2], b[0], b[1],
                b[2], c[0], c[1], c[2], S, S ? MS / S : 0.0, S ? ES / S : 0.0, sc.crn ? "true" : "false", sc.min_ess, FB,
                omp_get_max_threads());
    return 0;
}
