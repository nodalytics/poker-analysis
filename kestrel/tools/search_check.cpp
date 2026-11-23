// Checks of Kestrel's 9-max search (B3 of research/docs/2026-11-23-review-and-plan.md), the counterpart of
// research/scripts/ninemax_search_check.py:
//   bounty  a move whose value is exact (folding to a heads-up shove ends the hand) is scored with the trainer's reward,
//           end_equity - start_equity, knockout terms included (nm30, PKO); without bounties (nm25) it is plain ICM
//   ess     below the floor the result is the network's own policy (fell_back), above it the greedy move
//   crn     the variance of Q(a) - Q(b) over searches of the same spots with the same deals and new rollout draws,
//           with common random numbers and without (printed; the pairing is checked on the draws in Python)
//
//   kestrel_search_check --data DIR (holding models/ninemax/{nm25,nm30} and blueprint/pack) [--spots 12] [--repeats 16]
#include <algorithm>
#include <array>
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

using namespace kestrel;

namespace {

int fails = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    fails += !ok;
}

// a heads-up pre-flop spot: the first player shoves, the hero (the other) to act; ctx: prizes and, with beta, bounties
struct Spot {
    Table T;
    TourneyCtx ctx;
    std::vector<OppDecision> hist;
    int hero = 0;
};

Spot shove_spot(const Net& net, double beta, int64_t s0, int64_t s1) {
    const NetInfo& info = net.info();
    std::vector<int64_t> stacks(9, 0);
    stacks[0] = s0 * BB, stacks[1] = s1 * BB;
    Spot sp{Table(stacks, Menu::named(info.menu), info.rules), {}, {}, 0};
    sp.ctx.pay.assign(9, 0.0);
    sp.ctx.pay[0] = 0.65 * (1 - beta), sp.ctx.pay[1] = 0.35 * (1 - beta);
    if (beta > 0) sp.ctx.bounty = {0.7 * beta, 0.3 * beta, 0, 0, 0, 0, 0, 0, 0};
    sp.ctx.eq0 = start_equity(std::vector<double>(stacks.begin(), stacks.end()), sp.ctx);
    Table& T = sp.T;
    const int A = T.menu().A, i = T.actor();
    auto L = T.legal();
    sp.hist.push_back({i, 0, A - 1, features(T, info, &sp.ctx), L});
    T.step(A - 1);
    sp.hero = T.actor();
    return sp;
}

void part_bounty(const std::string& data) {
    std::printf("bounty\n");
    for (auto [name, beta] : {std::pair<const char*, double>{"nm30", 0.5}, {"nm25", 0.0}}) {
        Net net(data + "/models/ninemax/" + name);
        for (auto [s0, s1] : {std::pair<int64_t, int64_t>{12, 30}, {30, 12}, {20, 20}}) {
            Spot sp = shove_spot(net, beta, s0, s1);
            SearchConfig cfg;
            cfg.samples = 64, cfg.min_ess = 0;
            auto r = ninemax_search(net, sp.T, sp.ctx, sp.hist, sp.hero, {0, 5}, {}, cfg);
            Table F = sp.T;
            F.step(0);
            const std::vector<double> st0(sp.T.stack.begin(), sp.T.stack.end());
            const double chips = std::accumulate(st0.begin(), st0.end(), 0.0);
            const std::vector<int> ranks(9, 0);
            auto pay = F.payoffs(ranks);
            const double want = (end_equity(st0, pay, sp.ctx, F.contrib, F.folded, ranks)[sp.hero] - start_equity(st0, sp.ctx)[sp.hero]) * chips / BB;
            const double plain = (icm(end_stacks(st0, pay), sp.ctx.pay)[sp.hero] - icm(st0, sp.ctx.pay)[sp.hero]) * chips / BB;
            char w[200];
            std::snprintf(w, sizeof w, "%s %lldbb vs %lldbb: Q(fold) %.6f, the trainer's reward %.6f (plain ICM %.6f)",
                          name, (long long)s0, (long long)s1, r.q[0], want, plain);
            check(F.done() && std::fabs(r.q[0] - want) < 1e-9, w);
        }
    }
}

// post-flop spots of at most three players from self-play: the table at the hero's decision and what search needs
struct Live {
    Table T;
    TourneyCtx ctx;
    std::vector<OppDecision> hist;
    int hero;
    std::array<int, 2> hole;
    std::vector<int> board;
};

std::vector<Live> spots(const Net& net, int n, uint64_t seed) {
    const NetInfo& info = net.info();
    std::mt19937_64 rng(seed);
    std::vector<Live> out;
    static const double prizes[9] = {0.30, 0.20, 0.145, 0.105, 0.08, 0.065, 0.05, 0.035, 0.02};
    while (int(out.size()) < n) {
        std::vector<int64_t> stacks(9);
        for (auto& s : stacks) s = BB * (8 + int64_t(rng() % 40));
        TourneyCtx ctx;
        ctx.pay.assign(prizes, prizes + 9);
        ctx.eq0 = start_equity(std::vector<double>(stacks.begin(), stacks.end()), ctx);
        std::vector<int> deck(52);
        std::iota(deck.begin(), deck.end(), 0);
        std::shuffle(deck.begin(), deck.end(), rng);
        std::vector<std::array<int, 4>> bk(9);
        for (int s = 0; s < 9; ++s) {
            int c[7] = {deck[2 * s], deck[2 * s + 1], deck[18], deck[19], deck[20], deck[21], deck[22]};
            bk[s] = {bucket_of(c, 2), bucket_of(c, 5), bucket_of(c, 6), bucket_of(c, 7)};
        }
        Table T(stacks, Menu::named(info.menu), info.rules);
        const int hero = int(rng() % 9), nb[4] = {0, 3, 4, 5};
        std::vector<OppDecision> hist;
        bool taken = false;
        while (!T.done()) {
            const int i = T.actor(), st = T.street();
            auto L = T.legal();
            auto f = features(T, info, &ctx);
            int live = 0;
            for (int s = 0; s < 9; ++s) live += T.seated[s] && !T.folded[s];
            if (i == hero && st >= 1 && live <= 3 && !taken) {
                out.push_back({T, ctx, hist, hero, {deck[2 * hero], deck[2 * hero + 1]},
                               std::vector<int>(deck.begin() + 18, deck.begin() + 18 + nb[st])});
                taken = true;
            }
            std::vector<float> lg;
            float v;
            net.forward(f, bk[i], st, strength(info, st, bk[i][st]), lg, v);
            auto p = net.policy(lg, L);
            const int act = std::discrete_distribution<int>(p.begin(), p.end())(rng);
            if (i != hero) hist.push_back({i, st, act, f, L});
            T.step(act);
        }
    }
    return out;
}

void part_ess_crn(const std::string& data, int nspots, int repeats) {
    Net net(data + "/models/ninemax/nm25");
    const NetInfo& info = net.info();
    auto sp = spots(net, nspots, 11);
    std::printf("ess\n");
    int fell = 0, n = 0;
    for (auto& s : sp) {
        SearchConfig cfg;
        cfg.samples = 64;
        for (double floor : {0.0, 1.01}) {
            cfg.min_ess = floor;
            auto r = ninemax_search(net, s.T, s.ctx, s.hist, s.hero, s.hole, s.board, cfg);
            std::array<int, 4> bk{0, 0, 0, 0};
            const int st = s.T.street(), nb[4] = {0, 3, 4, 5};
            std::vector<int> c{s.hole[0], s.hole[1]};
            c.insert(c.end(), s.board.begin(), s.board.end());
            for (int k = 0; k <= st; ++k) bk[k] = bucket_of(c.data(), 2 + nb[k]);
            std::vector<float> lg;
            float v;
            net.forward(features(s.T, info, &s.ctx), bk, st, strength(info, st, bk[st]), lg, v);
            auto p = net.policy(lg, s.T.legal());
            double gap = 0;
            int ones = 0;
            for (size_t a = 0; a < p.size(); ++a) gap = std::max(gap, std::fabs(double(r.policy[a] - p[a]))), ones += r.policy[a] == 1.0f;
            const bool ok = floor > 1 ? r.fell_back && gap < 1e-5 : !r.fell_back && ones == 1;
            fell += ok, ++n;
        }
        SearchConfig d;
        d.samples = 256;
        auto r = ninemax_search(net, s.T, s.ctx, s.hist, s.hero, s.hole, s.board, d);
        std::printf("    spot ess %.1f of 256: %s\n", r.ess, r.fell_back ? "below the floor, the network's policy" : "searched");
    }
    check(fell == n, "floor 1.01: the network's policy (fell_back); floor 0: one greedy move (" + std::to_string(fell) +
                         " of " + std::to_string(n) + ")");

    std::printf("crn\n");
    std::vector<double> v[2];
    for (auto& s : sp) {
        for (int crn = 0; crn < 2; ++crn) {
            std::vector<std::vector<double>> Q;
            for (int k = 0; k < repeats; ++k) {
                SearchConfig cfg;  // the same deals every time (seed), new rollout draws (rollout_seed)
                cfg.samples = 256, cfg.min_ess = 0, cfg.crn = crn, cfg.seed = 7, cfg.rollout_seed = 1000 + k;
                Q.push_back(ninemax_search(net, s.T, s.ctx, s.hist, s.hero, s.hole, s.board, cfg).q);
            }
            for (size_t a = 0; a < Q[0].size(); ++a)
                for (size_t b = a + 1; b < Q[0].size(); ++b) {
                    if (std::isnan(Q[0][a]) || std::isnan(Q[0][b])) continue;
                    double m = 0, m2 = 0;
                    for (auto& q : Q) m += q[a] - q[b], m2 += (q[a] - q[b]) * (q[a] - q[b]);
                    m /= repeats;
                    v[crn].push_back((m2 / repeats - m * m) * repeats / (repeats - 1));
                }
        }
    }
    auto mean = [](const std::vector<double>& x) { return std::accumulate(x.begin(), x.end(), 0.0) / x.size(); };
    std::vector<double> lr;
    for (size_t k = 0; k < v[0].size(); ++k) lr.push_back(std::log(std::max(v[1][k], 1e-12) / std::max(v[0][k], 1e-12)));
    std::printf("  %d spots x %d searches, 256 samples, %zu move pairs, same deals: var of Q(a) - Q(b) independent %.4f, "
                "crn %.4f bb^2 (ratio of means %.3f, per-pair geometric mean %.3f)\n",
                nspots, repeats, v[0].size(), mean(v[0]), mean(v[1]), mean(v[1]) / mean(v[0]), std::exp(mean(lr)));
}

}  // namespace

int main(int argc, char** argv) {
    std::string data;
    int nspots = 12, repeats = 16;
    for (int k = 1; k + 1 < argc; k += 2) {
        std::string a = argv[k], v = argv[k + 1];
        if (a == "--data") data = v;
        else if (a == "--spots") nspots = std::stoi(v);
        else if (a == "--repeats") repeats = std::stoi(v);
        else return std::fprintf(stderr, "unknown option %s\n", a.c_str()), 2;
    }
    if (data.empty()) return std::fprintf(stderr, "--data DIR is needed\n"), 2;
    buckets_init(data + "/blueprint/pack");
    part_bounty(data);
    part_ess_crn(data, nspots, repeats);
    std::printf(fails ? "FAIL: %d checks\n" : "all search checks pass\n", fails);
    return fails ? 1 : 0;
}
