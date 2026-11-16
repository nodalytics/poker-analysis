#include "kestrel/ninemax_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

#include "kestrel/buckets.hpp"
#include "kestrel/icm.hpp"

extern "C" int evaluate_7cards(int a, int b, int c, int d, int e, int f, int g);

namespace kestrel {

namespace {

constexpr int kNet = 0, kFold = 1, kCall = 2, kRaise = 3;

// the network's probabilities tilted towards one kind of move (fold slot 0, call slot 1, raises 2+), renormalised
void tilt(float* p, int A, int kind, double factor) {
    if (kind == kNet) return;
    double z = 0;
    for (int a = 0; a < A; ++a) {
        bool hit = (kind == kFold && a == 0) || (kind == kCall && a == 1) || (kind == kRaise && a >= 2);
        if (hit) p[a] = float(p[a] * factor);
        z += p[a];
    }
    if (z > 0)
        for (int a = 0; a < A; ++a) p[a] = float(p[a] / z);
}

// the policy from logits and the legal mask, in place
void softmax_legal(float* lg, const uint8_t* legal, int A) {
    float mx = -INFINITY;
    for (int a = 0; a < A; ++a)
        if (legal[a]) mx = std::max(mx, lg[a]);
    double z = 0;
    for (int a = 0; a < A; ++a) z += lg[a] = legal[a] ? std::exp(lg[a] - mx) : 0.0f;
    for (int a = 0; a < A; ++a) lg[a] = float(lg[a] / z);
}

struct Deal {
    std::vector<std::array<int, 4>> bk;  // every seat's buckets, pre-flop .. river
    std::vector<int> rank;               // every seat's showdown rank (lower wins)
};

}  // namespace

SearchResult ninemax_search(const Net& net, const Table& T, const TourneyCtx& ctx, const std::vector<OppDecision>& hist,
                            int hero, const std::array<int, 2>& hole, const std::vector<int>& board,
                            const SearchConfig& cfg) {
    const NetInfo& info = net.info();
    const int P = T.P(), A = T.menu().A, N = std::clamp(cfg.samples, 16, 4096), nf = info.nfeat;
    std::mt19937_64 rng(cfg.seed);
    SearchResult res;
    res.samples = N;

    // 1. deals: the opponents' cards and the rest of the board, everything else known
    std::vector<int> deck;
    for (int c = 0; c < 52; ++c)
        if (c != hole[0] && c != hole[1] && std::find(board.begin(), board.end(), c) == board.end()) deck.push_back(c);
    std::vector<Deal> deals(N);
    for (int n = 0; n < N; ++n) {
        std::vector<int> d = deck;
        std::shuffle(d.begin(), d.end(), rng);
        size_t at = 0;
        std::array<int, 5> full{};
        for (size_t k = 0; k < 5; ++k) full[k] = k < board.size() ? board[k] : d[at++];
        Deal& D = deals[n];
        D.bk.assign(P, {0, 0, 0, 0});
        D.rank.assign(P, std::numeric_limits<int>::max());
        for (int s = 0; s < P; ++s) {
            if (!T.seated[s]) continue;
            int c[7];
            if (s == hero) c[0] = hole[0], c[1] = hole[1];
            else c[0] = d[at++], c[1] = d[at++];
            for (int k = 0; k < 5; ++k) c[2 + k] = full[k];
            D.bk[s] = {bucket_of(c, 2), bucket_of(c, 5), bucket_of(c, 6), bucket_of(c, 7)};
            D.rank[s] = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
        }
    }

    // 2. weights: how likely the network was to take each opponent's actual moves with that deal's cards
    std::vector<double> logw(N, 0.0);
    {
        std::vector<float> F(size_t(N) * nf), st(N), lg(size_t(N) * A);
        std::vector<int> bk(size_t(N) * 4), street(N);
        for (auto& h : hist) {
            for (int n = 0; n < N; ++n) {
                std::copy(h.feat.begin(), h.feat.end(), F.begin() + size_t(n) * nf);
                for (int k = 0; k < 4; ++k) bk[n * 4 + k] = deals[n].bk[h.seat][k];
                street[n] = h.street, st[n] = strength(info, h.street, deals[n].bk[h.seat][h.street]);
            }
            net.forward_batch(N, F.data(), bk.data(), street.data(), st.data(), lg.data(), nullptr);
            for (int n = 0; n < N; ++n) {
                float* p = lg.data() + size_t(n) * A;
                softmax_legal(p, h.legal.data(), A);
                logw[n] += std::log(std::max(double(p[h.slot]), 1e-12));
            }
        }
    }
    const double mx = *std::max_element(logw.begin(), logw.end());
    std::vector<double> w(N);
    double z = 0, z2 = 0;
    for (int n = 0; n < N; ++n) z += w[n] = std::exp(logw[n] - mx);
    for (auto& x : w) x /= z, z2 += x * x;
    res.ess = 1.0 / z2;

    // 3. every legal move on every deal (four times with continuation strategies), played out in lockstep
    std::vector<float> targets;
    const auto L = T.legal(&targets);
    std::vector<int> moves;
    for (int a = 0; a < A; ++a)
        if (L[a]) moves.push_back(a);
    const int C = cfg.robust ? 4 : 1, M = int(moves.size());
    const size_t R = size_t(M) * C * N;  // row r: move r / (C N), continuation (r / N) % C, deal r % N
    std::vector<Table> tabs(R, T);
    for (size_t r = 0; r < R; ++r) tabs[r].step(moves[r / (size_t(C) * N)]);
    double chips = 0;
    for (int s = 0; s < P; ++s) chips += double(T.stack[s]);
    const bool tourney = !ctx.pay.empty();
    const std::vector<double> eq0 = tourney ? icm(std::vector<double>(T.stack.begin(), T.stack.end()), ctx.pay)
                                            : std::vector<double>(P, 0.0);
    constexpr size_t kChunk = 2048;
    std::vector<size_t> live;
    std::vector<float> F, st, lg;
    std::vector<int> bk, street;
    std::uniform_real_distribution<double> u01(0, 1);
    for (int guard = 0; guard < 200; ++guard) {
        live.clear();
        for (size_t r = 0; r < R; ++r)
            if (!tabs[r].done()) live.push_back(r);
        if (live.empty()) break;
        for (size_t c0 = 0; c0 < live.size(); c0 += kChunk) {
            const int B = int(std::min(kChunk, live.size() - c0));
            F.resize(size_t(B) * nf), st.resize(B), lg.resize(size_t(B) * A), bk.resize(size_t(B) * 4), street.resize(B);
            std::vector<std::vector<uint8_t>> legal(B);
#pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < B; ++i) {
                const Table& U = tabs[live[c0 + i]];
                const Deal& D = deals[live[c0 + i] % N];
                const int s = U.actor();
                auto f = features(U, info, tourney ? &ctx : nullptr);
                std::copy(f.begin(), f.end(), F.begin() + size_t(i) * nf);
                legal[i] = U.legal();
                for (int k = 0; k < 4; ++k) bk[i * 4 + k] = D.bk[s][k];
                street[i] = U.street(), st[i] = strength(info, U.street(), D.bk[s][U.street()]);
            }
            net.forward_batch(B, F.data(), bk.data(), street.data(), st.data(), lg.data(), nullptr);
            for (int i = 0; i < B; ++i) {
                const size_t r = live[c0 + i];
                Table& U = tabs[r];
                float* p = lg.data() + size_t(i) * A;
                softmax_legal(p, legal[i].data(), A);
                if (U.actor() != hero) tilt(p, A, int((r / N) % C), cfg.bias);
                double x = u01(rng), acc = 0;
                int pick = 1;
                for (int a = 0; a < A; ++a)
                    if (legal[i][a] && (acc += p[a]) >= x) {
                        pick = a;
                        break;
                    }
                if (!legal[i][pick])
                    for (int a = 0; a < A; ++a)
                        if (legal[i][a]) pick = a;
                U.step(pick);
            }
        }
    }

    // 4. each move's value: the weighted mean result over the deals (the worst continuation, robust)
    res.q.assign(A, std::numeric_limits<double>::quiet_NaN());
    for (int m = 0; m < M; ++m) {
        double worst = std::numeric_limits<double>::infinity();
        for (int c = 0; c < C; ++c) {
            double v = 0;
            for (int n = 0; n < N; ++n) {
                const Table& U = tabs[(size_t(m) * C + c) * N + n];
                auto pay = U.payoffs(deals[n].rank);
                double r;
                if (tourney) {
                    auto eq1 = icm(end_stacks(std::vector<double>(T.stack.begin(), T.stack.end()), pay), ctx.pay);
                    r = (eq1[hero] - eq0[hero]) * chips / double(BB);
                } else {
                    r = pay[hero] / double(BB);
                }
                v += w[n] * r;
            }
            worst = std::min(worst, v);
        }
        res.q[moves[m]] = worst;
    }
    int best = moves[0];
    for (int a : moves)
        if (res.q[a] > res.q[best]) best = a;
    res.policy.assign(A, 0.0f);
    res.policy[best] = 1.0f;  // greedy: the tested version
    return res;
}

}  // namespace kestrel
