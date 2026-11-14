#include "kestrel/luck.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "kestrel/codec_acpc.hpp"
#include "kestrel/headsup.hpp"

extern "C" int evaluate_7cards(int a, int b, int c, int d, int e, int f, int g);

namespace kestrel {

double equity(const std::array<int, 2>& h, const std::array<int, 2>& v, const std::vector<int>& board) {
    bool used[52] = {false};
    used[h[0]] = used[h[1]] = used[v[0]] = used[v[1]] = true;
    for (int c : board) used[c] = true;
    std::vector<int> deck;
    for (int c = 0; c < 52; ++c)
        if (!used[c]) deck.push_back(c);
    int b[5];
    for (size_t k = 0; k < board.size(); ++k) b[k] = board[k];
    const int need = 5 - int(board.size()), n = int(deck.size());
    double win = 0;
    long count = 0;
    if (need == 5) {  // pre-flop: 20,000 random boards (sd of the estimate ~0.0035), seeded by the hands
        std::mt19937_64 rng(uint64_t(h[0]) * 1000003 + uint64_t(h[1]) * 10007 + uint64_t(v[0]) * 101 + uint64_t(v[1]));
        for (int s = 0; s < 20000; ++s) {
            for (int k = 0; k < 5; ++k) {
                std::uniform_int_distribution<int> pick(k, n - 1);
                std::swap(deck[k], deck[pick(rng)]);
                b[k] = deck[k];
            }
            int me = evaluate_7cards(h[0], h[1], b[0], b[1], b[2], b[3], b[4]);
            int op = evaluate_7cards(v[0], v[1], b[0], b[1], b[2], b[3], b[4]);
            win += me < op ? 1.0 : me == op ? 0.5 : 0.0;
        }
        return win / 20000.0;
    }
    auto score = [&]() {
        int me = evaluate_7cards(h[0], h[1], b[0], b[1], b[2], b[3], b[4]);
        int op = evaluate_7cards(v[0], v[1], b[0], b[1], b[2], b[3], b[4]);
        win += me < op ? 1.0 : me == op ? 0.5 : 0.0;
        ++count;
    };
    // every runout: choose `need` cards from the deck in increasing order
    std::vector<int> idx(need);
    for (int k = 0; k < need; ++k) idx[k] = k;
    if (need == 0) {
        score();
        return win;
    }
    while (true) {
        for (int k = 0; k < need; ++k) b[5 - need + k] = deck[idx[k]];
        score();
        int k = need - 1;
        while (k >= 0 && idx[k] == n - need + k) --k;
        if (k < 0) break;
        ++idx[k];
        for (int j = k + 1; j < need; ++j) idx[j] = idx[j - 1] + 1;
    }
    return win / double(count);
}

bool luck_hand(const nlohmann::json& r, int64_t big, int64_t stack, LuckHand& out) {
    if (big <= 0) return false;
    auto theirs = r.contains("their_hole") ? r["their_hole"] : r.value("slumbot_hole", nlohmann::json());
    if (!theirs.is_array() || theirs.size() != 2 || !r.contains("hole") || !r["hole"].is_array()) return false;
    std::array<int, 2> me{card_from(r["hole"][0]), card_from(r["hole"][1])};
    std::array<int, 2> op{card_from(theirs[0]), card_from(theirs[1])};
    std::vector<int> board;
    if (r.contains("board") && r["board"].is_array())
        for (auto& c : r["board"]) board.push_back(card_from(c));
    if (board.size() > 5 || !r.contains("pos") || !r["pos"].is_number_integer()) return false;
    out.pos = r["pos"].get<int>();
    if (out.pos != 0 && out.pos != 1) return false;
    {  // every card at most once
        bool seen[52] = {false};
        for (int c : {me[0], me[1], op[0], op[1]}) {
            if (seen[c]) return false;
            seen[c] = true;
        }
        for (int c : board) {
            if (seen[c]) return false;
            seen[c] = true;
        }
    }
    out.raw = r.contains("result_bb") ? r["result_bb"].get<double>() : r["winnings_chips"].get<double>() / double(big);
    out.adjusted = out.raw;
    out.pre = equity(me, op, {});
    out.board = 0;
    out.allin = false;
    // chips in per position and the street betting ended on (ACPC positions: 0 big blind, 1 small blind)
    auto st = acpc::tokens(r.value("action", ""));
    if (st.size() > 4) return false;
    int64_t total[2] = {0, 0};
    std::vector<int64_t> pots;
    for (size_t si = 0; si < st.size(); ++si) {
        pots.push_back(total[0] + total[1]);
        int64_t street[2] = {si == 0 ? big : 0, si == 0 ? big / 2 : 0}, last = si == 0 ? big : 0;
        int actor = si == 0 ? 1 : 0;
        for (auto& t : st[si]) {
            if (t.kind == 'b') last = t.to, street[actor] = t.to;
            else if (t.kind == 'c') street[actor] = last;
            actor = 1 - actor;
        }
        total[0] += street[0], total[1] += street[1];
    }
    int last_street = 0;
    for (size_t si = 0; si < st.size(); ++si)
        if (!st[si].empty()) last_street = int(si);
    const int at_street[4] = {0, 3, 4, 5};
    if (total[0] >= stack && total[1] >= stack) {
        size_t nb = std::min<size_t>(board.size(), at_street[last_street]);
        if (nb < 5) {
            std::vector<int> b(board.begin(), board.begin() + nb);
            double eq = equity(me, op, b);
            out.adjusted = (eq * double(total[0] + total[1]) - double(total[out.pos])) / double(big);
            out.allin = true;
        }
    }
    if (!out.allin) {
        double prev = out.pre;
        for (int si = 1; si <= 3; ++si) {
            size_t n = at_street[si];
            if (board.size() < n || size_t(si) >= pots.size()) break;
            double eq = equity(me, op, std::vector<int>(board.begin(), board.begin() + n));
            out.board += double(pots[si]) * (eq - prev) / double(big);
            prev = eq;
        }
    }
    return true;
}

nlohmann::json LuckEstimate::to_json() const {
    auto iv = [](double m, double se) { return nlohmann::json::array({m - 1.96 * se, m + 1.96 * se}); };
    return {{"hands", hands},
            {"allin_hands", allin},
            {"raw_bb100", raw},
            {"raw_ci95", iv(raw, raw_se)},
            {"adjusted_bb100", adjusted},
            {"adjusted_ci95", iv(adjusted, adjusted_se)},
            {"variance_reduction", variance_reduction}};
}

namespace {

double mean(const std::vector<double>& x) {
    double s = 0;
    for (double v : x) s += v;
    return x.empty() ? 0 : s / double(x.size());
}
double var(const std::vector<double>& x) {
    if (x.size() < 2) return 0;
    double m = mean(x), s = 0;
    for (double v : x) s += (v - m) * (v - m);
    return s / double(x.size() - 1);
}

// y - b*x, with b fitted on the other fold (by group, if groups are given): x has mean zero by construction, so the
// estimate stays unbiased
std::vector<double> cross_fit(const std::vector<double>& y, const std::vector<double>& x, const std::vector<int>& fold,
                              const std::vector<int>& group) {
    std::vector<double> out = y;
    for (int f = 0; f < 2; ++f)
        for (int g = 0; g < 2; ++g) {
            double ym = 0, n = 0;
            for (size_t k = 0; k < y.size(); ++k)
                if (fold[k] != f && group[k] == g) ym += y[k], ++n;
            ym = n > 0 ? ym / n : 0;
            double sxy = 0, sxx = 0;
            for (size_t k = 0; k < y.size(); ++k)
                if (fold[k] != f && group[k] == g) sxy += x[k] * (y[k] - ym), sxx += x[k] * x[k];
            double b = sxy / std::max(sxx, 1e-9);
            for (size_t k = 0; k < y.size(); ++k)
                if (fold[k] == f && group[k] == g) out[k] = y[k] - b * x[k];
        }
    return out;
}

}  // namespace

LuckEstimate luck_estimate(const std::vector<LuckHand>& hands, uint64_t seed) {
    LuckEstimate e;
    const size_t n = hands.size();
    e.hands = int(n);
    if (n < 2) return e;
    std::mt19937_64 rng(seed);
    std::vector<int> fold(n), pos(n), one(n, 0);
    std::vector<double> raw(n), adj(n), pre(n), board(n);
    for (size_t k = 0; k < n; ++k) {
        fold[k] = int(rng() & 1);
        pos[k] = hands[k].pos & 1;
        raw[k] = hands[k].raw, adj[k] = hands[k].adjusted, pre[k] = hands[k].pre - 0.5, board[k] = hands[k].board;
        e.allin += hands[k].allin;
    }
    auto cards = cross_fit(adj, pre, fold, pos);
    auto full = cross_fit(cards, board, fold, one);
    e.raw = 100 * mean(raw), e.raw_se = 100 * std::sqrt(var(raw) / double(n));
    e.adjusted = 100 * mean(full), e.adjusted_se = 100 * std::sqrt(var(full) / double(n));
    e.variance_reduction = var(full) > 0 ? var(raw) / var(full) : 1;
    return e;
}

}  // namespace kestrel
