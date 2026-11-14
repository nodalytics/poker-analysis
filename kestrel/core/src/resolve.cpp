// Safe river re-solving (Brown & Sandholm 2017, with the re-solve gadget of Burch, Johanson & Bowling 2014): a port
// of llm-research/blueprint/resolve.cpp onto Kestrel's blueprint, so it runs on the data pack too.
#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_map>

#include "kestrel/buckets.hpp"
#include "kestrel/headsup.hpp"

extern "C" int evaluate_7cards(int a, int b, int c, int d, int e, int f, int g);

namespace kestrel {

namespace {

constexpr int kThreads = 4;  // the root's subtrees in parallel: 4 threads were fastest on 64 cores (0.26 s vs 0.42)

struct Combos {
    int a[1326], b[1326];
    Combos() {
        int k = 0;
        for (int x = 0; x < 52; ++x)
            for (int y = x + 1; y < 52; ++y) a[k] = x, b[k] = y, ++k;
    }
};
const Combos& CB() {
    static const Combos c;
    return c;
}

struct RBoard {
    int n = 0;
    std::vector<int> hand, bucket, rank, order;
};

void compatible(const RBoard& b, const float* opp, float* out) {
    const auto& C = CB();
    double total = 0, per[52] = {0};
    for (int i = 0; i < b.n; ++i) {
        total += opp[i];
        per[C.a[b.hand[i]]] += opp[i];
        per[C.b[b.hand[i]]] += opp[i];
    }
    for (int i = 0; i < b.n; ++i)
        out[i] = float(total - per[C.a[b.hand[i]]] - per[C.b[b.hand[i]]] + opp[i]);
}

void showdown(const RBoard& b, const float* opp, float paid, float* out) {
    const auto& C = CB();
    double acc = 0, per[52] = {0};
    int i = b.n - 1;
    while (i >= 0) {
        int j = i;
        while (j >= 0 && b.rank[b.order[j]] == b.rank[b.order[i]]) --j;
        for (int k = i; k > j; --k) {
            int h = b.order[k];
            out[h] = float(acc - per[C.a[b.hand[h]]] - per[C.b[b.hand[h]]]);
        }
        for (int k = i; k > j; --k) {
            int h = b.order[k];
            acc += opp[h];
            per[C.a[b.hand[h]]] += opp[h];
            per[C.b[b.hand[h]]] += opp[h];
        }
        i = j;
    }
    acc = 0;
    std::fill(per, per + 52, 0.0);
    i = 0;
    while (i < b.n) {
        int j = i;
        while (j < b.n && b.rank[b.order[j]] == b.rank[b.order[i]]) ++j;
        for (int k = i; k < j; ++k) {
            int h = b.order[k];
            out[h] -= float(acc - per[C.a[b.hand[h]]] - per[C.b[b.hand[h]]]);
            out[h] *= paid;
        }
        for (int k = i; k < j; ++k) {
            int h = b.order[k];
            acc += opp[h];
            per[C.a[b.hand[h]]] += opp[h];
            per[C.b[b.hand[h]]] += opp[h];
        }
        i = j;
    }
}

}  // namespace

struct Resolver {
    const HUBlueprint& bp;
    const std::vector<HUBlueprint::Node>& T;
    RBoard b;
    double purify;
    std::unordered_map<int, size_t> at;
    std::vector<float> R, S;
    int us = 0;

    struct Scratch {
        std::vector<float> buf;
        size_t top = 0;
        float* get(size_t k) {
            float* p = buf.data() + top;
            top += k;
            if (top > buf.size()) throw std::runtime_error("re-solve scratch overflow");
            return p;
        }
    };
    std::vector<Scratch> scratch;  // one per thread: [0] the caller, [1 + k] OpenMP thread k

    // the current strategy (regret matching) at a node for every hand, into sg[a * n + i]
    void current(int id, float* sg) const {
        const int na = int(T[id].children.size()), n = b.n;
        const float* x = R.data() + at.at(id);
        for (int i = 0; i < n; ++i) {
            float tot = 0;
            for (int k = 0; k < na; ++k) tot += std::max(x[k * n + i], 0.0f);
            for (int a = 0; a < na; ++a) sg[a * n + i] = tot > 0 ? std::max(x[a * n + i], 0.0f) / tot : 1.0f / na;
        }
    }

    Resolver(const HUBlueprint& p, const std::vector<int>& board) : bp(p), T(p.nodes_), purify(p.purify_) {
        const auto& C = CB();
        bool used[52] = {false};
        for (int c : board) used[c] = true;
        for (int h = 0; h < 1326; ++h) {
            if (used[C.a[h]] || used[C.b[h]]) continue;
            int cs[7] = {C.a[h], C.b[h], board[0], board[1], board[2], board[3], board[4]};
            b.hand.push_back(h);
            b.bucket.push_back(bucket_of(cs, 7));
            b.rank.push_back(evaluate_7cards(cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6]));
        }
        b.n = int(b.hand.size());
        b.order.resize(b.n);
        for (int i = 0; i < b.n; ++i) b.order[i] = i;
        std::sort(b.order.begin(), b.order.end(), [&](int x, int y) { return b.rank[x] < b.rank[y]; });
    }

    // the blueprint's strategy at a river node for hand i, normalised and purified (as resolve.cpp's table)
    float sig(int id, int i, int a) const {
        const auto& nd = T[id];
        const int na = int(nd.children.size());
        uint64_t row = nd.offset + uint64_t(b.bucket[i]) * na;
        double x[16], sum = 0;
        for (int k = 0; k < na; ++k) sum += x[k] = bp.av(row + k);
        for (int k = 0; k < na; ++k) x[k] = sum > 0 ? x[k] / sum : 1.0 / na;
        double keep = 0;
        for (int k = 0; k < na; ++k) keep += x[k] >= purify ? x[k] : 0;
        if (keep > 0)
            for (int k = 0; k < na; ++k) x[k] = x[k] >= purify ? x[k] / keep : 0;
        return float(x[a]);
    }

    float strat(bool sub, bool avg, int id, int i, int a) const {
        if (!sub) return sig(id, i, a);
        const int na = int(T[id].children.size()), n = b.n;
        const float* x = (avg ? S.data() : R.data()) + at.at(id);
        float tot = 0;
        for (int k = 0; k < na; ++k) tot += std::max(x[k * n + i], 0.0f);
        if (tot <= 0) return avg ? sig(id, i, a) : 1.0f / na;
        return std::max(x[a * n + i], 0.0f) / tot;
    }

    void build(int root) {
        size_t total = 0;
        std::vector<int> stack{root};
        while (!stack.empty()) {
            int id = stack.back();
            stack.pop_back();
            if (T[id].kind != 0) continue;
            at[id] = total;
            total += T[id].children.size() * size_t(b.n);
            for (int c : T[id].children) stack.push_back(c);
        }
        R.assign(total, 0.0f);
        S.assign(total, 0.0f);
        int threads = 1;
#ifdef _OPENMP
        threads += std::min(kThreads, omp_get_max_threads());
#endif
        scratch.assign(threads, Scratch{});
        for (auto& sc : scratch) sc.buf.resize(size_t(1326) * 80 * 64);
    }

    void terminal(int id, int trav, const float* ro, float* out) const {
        const auto& nd = T[id];
        if (nd.kind == 1) {
            float paid = float(nd.folder ? nd.c[1] : nd.c[0]);
            compatible(b, ro, out);
            float sign = nd.folder == trav ? -1.0f : 1.0f;
            for (int i = 0; i < b.n; ++i) out[i] *= sign * paid;
        } else {
            showdown(b, ro, float(std::min(nd.c[0], nd.c[1])), out);
        }
    }

    // CFR+ for player trav. At the root (par) the actions' subtrees run in parallel; they touch disjoint regrets and
    // their values are summed in action order, so the result is the serial solve's
    void cfr(int id, int trav, const float* rt, const float* ro, float* out, float weight, Scratch& ar, bool par) {
        const auto& nd = T[id];
        const int n = b.n;
        if (nd.kind != 0) return terminal(id, trav, ro, out);
        const int na = int(nd.children.size());
        const size_t mark = ar.top;
        const bool mine = nd.player == trav;
        float* sg = ar.get(size_t(na) * n);
        current(id, sg);
        float* vals = ar.get(size_t(na) * n);
#ifdef _OPENMP
        if (par && na > 1) {
            float* reach = ar.get(size_t(na) * n);
            for (int a = 0; a < na; ++a)
                for (int i = 0; i < n; ++i) reach[size_t(a) * n + i] = (mine ? rt[i] : ro[i]) * sg[size_t(a) * n + i];
#pragma omp parallel for schedule(dynamic, 1) num_threads(std::min(kThreads, omp_get_max_threads()))
            for (int a = 0; a < na; ++a) {
                const float* r = reach + size_t(a) * n;
                cfr(nd.children[a], trav, mine ? r : rt, mine ? ro : r, vals + size_t(a) * n, weight,
                    scratch[1 + omp_get_thread_num()], false);
            }
        } else
#endif
        {
            float* reach = ar.get(n);
            for (int a = 0; a < na; ++a) {
                for (int i = 0; i < n; ++i) reach[i] = (mine ? rt[i] : ro[i]) * sg[size_t(a) * n + i];
                cfr(nd.children[a], trav, mine ? reach : rt, mine ? ro : reach, vals + size_t(a) * n, weight, ar,
                    false);
            }
        }
        std::fill(out, out + n, 0.0f);
        if (mine) {
            const size_t off = at.at(id);
            float* Rn = R.data() + off;
            float* Sn = S.data() + off;
            for (int a = 0; a < na; ++a)
                for (int i = 0; i < n; ++i) out[i] += sg[size_t(a) * n + i] * vals[size_t(a) * n + i];
            for (int a = 0; a < na; ++a)
                for (int i = 0; i < n; ++i) {
                    Rn[a * n + i] = std::max(Rn[a * n + i] + vals[size_t(a) * n + i] - out[i], 0.0f);
                    Sn[a * n + i] += weight * rt[i] * sg[size_t(a) * n + i];
                }
        } else {
            for (int a = 0; a < na; ++a)
                for (int i = 0; i < n; ++i) out[i] += vals[size_t(a) * n + i];
        }
        ar.top = mark;
    }

    void best_response(bool sub, int id, int br, const float* ro, float* out) const {
        const auto& nd = T[id];
        const int n = b.n;
        if (nd.kind != 0) return terminal(id, br, ro, out);
        const int na = int(nd.children.size());
        std::vector<float> child(n), reach(n);
        if (nd.player == br) {
            std::fill(out, out + n, -1e30f);
            for (int a = 0; a < na; ++a) {
                best_response(sub, nd.children[a], br, ro, child.data());
                for (int i = 0; i < n; ++i) out[i] = std::max(out[i], child[i]);
            }
        } else {
            std::fill(out, out + n, 0.0f);
            for (int a = 0; a < na; ++a) {
                for (int i = 0; i < n; ++i) reach[i] = ro[i] * strat(sub, true, id, i, a);
                best_response(sub, nd.children[a], br, reach.data(), child.data());
                for (int i = 0; i < n; ++i) out[i] += child[i];
            }
        }
    }

    void solve(int root, const float* r_us, const float* r_opp, const float* cbv, int iters, float floor) {
        const int n = b.n, opp = 1 - us;
        build(root);
        std::vector<float> g(2 * size_t(n), 0.0f), ro(n), v(n), rg(r_opp, r_opp + n);
        float top = *std::max_element(rg.begin(), rg.end());
        for (auto& x : rg) x = std::max(x, floor * top);
        auto enter = [&](int i) {
            float e = std::max(g[i], 0.0f), q = std::max(g[n + i], 0.0f);
            return e + q > 0 ? e / (e + q) : 0.5f;
        };
        for (int t = 1; t <= iters; ++t) {
            for (int i = 0; i < n; ++i) ro[i] = rg[i] * enter(i);
            cfr(root, us, r_us, ro.data(), v.data(), float(t), scratch[0], true);
            cfr(root, opp, rg.data(), r_us, v.data(), float(t), scratch[0], true);
            for (int i = 0; i < n; ++i) {
                float e = enter(i), mix = e * v[i] + (1 - e) * cbv[i];
                g[i] = std::max(g[i] + v[i] - mix, 0.0f);
                g[n + i] = std::max(g[n + i] + cbv[i] - mix, 0.0f);
            }
        }
    }
};

bool HUBlueprint::river_resolve(int root, int target, const std::vector<int>& board, const std::vector<double>& r0,
                                const std::vector<double>& r1, std::vector<float>& out) const {
    if (board.size() != 5) return false;
    for (int id : {root, target})
        if (id < 0 || id >= int(nodes_.size()) || nodes_[id].kind != 0 || nodes_[id].street != 3) return false;
    Resolver rs(*this, board);
    const int n = rs.b.n;
    if (n == 0) return false;
    for (int i = 0; i < n; ++i)
        if (rs.b.bucket[i] < 0) return false;
    rs.us = nodes_[target].player;
    std::vector<float> r[2] = {std::vector<float>(n), std::vector<float>(n)};
    for (int i = 0; i < n; ++i) r[0][i] = float(r0[rs.b.hand[i]]), r[1][i] = float(r1[rs.b.hand[i]]);
    if (*std::max_element(r[1 - rs.us].begin(), r[1 - rs.us].end()) <= 0) return false;
    std::vector<float> cbv(n);
    rs.best_response(false, root, 1 - rs.us, r[rs.us].data(), cbv.data());
    rs.solve(root, r[rs.us].data(), r[1 - rs.us].data(), cbv.data(), resolve_iters_, float(min_reach_));
    if (!rs.at.count(target)) return false;
    out.assign(1326 * 6, 0.0f);
    const int na = int(nodes_[target].children.size());
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < na; ++a) out[size_t(rs.b.hand[i]) * 6 + a] = rs.strat(true, true, target, i, a);
    return true;
}

}  // namespace kestrel
