// builds on its own. Suit-isomorphic indexing (hand-isomorphism, Kevin Waugh) per street into the pack's bucket
// and the pack's equal-population cut points (river_cuts.f32). 7-card evaluation: PokerHandEvaluator (Henry Lee).
#include "kestrel/buckets.hpp"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define _Bool bool
extern "C" {
#include "hand_index.h"
int evaluate_7cards(int a, int b, int c, int d, int e, int f, int g);
}

namespace kestrel {

namespace {

struct Tables {
    hand_indexer_t pre, flop, turn, river;
    std::vector<uint16_t> fb, tb, rb;
    std::vector<float> rcut;
};

std::vector<uint16_t> load(const std::string& dir, const char* street) {
    std::vector<uint16_t> out;
    std::string p16 = dir + "/" + street + "_buckets.u16", p8 = dir + "/" + street + "_buckets.u8";
    FILE* f = std::fopen(p16.c_str(), "rb");
    bool wide = f != nullptr;
    if (!f) f = std::fopen(p8.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    size_t n = size_t(std::ftell(f)) / (wide ? 2 : 1);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n);
    if (wide) {
        if (std::fread(out.data(), 2, n, f) != n) out.clear();
    } else {
        std::vector<uint8_t> b(n);
        if (std::fread(b.data(), 1, n, f) != n) out.clear();
        else std::copy(b.begin(), b.end(), out.begin());
    }
    std::fclose(f);
    return out;
}

Tables* T = nullptr;
std::mutex init_mu;

int river_from_hs(const uint8_t* c) {
    bool used[52] = {false};
    for (int k = 0; k < 7; ++k) used[c[k]] = true;
    int me = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    double win = 0;
    int n = 0;
    for (int a = 0; a < 52; ++a) {
        if (used[a]) continue;
        for (int b = a + 1; b < 52; ++b) {
            if (used[b]) continue;
            int op = evaluate_7cards(a, b, c[2], c[3], c[4], c[5], c[6]);
            win += me < op ? 1.0 : me == op ? 0.5 : 0.0;
            ++n;
        }
    }
    float hs = float(win / n);
    int bk = int(std::lower_bound(T->rcut.begin(), T->rcut.end(), hs) - T->rcut.begin());
    return std::min(bk, int(T->rcut.size()) - 1);
}

}

void buckets_init(const std::string& dir) {
    std::lock_guard<std::mutex> lk(init_mu);
    if (T) return;
    auto* t = new Tables();
    t->fb = load(dir, "flop"), t->tb = load(dir, "turn"), t->rb = load(dir, "river");
    if (t->rb.empty()) {
        if (FILE* f = std::fopen((dir + "/river_cuts.f32").c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            t->rcut.resize(size_t(std::ftell(f)) / 4);
            std::fseek(f, 0, SEEK_SET);
            if (std::fread(t->rcut.data(), 4, t->rcut.size(), f) != t->rcut.size()) t->rcut.clear();
            std::fclose(f);
        }
    }
    if (t->fb.empty() || t->tb.empty() || (t->rb.empty() && t->rcut.empty())) {
        delete t;
        throw std::runtime_error("no blueprint buckets in " + dir);
    }
    const uint8_t c1[] = {2}, c2[] = {2, 3}, c3[] = {2, 4}, c4[] = {2, 5};
    if (!hand_indexer_init(1, c1, &t->pre) || !hand_indexer_init(2, c2, &t->flop) ||
        !hand_indexer_init(2, c3, &t->turn) || !hand_indexer_init(2, c4, &t->river))
        throw std::runtime_error("hand indexer init failed");
    T = t;
}

int bucket_of(const int* cards, int n) {
    if (!T || (n != 2 && n != 5 && n != 6 && n != 7)) return -1;
    uint8_t c[7];
    bool seen[52] = {false};
    for (int i = 0; i < n; ++i) {
        if (cards[i] < 0 || cards[i] > 51 || seen[cards[i]]) return -1;
        seen[cards[i]] = true;
        c[i] = uint8_t(cards[i]);
    }
    if (n == 2) return int(hand_index_last(&T->pre, c));
    if (n == 5) return T->fb[hand_index_last(&T->flop, c)];
    if (n == 6) return T->tb[hand_index_last(&T->turn, c)];
    return T->rb.empty() ? river_from_hs(c) : T->rb[hand_index_last(&T->river, c)];
}

}
