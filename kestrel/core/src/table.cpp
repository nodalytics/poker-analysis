#include "kestrel/table.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace kestrel {

namespace {
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
}

Menu Menu::named(const std::string& name) {
    Menu m;
    auto bb = [](std::vector<float> v) {
        for (auto& x : v) x *= BB;
        return v;
    };
    if (name == "lean") {
        m.opens = bb({2.5f}), m.rer = {{3.0f}, {2.3f}}, m.max_pre = 3;
        m.hb = {0.5f, 1.0f}, m.hr = {1.0f}, m.mb = {0.66f}, m.mr = {}, m.max_post = 2;
    } else if (name == "medium") {
        m.opens = bb({2.5f}), m.rer = {{3.0f}, {2.3f}}, m.max_pre = 4;
        m.hb = {0.33f, 0.75f, 1.5f}, m.hr = {1.0f}, m.mb = {0.5f, 1.0f}, m.mr = {1.0f}, m.max_post = 3;
    } else if (name == "mtt") {
        m.opens = bb({2.0f, 2.5f, 3.0f}), m.rer = {{3.0f}, {2.3f}}, m.max_pre = 4;
        m.hb = {0.33f, 0.75f, 1.5f}, m.hr = {1.0f}, m.mb = {0.5f, 1.0f}, m.mr = {1.0f}, m.max_post = 3;
    } else if (name == "wide") {
        m.opens = bb({2.0f, 2.5f, 3.5f}), m.rer = {{2.5f, 3.5f}, {2.2f, 3.0f}}, m.max_pre = 4;
        m.hb = {0.25f, 0.5f, 0.75f, 1.25f, 2.0f}, m.hr = {0.75f, 1.5f}, m.mb = {0.33f, 0.75f}, m.mr = {1.0f};
        m.max_post = 3;
    } else {
        throw std::invalid_argument("unknown menu: " + name);
    }
    m.finish();
    return m;
}

void Menu::finish() {
    size_t s = std::max({opens.size(), hb.size(), hr.size(), mb.size(), mr.size()});
    for (auto& r : rer) s = std::max(s, r.size());
    S = int(s);
    A = S + 3;
}

Table::Table(const std::vector<int64_t>& stacks, const Menu& menu, const Rules& rules)
    : stack(stacks), P_(int(stacks.size())), menu_(menu), rules_(rules) {
    if (P_ < 2) throw std::invalid_argument("at least two seats");
    seated.assign(P_, 0);
    int n = 0;
    for (int j = 0; j < P_; ++j) {
        seated[j] = stack[j] > 0;
        n += seated[j];
        if (seated[j] && stack[j] < 2 * BB && !rules_.short_stacks) throw std::invalid_argument("every stack must be at least 2bb (0: empty)");
    }
    if (!seated[0] || !seated[1]) throw std::invalid_argument("the blinds must be seated");
    hu = n == 2;
    contrib.assign(P_, 0);
    contrib[0] = std::min<int64_t>(SB, stack[0]), contrib[1] = std::min<int64_t>(BB, stack[1]);
    folded.resize(P_);
    for (int j = 0; j < P_; ++j) folded[j] = !seated[j];
    allin.assign(P_, 0);
    acted.assign(P_, 0);
    dead = std::max<int64_t>(0, std::min<int64_t>(stack[1] - BB, int64_t(std::llround(rules_.ante_bb * BB))));
    avail = stack;
    avail[1] -= dead;
    allin[0] = avail[0] <= contrib[0];
    allin[1] = avail[1] <= BB;
    advance();
}

std::vector<uint8_t> Table::needs() const {
    std::vector<uint8_t> r(P_);
    for (int j = 0; j < P_; ++j) r[j] = !folded[j] && !allin[j] && (!acted[j] || contrib[j] < bet);
    return r;
}

void Table::advance() {
    for (int it = 0; it < 5; ++it) {
        if (done_) {
            actor_ = -1;
            return;
        }
        int n_live = 0, can_act = 0;
        for (int j = 0; j < P_; ++j) {
            n_live += !folded[j];
            can_act += !folded[j] && !allin[j];
        }
        bool alone = n_live <= 1;
        actor_ = -1;
        if (!alone) {
            auto need = needs();
            for (int k = 0; k < P_; ++k) {
                int j = (to_act_ + 1 + k) % P_;
                if (need[j]) {
                    actor_ = j;
                    return;
                }
            }
        }
        bool finish = n_live <= 1 || street_ == 3 || can_act <= 1;
        if (finish) {
            showdown_ = showdown_ || n_live > 1;
            done_ = true;
            return;
        }
        street_ += 1;
        raises = 0;
        last_raise = BB;
        std::fill(acted.begin(), acted.end(), 0);
        to_act_ = hu ? 0 : P_ - 1;
    }
}

std::vector<float> Table::targets() const {
    const Menu& m = menu_;
    std::vector<float> t(m.S, NaN);
    int i = actor_;
    int64_t pot = std::accumulate(contrib.begin(), contrib.end(), int64_t(0)) + dead;
    int64_t to_call = bet - contrib[i];
    int live = 0;
    for (int j = 0; j < P_; ++j) live += !folded[j];
    bool multi = live > 2, unopened = raises == 0;
    if (street_ == 0) {
        int lvl = std::clamp(raises - 1, 0, int(m.rer.size()) - 1);
        for (int k = 0; k < m.S; ++k) {
            float x = NaN;
            if (unopened) {
                if (k < int(m.opens.size())) x = m.opens[k];
            } else if (k < int(m.rer[lvl].size())) {
                x = m.rer[lvl][k] * float(bet);
            }
            t[k] = std::floor(x);
        }
    } else {
        const auto& fr = multi ? (unopened ? m.mb : m.mr) : (unopened ? m.hb : m.hr);
        for (int k = 0; k < m.S; ++k)
            if (k < int(fr.size())) t[k] = std::floor(float(bet) + fr[k] * float(pot + to_call));
    }
    return t;
}

std::vector<uint8_t> Table::legal(std::vector<float>* tout) const {
    const Menu& m = menu_;
    std::vector<uint8_t> L(m.A, 0);
    if (actor_ < 0) return L;
    int i = actor_;
    int64_t to_call = bet - contrib[i];
    auto t = targets();
    int cap = street_ == 0 ? m.max_pre : m.max_post;
    bool reopened = !acted[i];
    bool others = false;
    for (int j = 0; j < P_; ++j)
        if (j != i && !folded[j] && !allin[j] && avail[j] > bet) others = true;
    reopened = reopened && others;
    int64_t stk = avail[i];
    bool any_size = false;
    for (int k = 0; k < m.S; ++k) {
        bool ok = !std::isnan(t[k]) && (t[k] - float(bet) >= float(last_raise)) && (t[k] < float(stk)) &&
                  (raises < cap) && reopened;
        L[2 + k] = ok;
        any_size = any_size || ok;
    }
    L[0] = to_call > 0;
    bool call = true;
    bool shove = (stk > bet) && reopened;
    if (!rules_.open_limp) {
        bool first_in = street_ == 0 && pre_raises == 0 && bet == BB && i >= 2;
        call = call && !(first_in && (any_size || shove));
    }
    L[1] = call;
    L[m.A - 1] = shove;
    if (tout) {
        tout->assign(m.S, 0.0f);
        for (int k = 0; k < m.S; ++k) (*tout)[k] = std::isnan(t[k]) ? 0.0f : t[k];
    }
    return L;
}

void Table::step(int a) {
    std::vector<float> t;
    auto L = legal(&t);
    if (actor_ < 0 || a < 0 || a >= menu_.A || !L[a]) throw std::invalid_argument("illegal action");
    int i = actor_, S = menu_.S;
    bool fold = a == 0, call = a == 1, size = a >= 2 && a < 2 + S;
    int64_t stk = avail[i];
    int64_t total = size ? int64_t(t[a - 2]) : stk;
    if (call) total = std::min(bet, stk);
    if (fold) total = contrib[i];
    commit(i, fold, call, total);
}

void Table::step_to(int64_t total) {
    if (actor_ < 0) throw std::invalid_argument("the hand is over");
    const int i = actor_;
    total = std::min(total, avail[i]);
    if (total <= bet) return step(1);
    commit(i, false, false, total);
}

void Table::commit(int i, bool fold, bool call, int64_t total) {
    int64_t stk = avail[i];
    int64_t inc = total - bet;
    bool full = !fold && !call && inc >= last_raise;
    if (fold) folded[i] = 1;
    contrib[i] = total;
    if (total >= stk) allin[i] = 1;
    if (full)
        for (int j = 0; j < P_; ++j)
            if (j != i) acted[j] = 0;
    acted[i] = 1;
    if (full) last_raise = inc;
    raises += full;
    pre_raises += full && street_ == 0;
    if (full || (inc > 0 && !fold && !call)) aggressor = i;
    bet = std::max(bet, total);
    to_act_ = i;
    advance();
}

std::vector<double> Table::payoffs(const std::vector<int>& ranks) const {
    std::vector<double> c(contrib.begin(), contrib.end()), win(P_, 0.0);
    std::vector<int64_t> levels(contrib);
    std::sort(levels.begin(), levels.end());
    const int big = std::numeric_limits<int>::max();
    double prev = 0;
    for (int k = 0; k < P_; ++k) {
        double L = double(levels[k]), pot = 0;
        for (int j = 0; j < P_; ++j) pot += std::min(c[j], L) - std::min(c[j], prev);
        int best = big, nw = 0;
        for (int j = 0; j < P_; ++j)
            if (!folded[j] && c[j] >= L && pot > 0) best = std::min(best, ranks[j]);
        std::vector<uint8_t> w(P_, 0);
        for (int j = 0; j < P_; ++j)
            if (!folded[j] && c[j] >= L && pot > 0 && ranks[j] == best) w[j] = 1, ++nw;
        if (nw == 0 && pot > 0) {
            int bl = big;
            for (int j = 0; j < P_; ++j)
                if (!folded[j]) bl = std::min(bl, ranks[j]);
            for (int j = 0; j < P_; ++j)
                if (!folded[j] && ranks[j] == bl) w[j] = 1, ++nw;
        }
        for (int j = 0; j < P_; ++j)
            if (w[j]) win[j] += pot / std::max(nw, 1);
        prev = L;
    }
    if (dead > 0) {
        int bl = big, n = 0;
        for (int j = 0; j < P_; ++j)
            if (!folded[j]) bl = std::min(bl, ranks[j]);
        for (int j = 0; j < P_; ++j) n += !folded[j] && ranks[j] == bl;
        for (int j = 0; j < P_; ++j)
            if (!folded[j] && ranks[j] == bl) win[j] += double(dead) / std::max(n, 1);
        c[1] += double(dead);
    }
    for (int j = 0; j < P_; ++j) win[j] -= c[j];
    return win;
}

}
