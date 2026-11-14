#include "kestrel/practice.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "kestrel/buckets.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/icm.hpp"

extern "C" int evaluate_7cards(int a, int b, int c, int d, int e, int f, int g);

namespace kestrel {

using json = nlohmann::json;

// position names for n players by seat (0 small blind, 1 big blind, the last occupied seat the button)
std::vector<std::string> position_names(int n) {
    static const char* nine[] = {"SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN"};
    std::vector<std::string> pos(9, "");
    if (n == 2) {
        pos[0] = "BTN/SB", pos[1] = "BB";
        return pos;
    }
    pos[0] = "SB", pos[1] = "BB";
    for (int k = 0; k < n - 2; ++k) pos[9 - (n - 2) + k] = nine[9 - (n - 2) + k];
    return pos;
}

std::string bb_str(double chips) {
    char b[32];
    double x = chips / double(BB);
    std::snprintf(b, sizeof b, std::fabs(x - std::round(x)) < 1e-9 ? "%.0f" : "%.1f", x);
    return b;
}

namespace {


}  // namespace

PracticeTournament::PracticeTournament(const Net& net, const PracticeConfig& c, std::string label)
    : net_(net), cfg_(c), label_(std::move(label)), rng_(c.seed) {
    int n = std::clamp(cfg_.players, 2, 9);
    std::uniform_real_distribution<double> u(-cfg_.spread, cfg_.spread);
    for (int k = 0; k < n; ++k) {
        double bb = std::max(4.0, cfg_.avg_bb * std::exp(u(rng_)));
        players_.push_back({label_ + " " + std::to_string(k + 1), int64_t(std::llround(bb * BB)), 0});
    }
    if (cfg_.human) players_[0].name = "You";
    button_ = int(rng_() % uint64_t(n));
}

bool PracticeTournament::over() const {
    int alive = 0;
    for (auto& p : players_) alive += p.place == 0;
    return alive < 2;
}

json PracticeTournament::standings() const {
    json a = json::array();
    for (auto& p : players_)
        a.push_back({{"name", p.name}, {"chips_bb", double(p.chips) / BB}, {"place", p.place},
                     {"prize", p.place ? cfg_.payouts[p.place - 1] : 0.0}});
    return a;
}

void PracticeTournament::play_hand(const std::function<void(const json&)>& emit,
                                   const std::function<int(const json&)>& ask) {
    if (over()) return;
    const int N = int(players_.size());
    // blinds rise: the chips shrink against a fixed big blind (ICM does not care about the scale)
    if (hands_ > 0 && hands_ % cfg_.hands_per_level == 0) {
        ++level_;
        for (auto& p : players_)
            if (p.place == 0) p.chips = std::max<int64_t>(1, std::llround(double(p.chips) / cfg_.level_factor));
    }
    // a player below two big blinds cannot post and play here: they finish (the network never trained on it)
    auto alive = [&] {
        std::vector<int> a;
        for (int k = 0; k < N; ++k)
            if (players_[k].place == 0) a.push_back(k);
        return a;
    };
    auto finish = [&](std::vector<int> out) {
        int left = int(alive().size());
        std::sort(out.begin(), out.end(), [&](int x, int y) { return players_[x].chips > players_[y].chips; });
        for (int k = int(out.size()) - 1; k >= 0; --k) players_[out[k]].place = left--;
    };
    {
        std::vector<int> short_;
        for (int k : alive())
            if (players_[k].chips < 2 * BB) short_.push_back(k);
        if (!short_.empty()) {
            finish(short_);
            for (int k : short_) players_[k].chips = 0;
        }
    }
    auto ring = alive();
    const int n = int(ring.size());
    if (n < 2) {
        if (n == 1) players_[ring[0]].place = 1;
        return;
    }
    while (players_[button_].place != 0) button_ = (button_ + 1) % N;
    int bpos = int(std::find(ring.begin(), ring.end(), button_) - ring.begin());
    // seats: 0 small blind, 1 big blind, then the others in order with the button last; early seats empty
    std::vector<int> who(9, -1);
    if (n == 2) {
        who[0] = ring[bpos], who[1] = ring[(bpos + 1) % n];
    } else {
        who[0] = ring[(bpos + 1) % n], who[1] = ring[(bpos + 2) % n];
        for (int k = 0; k < n - 2; ++k) who[9 - (n - 2) + k] = ring[(bpos + 3 + k) % n];
    }
    std::vector<int64_t> stacks(9, 0);
    for (int s = 0; s < 9; ++s)
        if (who[s] >= 0) stacks[s] = players_[who[s]].chips;
    Table T(stacks, Menu::named(net_.info().menu), net_.info().rules);
    TourneyCtx ctx;
    ctx.pay.assign(9, 0.0);
    for (int k = 0; k < n && k < int(cfg_.payouts.size()); ++k) ctx.pay[k] = cfg_.payouts[k];
    ctx.eq0 = icm(std::vector<double>(stacks.begin(), stacks.end()), ctx.pay);
    // the deal
    std::vector<int> deck(52);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng_);
    std::vector<std::array<int, 2>> hole(9);
    int d = 0;
    for (int s = 0; s < 9; ++s)
        if (who[s] >= 0) hole[s] = {deck[d], deck[d + 1]}, d += 2;
    std::vector<int> board(deck.begin() + d, deck.begin() + d + 5);
    std::vector<std::array<int, 4>> bk(9);
    std::vector<int> rank(9, 1 << 30);
    for (int s = 0; s < 9; ++s) {
        if (who[s] < 0) continue;
        int c[7] = {hole[s][0], hole[s][1], board[0], board[1], board[2], board[3], board[4]};
        bk[s] = {bucket_of(c, 2), bucket_of(c, 5), bucket_of(c, 6), bucket_of(c, 7)};
        rank[s] = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    }
    ++hands_;
    const auto pos = position_names(n);
    const int nb[4] = {0, 3, 4, 5};
    // what a seat may see: everything in a model-only table; with a human, only their own cards until the showdown
    auto visible = [&](int s, bool showdown) {
        return !cfg_.human || who[s] == 0 || (showdown && !T.folded[s]);
    };
    auto hole_json = [&](int s, bool showdown) {
        return visible(s, showdown) ? json{card_str(hole[s][0]), card_str(hole[s][1])} : json(nullptr);
    };
    auto seats_json = [&]() {
        json a = json::array();
        for (int s = 0; s < 9; ++s) {
            if (who[s] < 0) continue;
            a.push_back({{"seat", s}, {"pos", pos[s]}, {"name", players_[who[s]].name},
                         {"stack_bb", double(T.stack[s]) / BB}, {"in_bb", double(T.contrib[s]) / BB},
                         {"folded", bool(T.folded[s])}, {"allin", bool(T.allin[s])},
                         {"hole", hole_json(s, false)}, {"icm", ctx.eq0[s]}, {"you", cfg_.human && who[s] == 0}});
        }
        return a;
    };
    auto board_json = [&](int street) {
        json a = json::array();
        for (int k = 0; k < nb[std::min(street, 3)]; ++k) a.push_back(card_str(board[k]));
        return a;
    };
    json log = json::array();
    while (!T.done()) {
        const int i = T.actor(), st = T.street();
        std::vector<float> t;
        auto L = T.legal(&t);
        auto f = features(T, net_.info(), &ctx);
        std::vector<float> lg;
        float v;
        net_.forward(f, bk[i], st, strength(net_.info(), st, bk[i][st]), lg, v);
        auto p = net_.policy(lg, L);
        const int64_t to_call = T.bet - T.contrib[i];
        const int A = T.menu().A;
        auto label = [&](int a) {
            return a == 0       ? std::string("fold")
                   : a == 1     ? (to_call ? "call " + bb_str(double(std::min(to_call, T.avail[i] - T.contrib[i]))) : "check")
                   : a == A - 1 ? "all-in " + bb_str(double(T.avail[i]))
                                : "raise to " + bb_str(double(t[a - 2]));
        };
        const double pot = (double(std::accumulate(T.contrib.begin(), T.contrib.end(), int64_t(0))) + double(T.dead)) / BB;
        const bool human_turn = cfg_.human && who[i] == 0 && ask;
        int act;
        if (human_turn) {
            json choices = json::array();
            for (int a = 0; a < A; ++a)
                if (L[a]) choices.push_back({{"slot", a}, {"label", label(a)}, {"advice", p[a]}});
            act = ask({{"type", "practice_your_turn"}, {"hand", hands_}, {"level", level_}, {"street", st},
                       {"actor", i}, {"pos", pos[i]}, {"choices", choices}, {"seats", seats_json()},
                       {"board", board_json(st)}, {"pot_bb", pot}, {"log", log}, {"players_left", n}});
            if (act < 0) return;  // stopping
            if (act >= A || !L[act]) act = L[1] ? 1 : 0;
        } else {
            act = std::discrete_distribution<int>(p.begin(), p.end())(rng_);
        }
        json opts = json::array();
        for (int a = 0; a < A; ++a) {
            if (!L[a] || (p[a] < 0.005 && a != act)) continue;
            if (cfg_.human && a != act) continue;  // against a person, the model's probabilities stay hidden
            opts.push_back({{"label", label(a)}, {"prob", cfg_.human ? json(nullptr) : json(p[a])}, {"chosen", a == act}});
        }
        json ev = {{"type", "practice_decision"}, {"hand", hands_},  {"level", level_},       {"street", st},
                   {"actor", i},                  {"pos", pos[i]},   {"name", players_[who[i]].name},
                   {"options", opts},             {"seats", seats_json()}, {"board", board_json(st)},
                   {"pot_bb", pot},               {"players_left", n}, {"hidden", cfg_.human}};
        if (!cfg_.human) ev["value_bb"] = v;
        T.step(act);
        std::string said;
        for (auto& o : opts)
            if (o["chosen"]) said = o["label"];
        log.push_back({{"street", st}, {"pos", pos[i]}, {"name", players_[who[i]].name}, {"action", said}});
        ev["log"] = log;
        emit(ev);
    }
    auto u = T.payoffs(rank);
    // chips: payoffs can be fractional on split pots; round, and give the remainder to the largest winner
    std::vector<int64_t> after(9, 0);
    int64_t total_before = 0, total_after = 0;
    int best = -1;
    for (int s = 0; s < 9; ++s) {
        if (who[s] < 0) continue;
        total_before += stacks[s];
        after[s] = stacks[s] + std::llround(u[s]);
        total_after += after[s];
        if (best < 0 || u[s] > u[best]) best = s;
    }
    after[best] += total_before - total_after;
    std::vector<int> busted;
    json res = json::array();
    for (int s = 0; s < 9; ++s) {
        if (who[s] < 0) continue;
        res.push_back({{"seat", s}, {"pos", pos[s]}, {"name", players_[who[s]].name}, {"result_bb", u[s] / BB},
                       {"hole", hole_json(s, T.showdown())}, {"folded", bool(T.folded[s])},
                       {"you", cfg_.human && who[s] == 0}});
        players_[who[s]].chips = std::max<int64_t>(0, after[s]);
        if (players_[who[s]].chips == 0) busted.push_back(who[s]);
    }
    if (!busted.empty()) {
        // busted together: the bigger starting stack places higher
        std::vector<int64_t> start(N);
        for (int s = 0; s < 9; ++s)
            if (who[s] >= 0) start[who[s]] = stacks[s];
        int left = int(alive().size());
        std::sort(busted.begin(), busted.end(), [&](int x, int y) { return start[x] > start[y]; });
        for (int k = int(busted.size()) - 1; k >= 0; --k) players_[busted[k]].place = left--;
    }
    if (over())
        for (auto& p : players_)
            if (p.place == 0) p.place = 1;
    button_ = (button_ + 1) % N;
    // the board as far as it was dealt: everything at a showdown, else up to the street the hand ended on
    emit({{"type", "practice_hand"},  {"hand", hands_},  {"level", level_},
          {"board", board_json(T.showdown() ? 3 : T.street())},
          {"results", res},           {"showdown", T.showdown()},          {"log", log},
          {"standings", standings()}, {"over", over()}});
}

}  // namespace kestrel
