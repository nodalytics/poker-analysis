#include "kestrel/ninemax_spot.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "kestrel/buckets.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/icm.hpp"
#include "kestrel/practice.hpp"

namespace kestrel {

using json = nlohmann::json;

NMSpot NMSpot::from_json(const json& j) {
    NMSpot s;
    for (auto& p : j.at("players")) s.players.push_back({p.at("id").get<std::string>(), p.at("stack").get<double>()});
    s.button = j.at("button").get<std::string>();
    s.me = j.at("me").get<std::string>();
    s.sb = j.at("sb").get<double>(), s.bb = j.at("bb").get<double>(), s.ante = j.value("ante", 0.0);
    auto h = j.at("hole").get<std::vector<std::string>>();
    if (h.size() != 2) throw std::invalid_argument("two hole cards");
    s.hole = {h[0], h[1]};
    s.board = j.value("board", std::vector<std::string>{});
    if (j.contains("actions"))
        for (auto& a : j["actions"])
            s.actions.push_back({a.at("player").get<std::string>(), a.value("street", -1), a.at("kind").get<std::string>(),
                                 a.value("to", 0.0)});
    s.payouts = j.value("payouts", std::vector<double>{});
    return s;
}

json NMAdvice::to_json() const {
    json o = json::array();
    for (auto& x : options) o.push_back({{"action", x.action}, {"to", x.to}, {"add", x.add}, {"prob", x.prob}, {"slot", x.slot}});
    json j = {{"options", o}, {"value_bb", value_bb}, {"translated", translated}};
    if (!error.empty()) j["error"] = error;
    return j;
}

NMAdvice ninemax_decide(const Net& net, const NMSpot& sp) {
    NMAdvice adv;
    const int n = int(sp.players.size());
    if (n < 2 || n > 9) return adv.error = "2 to 9 players", adv;
    if (sp.bb <= 0) return adv.error = "no big blind", adv;
    int b = -1;
    for (int k = 0; k < n; ++k)
        if (sp.players[k].id == sp.button) b = k;
    if (b < 0) return adv.error = "the button is not a seated player", adv;
    // Kestrel's seats: 0 small blind, 1 big blind, then the others with the button last; early seats empty
    std::vector<int> who(9, -1);
    if (n == 2) {
        who[0] = b, who[1] = (b + 1) % n;
    } else {
        who[0] = (b + 1) % n, who[1] = (b + 2) % n;
        for (int k = 0; k < n - 2; ++k) who[9 - (n - 2) + k] = (b + 3 + k) % n;
    }
    auto seat_of = [&](const std::string& id) {
        for (int s = 0; s < 9; ++s)
            if (who[s] >= 0 && sp.players[who[s]].id == id) return s;
        return -1;
    };
    const double scale = double(BB) / sp.bb;
    std::vector<int64_t> stacks(9, 0);
    for (int s = 0; s < 9; ++s)
        if (who[s] >= 0) stacks[s] = std::max<int64_t>(1, std::llround(sp.players[who[s]].stack * scale));
    Rules replay = net.info().rules;
    replay.ante_bb = sp.ante / sp.bb;
    replay.open_limp = true;  // opponents may limp on a platform; the model's own rule is restored for its move
    replay.short_stacks = true;
    Table T(stacks, Menu::named(net.info().menu), replay);
    const int A = T.menu().A;
    const auto pos = position_names(n);
    json log = json::array();
    auto said = [&](int s, const std::string& what) {
        log.push_back({{"street", T.street()}, {"pos", pos[s]}, {"name", sp.players[who[s]].id}, {"action", what}});
    };
    // the table counts chips over the whole hand; platforms say "to" per street
    std::vector<int64_t> base(9, 0);
    int cur = 0;
    auto sync = [&] {
        if (T.street() != cur) base = T.contrib, cur = T.street();
    };
    for (auto& a : sp.actions) {
        if (T.done()) return adv.error = "the hand is over before all actions were replayed", adv;
        sync();
        int s = seat_of(a.player);
        if (s != T.actor()) return adv.error = "action out of turn by " + a.player, adv;
        if (a.street >= 0 && a.street != T.street())
            return adv.error = a.player + "'s action is on street " + std::to_string(a.street) + ", the table is on " +
                               std::to_string(T.street()),
                   adv;
        std::vector<float> t;
        auto L = T.legal(&t);
        int slot = 1;
        if (a.kind == "fold") {
            slot = L[0] ? 0 : 1;
            said(s, L[0] ? "fold" : "check");
        } else if (a.kind == "check" || a.kind == "call") {
            slot = 1;
            said(s, T.bet > T.contrib[s] ? "call " + bb_str(double(std::min(T.bet, T.avail[s]) - base[s])) : "check");
        } else if (a.kind != "bet" && a.kind != "raise" && a.kind != "allin") {
            return adv.error = "unknown action " + a.kind, adv;
        } else {
            // the platform's own amount: the table state the network reads is then exactly the platform's
            double target = double(base[s]) + a.to * scale;
            // all-in when the platform says so, or within half a platform chip of it (rounding)
            int64_t total = a.kind == "allin" || target >= double(T.avail[s]) - 0.5 * scale - 0.5 ? T.avail[s] : std::llround(target);
            bool on_menu = total == T.avail[s];
            for (int k = 0; k < T.menu().S; ++k) on_menu = on_menu || (L[2 + k] && std::fabs(t[k] - double(total)) <= 0.5);
            adv.translated += !on_menu;
            said(s, total >= T.avail[s] ? "all-in " + bb_str(double(T.avail[s] - base[s]))
                                        : (T.bet > base[s] ? "raise to " : "bet ") + bb_str(double(total - base[s])));
            T.step_to(total);
            continue;
        }
        T.step(slot);
    }
    sync();
    const int me = seat_of(sp.me);
    if (T.done() || T.actor() != me) return adv.error = "it is not " + sp.me + "'s turn", adv;
    // the model's own rule for its move (e.g. no open-limping), so its legal moves are those it trained with
    Rules mine = net.info().rules;
    mine.ante_bb = replay.ante_bb, mine.short_stacks = true;
    T.set_rules(mine);
    TourneyCtx ctx;
    ctx.pay.assign(9, 0.0);
    if (sp.payouts.empty()) ctx.pay[0] = 1.0;
    for (size_t k = 0; k < sp.payouts.size() && k < 9; ++k) ctx.pay[k] = sp.payouts[k];
    ctx.eq0 = icm(std::vector<double>(stacks.begin(), stacks.end()), ctx.pay);
    std::vector<int> cards{card_from(sp.hole[0]), card_from(sp.hole[1])};
    for (auto& c : sp.board) cards.push_back(card_from(c));
    const int st = T.street(), nb[4] = {0, 3, 4, 5};
    if (int(sp.board.size()) < nb[st]) return adv.error = "the board is shorter than the street needs", adv;
    std::array<int, 4> bk{0, 0, 0, 0};
    for (int s = 0; s <= st; ++s) bk[s] = bucket_of(cards.data(), 2 + nb[s]);
    std::vector<float> t;
    auto L = T.legal(&t);
    auto f = features(T, net.info(), &ctx);
    std::vector<float> lg;
    float v;
    net.forward(f, bk, st, strength(net.info(), st, bk[st]), lg, v);
    auto p = net.policy(lg, L);
    adv.value_bb = v;
    json seats = json::array();
    for (int s = 0; s < 9; ++s) {
        if (who[s] < 0) continue;
        seats.push_back({{"seat", s}, {"pos", pos[s]}, {"name", sp.players[who[s]].id},
                         {"stack_bb", double(T.avail[s] - T.contrib[s]) / BB}, {"in_bb", double(T.contrib[s]) / BB},
                         {"folded", bool(T.folded[s])}, {"allin", bool(T.allin[s])}, {"icm", ctx.eq0[s]},
                         {"hole", s == me ? json{sp.hole[0], sp.hole[1]} : json(nullptr)}, {"you", s == me}});
    }
    const double pot = double(std::accumulate(T.contrib.begin(), T.contrib.end(), int64_t(0)) + T.dead) / BB;
    adv.view = {{"seats", seats}, {"actor", me}, {"pos", pos[me]}, {"name", sp.me}, {"street", st},
                {"pot_bb", pot}, {"board", sp.board}, {"log", log}, {"players_left", n}};
    const int64_t to_call = T.bet - T.contrib[me];
    for (int a = 0; a < A; ++a) {
        if (!L[a] || p[a] <= 0) continue;
        NMOption o;
        o.slot = a, o.prob = p[a];
        int64_t total = T.contrib[me];
        if (a == 0) o.action = "fold";
        else if (a == 1) o.action = to_call ? "call" : "check", total = std::min(T.bet, T.avail[me]);
        else if (a == A - 1) o.action = "allin", total = T.avail[me];
        else o.action = "raise", total = int64_t(t[a - 2]);
        o.label = a == 0 ? "fold" : a == 1 ? (to_call ? "call " + bb_str(double(total - base[me])) : "check")
                : a == A - 1 ? "all-in " + bb_str(double(total - base[me]))
                             : (T.bet > base[me] ? "raise to " : "bet ") + bb_str(double(total - base[me]));
        o.to = double(total - base[me]) / scale;
        o.add = double(total - T.contrib[me]) / scale;
        adv.options.push_back(o);
    }
    return adv;
}

}  // namespace kestrel
