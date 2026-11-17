#include "kestrel/opponents.hpp"

#include <algorithm>

#include "kestrel/profile.hpp"

#include "kestrel/store.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>

namespace kestrel {

using json = nlohmann::json;

namespace {

double pct(int64_t a, int64_t b) { return b ? std::round(1000.0 * double(a) / double(b)) / 10.0 : -1.0; }

OpponentStats from_json(const json& j) {
    OpponentStats s;
    auto g = [&](const char* k) { return j.value(k, int64_t(0)); };
    s.hands = g("hands"), s.vpip = g("vpip_n"), s.pfr = g("pfr_n"), s.threebet = g("threebet_n");
    s.threebet_chance = g("threebet_chances"), s.cbet_folds = g("cbet_folds"), s.cbet_faced = g("cbet_faced");
    s.aggressive = g("aggressive"), s.passive = g("passive"), s.saw_flop = g("saw_flop"), s.showdown = g("showdown");
    if (j.contains("shown") && j["shown"].is_array())
        for (auto& x : j["shown"]) s.shown.push_back(x);
    for (int d = 0; d < 6; ++d) {  // the forgetting counts (books written before 2026-11-17 have none: start empty)
        if (j.contains("recent_x") && j["recent_x"].size() == 6) s.recent_x[d] = j["recent_x"][d].get<double>();
        if (j.contains("recent_n") && j["recent_n"].size() == 6) s.recent_n[d] = j["recent_n"][d].get<double>();
    }
    return s;
}

}  // namespace

json OpponentStats::to_json() const {
    return {{"hands", hands},
            {"vpip", pct(vpip, hands)},
            {"pfr", pct(pfr, hands)},
            {"threebet", pct(threebet, threebet_chance)},
            {"fold_to_cbet", pct(cbet_folds, cbet_faced)},
            {"aggression", passive ? std::round(10.0 * double(aggressive) / double(passive)) / 10.0 : -1.0},
            {"went_to_showdown", pct(showdown, saw_flop)},
            // the counts, so the book can be read back and the rates recomputed
            {"vpip_n", vpip}, {"pfr_n", pfr}, {"threebet_n", threebet}, {"threebet_chances", threebet_chance},
            {"cbet_folds", cbet_folds}, {"cbet_faced", cbet_faced}, {"aggressive", aggressive}, {"passive", passive},
            {"saw_flop", saw_flop}, {"showdown", showdown}, {"shown", json(shown)},
            {"recent_x", recent_x}, {"recent_n", recent_n},
            {"profile", profile_json(*this)}};  // the closest player type (profile.hpp); not read back
}

OpponentBook::OpponentBook(std::string path) : path_(std::move(path)) {
    std::ifstream f(path_);
    json j = json::parse(f, nullptr, false);
    if (!j.is_object()) return;
    for (auto& [plat, players] : j.items())
        if (players.is_object())
            for (auto& [name, s] : players.items()) book_[plat][name] = from_json(s);
}

void OpponentBook::record(const std::string& platform, const std::vector<std::string>& players,
                          const std::vector<SeenAction>& acts, const std::map<std::string, std::vector<std::string>>& shown,
                          const std::vector<std::string>& board, const std::string& me) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& plat = book_[platform];
    std::set<std::string> in(players.begin(), players.end());
    in.erase(me);
    // before the flop: who put money in, who raised, who re-raised a single raise (and who had the chance)
    int raises = 0;
    std::string aggressor;
    std::set<std::string> vpip, pfr, three, three_chance, folded_pre;
    for (auto& a : acts) {
        if (a.street != 0) continue;
        if (raises == 1 && a.kind != "check") three_chance.insert(a.player);
        if (a.kind == "call" || a.kind == "bet" || a.kind == "raise" || a.kind == "allin") vpip.insert(a.player);
        if (a.kind == "raise" || a.kind == "bet" || a.kind == "allin") {
            if (raises == 1) three.insert(a.player);
            ++raises, pfr.insert(a.player), aggressor = a.player;
        }
        if (a.kind == "fold") folded_pre.insert(a.player);
    }
    // the flop: the pre-flop raiser's first bet (a continuation bet), and who folded to it
    bool cbet = false, flop_seen = false;
    std::set<std::string> faced, folded_cbet;
    for (auto& a : acts) {
        if (a.street < 1) continue;
        flop_seen = true;
        if (a.street > 1) break;
        if (!cbet && (a.kind == "bet" || a.kind == "raise" || a.kind == "allin")) {
            if (a.player != aggressor) break;  // someone else bet first: no continuation bet
            cbet = true;
            continue;
        }
        if (cbet && a.player != aggressor && !faced.count(a.player)) {
            faced.insert(a.player);
            if (a.kind == "fold") folded_cbet.insert(a.player);
        }
    }
    if (!board.empty() && board.size() >= 3) flop_seen = true;
    for (auto& name : in) {
        OpponentStats& s = plat[name];
        double x0[kProfileDim], n0[kProfileDim];
        profile_counts(s, false, x0, n0);  // before this hand, for the forgetting counts below
        ++s.hands;
        s.vpip += vpip.count(name), s.pfr += pfr.count(name);
        s.threebet += three.count(name), s.threebet_chance += three_chance.count(name);
        s.cbet_faced += faced.count(name), s.cbet_folds += folded_cbet.count(name);
        if (flop_seen && !folded_pre.count(name)) ++s.saw_flop;
        for (auto& a : acts)
            if (a.player == name && a.street >= 1) {
                if (a.kind == "bet" || a.kind == "raise" || a.kind == "allin") ++s.aggressive;
                else if (a.kind == "call") ++s.passive;
            }
        if (auto it = shown.find(name); it != shown.end() && !it->second.empty()) {
            ++s.showdown;
            s.shown.push_back({{"cards", it->second}, {"board", board}});
            while (s.shown.size() > 20) s.shown.pop_front();
        }
        double x1[kProfileDim], n1[kProfileDim];
        profile_counts(s, false, x1, n1);
        for (int d = 0; d < kProfileDim; ++d) {  // forget a little, then add this hand's events and chances
            s.recent_x[d] = kRecentDecay * s.recent_x[d] + (x1[d] - x0[d]);
            s.recent_n[d] = kRecentDecay * s.recent_n[d] + (n1[d] - n0[d]);
        }
    }
    if (++unsaved_ >= 10) save_locked();
}

void OpponentBook::save_locked() {
    unsaved_ = 0;
    json j = json::object();
    for (auto& [plat, players] : book_)
        for (auto& [name, s] : players) j[plat][name] = s.to_json();
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
    const std::string tmp = path_ + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return;
        f << j.dump(1);
    }
    std::filesystem::rename(tmp, path_, ec);  // whole or not at all
    if (store_) {
        try {
            store_->put_opponents(j);
        } catch (const std::exception&) {  // the database is a copy: the file above is saved either way
        }
    }
}

void OpponentBook::set_store(Store* s) {
    std::lock_guard<std::mutex> lk(mu_);
    store_ = s;
}

void OpponentBook::flush() {
    std::lock_guard<std::mutex> lk(mu_);
    save_locked();
}

RatePrior OpponentBook::prior_locked(const std::map<std::string, OpponentStats>& players) const {
    std::vector<const OpponentStats*> v;
    for (auto& [name, s] : players) v.push_back(&s);
    return population_prior(v);
}

json OpponentBook::platform(const std::string& platform) const {
    std::lock_guard<std::mutex> lk(mu_);
    json out = json::object();
    auto it = book_.find(platform);
    if (it == book_.end()) return out;
    const RatePrior prior = prior_locked(it->second);
    for (auto& [name, s] : it->second) {
        out[name] = s.to_json();
        out[name]["profile"] = profile_json(s, &prior);  // shrunk towards this platform's own players
    }
    return out;
}

json OpponentBook::player(const std::string& platform, const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = book_.find(platform);
    if (it == book_.end()) return nullptr;
    auto p = it->second.find(name);
    if (p == it->second.end()) return nullptr;
    json j = p->second.to_json();
    const RatePrior prior = prior_locked(it->second);
    j["profile"] = profile_json(p->second, &prior);
    return j;
}

json OpponentBook::platforms() const {
    std::lock_guard<std::mutex> lk(mu_);
    json out = json::array();
    for (auto& [plat, players] : book_) out.push_back({{"platform", plat}, {"players", players.size()}});
    return out;
}

std::string OpponentBook::hud(const std::string& platform, const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = book_.find(platform);
    if (it == book_.end()) return "";
    auto p = it->second.find(name);
    if (p == it->second.end() || p->second.hands == 0) return "";
    const OpponentStats& s = p->second;
    auto r = [](int64_t a, int64_t b) { return b ? std::to_string(int(std::lround(100.0 * double(a) / double(b)))) : std::string("-"); };
    std::string line = r(s.vpip, s.hands) + "/" + r(s.pfr, s.hands) + "/" + r(s.threebet, s.threebet_chance) + " · " +
                       std::to_string(s.hands) + "h";
    const RatePrior prior = prior_locked(it->second);
    const json pr = profile_json(s, &prior);
    if (pr["settled"].get<bool>()) line += " · " + pr["type"].get<std::string>();
    return line;
}

json OpponentBook::similar(const std::string& platform, const std::string& name, int k, bool all, int min_hands) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto pit = book_.find(platform);
    if (pit == book_.end()) return json::array();
    auto me = pit->second.find(name);
    if (me == pit->second.end()) return json::array();
    const auto v = profile_vector(profile_rates(me->second));
    std::vector<std::pair<double, json>> found;
    for (auto& [plat, players] : book_) {
        if (!all && plat != platform) continue;
        for (auto& [who, s] : players) {
            if ((plat == platform && who == name) || s.hands < min_hands) continue;
            const auto w = profile_vector(profile_rates(s));
            double ab = 0, aa = 0, bb = 0;
            for (int d = 0; d < kProfileDim; ++d) ab += v[d] * w[d], aa += v[d] * v[d], bb += w[d] * w[d];
            const double c = aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0;
            found.push_back({c, {{"platform", plat}, {"player", who}, {"hands", s.hands},
                                 {"similarity", std::round(c * 1000.0) / 1000.0}}});
        }
    }
    std::sort(found.begin(), found.end(), [](auto& a, auto& b) { return a.first > b.first; });
    json out = json::array();
    for (int i = 0; i < int(found.size()) && i < k; ++i) out.push_back(found[i].second);
    return out;
}

}  // namespace kestrel
