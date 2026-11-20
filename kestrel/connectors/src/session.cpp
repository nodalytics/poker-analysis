#include "kestrel/session.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kestrel/codec_acpc.hpp"
#include "kestrel/luck.hpp"
#include "kestrel/ninemax_spot.hpp"
#include "kestrel/store.hpp"

namespace kestrel {

json SessionSummary::to_json() const {
    json j = {{"connector", connector}, {"hands", hands},      {"bb100", bb100},         {"ci95", {ci_lo, ci_hi}},
              {"errors", errors},       {"off_tree", off_tree}, {"fallbacks", fallbacks}, {"stopped", stopped}};
    if (!luck.is_null()) j["luck_adjusted"] = luck;
    if (!tournaments.is_null()) j["tournaments"] = tournaments;
    return j;
}

namespace {
// a model as the database names it: a folder's name (kestreld play takes a path, the server a name), so spots stored
// by either can be compared
std::string model_name(const std::string& m) {
    std::string s = m;
    while (s.size() > 1 && (s.back() == '/' || s.back() == '\\')) s.pop_back();
    const auto k = s.find_last_of("/\\");
    return k == std::string::npos ? s : s.substr(k + 1);
}

// the session's rows in the database, when there is one; a database error is reported once and never stops play
struct DbLog {
    Store* st = nullptr;
    int64_t id = 0;
    bool warned = false;
    DbLog(const SessionPlan& plan, const std::string& connector) : st(plan.store) {
        guard([&] { id = st->begin_session("session", connector, model_name(plan.model), plan.solver, plan.out_dir); });
    }
    template <class F> void guard(F f) {
        if (!st) return;
        try {
            f();
        } catch (const std::exception& e) {
            if (!warned) std::fprintf(stderr, "kestrel: database: %s (the session's files are still written)\n", e.what());
            warned = true;
        }
    }
    int64_t hand(const json& h) {  // the hand's row id (0: not stored)
        int64_t row = 0;
        guard([&] { if (id) row = st->add_hand(id, h); });
        return row;
    }
    void spots(int64_t hand, const std::string& model, const std::vector<Store::Spot>& sp) {
        if (hand && !sp.empty()) guard([&] { st->add_spots(hand, model, sp); });
    }
    void end(const json& summary) { guard([&] { if (id) st->end_session(id, summary); }); }
};


json at(const json& r, const json& fields, const char* key) {
    if (!fields.contains(key)) return nullptr;
    const json& f = fields[key];
    if (f.is_object() && f.contains("value")) return render(f["value"], {});
    if (f.is_object() && f.contains("entries")) {  // an object read as items: "$key", "$value", or pointers into the value
        json::json_pointer p(f.value("at", ""));
        if (!r.contains(p) || !r[p].is_object()) return nullptr;
        std::vector<std::string> keys;
        for (auto& [k, v] : r[p].items()) keys.push_back(k);
        auto numeric = [](const std::string& k) { return !k.empty() && k.find_first_not_of("0123456789") == std::string::npos; };
        std::sort(keys.begin(), keys.end(), [&](const std::string& a, const std::string& b) {
            const bool na = numeric(a), nb = numeric(b);
            if (na != nb) return na;  // numbers first, compared as numbers of any length
            if (na && a.size() != b.size()) return a.size() < b.size();
            return a < b;
        });
        json out = json::array();
        for (auto& k : keys) {
            const json& v = r[p][k];
            json o = json::object();
            for (auto& [name, how] : f["entries"].items()) {
                std::string h = how.get<std::string>();
                o[name] = h == "$key" ? json(k) : h == "$value" ? v : (v.is_object() && v.contains(json::json_pointer(h)) ? v[json::json_pointer(h)] : json());
            }
            out.push_back(o);
        }
        return out;
    }
    if (f.is_object() && f.contains("items")) {
        json::json_pointer p(f.value("at", ""));
        json out = json::array();
        if (!r.contains(p) || !r[p].is_array()) return nullptr;
        for (auto& item : r[p]) {
            json o = json::object();
            for (auto& [k, sub] : f["items"].items()) o[k] = at(item, f["items"], k.c_str());
            out.push_back(o);
        }
        return out;
    }
    if (!f.is_string() && !f.is_object())
        throw std::runtime_error(std::string("field ") + key + ": a JSON pointer or {at: ...}, not " + f.dump());
    std::string ptr = f.is_string() ? f.get<std::string>() : f.value("at", "");
    json::json_pointer p(ptr);
    if (!r.contains(p)) return nullptr;
    json v = r[p];
    if (f.is_object()) {
        if (f.contains("map")) {
            std::string k = v.is_string() ? v.get<std::string>() : v.dump();
            if (f["map"].contains(k)) v = f["map"][k];
        }
        if (f.contains("scale") && v.is_number()) v = v.get<double>() * f["scale"].get<double>();
        if (f.contains("equals")) v = v == f["equals"];
    }
    return v;
}

std::vector<std::string> cards(const json& j) {
    std::vector<std::string> v;
    if (j.is_array()) {
        for (auto& x : j) v.push_back(x.get<std::string>());
    } else if (j.is_string()) {
        std::string s;
        for (char c : j.get<std::string>())
            if (!std::isspace(static_cast<unsigned char>(c)) && c != ',') s += c;
        for (size_t k = 0; k + 1 < s.size(); k += 2) v.push_back(s.substr(k, 2));
    }
    return v;
}

void bootstrap(const std::vector<double>& x, double& mean, double& lo, double& hi) {
    mean = lo = hi = 0;
    if (x.empty()) return;
    for (double v : x) mean += v;
    mean /= double(x.size());
    std::mt19937_64 g(1);
    std::uniform_int_distribution<size_t> pick(0, x.size() - 1);
    std::vector<double> m(1000);
    for (auto& b : m) {
        double s = 0;
        for (size_t k = 0; k < x.size(); ++k) s += x[pick(g)];
        b = s / double(x.size());
    }
    std::sort(m.begin(), m.end());
    lo = m[25], hi = m[974];
}

}

// an ACPC action string (k c f b<total>, streets split by /) as actions by name: the small blind acts first before the
// flop, the big blind after it; pos 1: we are the small blind
std::vector<SeenAction> acpc_actions(const std::string& action, int pos, const std::string& me, const std::string& them) {
    std::vector<SeenAction> out;
    const std::string sb = pos == 1 ? me : them, bb = pos == 1 ? them : me;
    int street = 0;
    bool bet = true;  // before the flop the big blind is a bet
    std::string actor = sb;
    for (size_t k = 0; k < action.size(); ++k) {
        char c = action[k];
        if (c == '/') {
            ++street, bet = false, actor = bb;
            continue;
        }
        std::string kind;
        if (c == 'k') kind = "check";
        else if (c == 'c') kind = bet ? "call" : "check";
        else if (c == 'f') kind = "fold";
        else if (c == 'b') {
            kind = bet ? "raise" : "bet", bet = true;
            while (k + 1 < action.size() && std::isdigit(static_cast<unsigned char>(action[k + 1]))) ++k;
        } else {
            continue;
        }
        out.push_back({actor, street, kind});
        actor = actor == sb ? bb : sb;
    }
    return out;
}

std::string id_of(const json& v) { return v.is_string() ? v.get<std::string>() : v.is_null() ? "" : v.dump(); }

double num(const json& v, double dflt = 0) {
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) try {
            return std::stod(v.get<std::string>());
        } catch (...) {
        }
    return dflt;
}

bool truthy(const json& v) {
    return v.is_boolean() ? v.get<bool>() : v.is_number() ? v.get<double>() != 0 : v.is_string() ? !v.get<std::string>().empty() : false;
}

SessionSummary run_session(const ConnectorSpec& spec, const SessionPlan& plan, HUBlueprint& engine,
                           const std::function<void(const json&)>& on_hand, const std::atomic<bool>* stop,
                           const std::function<void(const json&)>& on_event) {
    auto emit = [&](json e) {
        if (on_event) on_event(std::move(e));
    };
    if (spec.codec != "acpc") throw std::runtime_error(spec.name + ": codec " + spec.codec + " is not built yet");
    if (spec.players != 2) throw std::runtime_error(spec.name + ": the acpc codec is heads-up");
    auto tr = make_transport(spec);
    const json& routes = spec.raw["routes"];
    const json& flow = spec.raw["flow"];
    const json& start = routes[flow["start"].get<std::string>()];
    const json& act = routes[flow["act"].get<std::string>()];
    auto fields_of = [&](const json& route) {
        json f = spec.raw["fields"];
        if (route.contains("response") && route["response"].contains("fields"))
            for (auto& [k, v] : route["response"]["fields"].items()) f[k] = v;
        return f;
    };
    const json start_fields = fields_of(start), act_fields = fields_of(act);
    json fields = start_fields;
    std::string token_ptr;
    if (spec.raw.contains("session") && spec.raw["session"].contains("token_field"))
        token_ptr = spec.raw["session"]["token_field"].get<std::string>();
    acpc::Game game{spec.big, spec.stack};

    std::filesystem::create_directories(plan.out_dir);
    std::ofstream log(plan.out_dir + "/hands.jsonl", std::ios::app);
    DbLog db(plan, spec.name);
    std::mt19937_64 rng(plan.seed);
    SessionSummary sum;
    sum.connector = spec.name;
    std::vector<double> results;
    std::vector<LuckHand> luck;
    std::map<std::string, std::string> vars;

    auto keep_token = [&](const json& r) {
        if (!token_ptr.empty()) {
            json::json_pointer p(token_ptr);
            if (r.contains(p) && r[p].is_string()) vars["token"] = r[p].get<std::string>();
        }
    };
    auto platform_error = [&](const json& r) {
        json e = at(r, fields, "error");
        return e.is_null() ? std::string() : e.is_string() ? e.get<std::string>() : e.dump();
    };

    if (flow.contains("login")) {
        json r = tr->call(routes[flow["login"].get<std::string>()], vars);
        if (auto e = platform_error(r); !e.empty()) throw std::runtime_error(spec.name + ": login refused: " + e);
        keep_token(r);
    }
    for (int h = 0; h < plan.hands; ++h) {
        if (stop && *stop) {
            sum.stopped = "stopped";
            break;
        }
        json r = tr->call(start, vars);
        fields = start_fields;
        keep_token(r);
        emit({{"type", "deal"}, {"hand", h + 1}, {"pos", at(r, fields, "my_seat")}, {"hole", at(r, fields, "hole")}});
        int guard = 0;
        bool failed = false;
        while (at(r, fields, "result").is_null()) {
            if (auto e = platform_error(r); !e.empty() || ++guard > 60) {
                ++sum.errors;
                failed = true;
                break;
            }
            HUSpot spot;
            std::string action = at(r, fields, "action").get<std::string>();
            int pos = at(r, fields, "my_seat").get<int>();
            if (!acpc::spot(action, pos, cards(at(r, fields, "hole")), cards(at(r, fields, "board")), game, spot)) {
                ++sum.errors;
                failed = true;
                break;
            }
            HUAnswer ans = engine.decide(spot);
            json opts = json::array();
            for (auto& o : ans.options)
                opts.push_back({{"kind", o.kind == 'f' ? "fold" : o.kind == 'k' ? (spot.facing ? "call" : "check") : "raise"},
                                {"to", o.to}, {"prob", o.prob}, {"label", o.label}});
            HUOption pick{'k', 0, 1.0, "fallback"};
            if (!ans.on_tree) {
                ++sum.off_tree;
            } else {
                std::vector<double> w;
                for (auto& o : ans.options) w.push_back(o.prob);
                pick = ans.options[std::discrete_distribution<size_t>(w.begin(), w.end())(rng)];
            }
            vars["action"] = acpc::encode(pick, spot);
            emit({{"type", "decision"}, {"hand", h + 1}, {"action", action}, {"pos", pos},
                  {"hole", at(r, fields, "hole")}, {"board", at(r, fields, "board")}, {"pot", spot.pot},
                  {"my_bet", spot.my_bet}, {"their_bet", spot.their_bet}, {"my_stack", spot.my_stack},
                  {"their_stack", spot.their_stack}, {"facing", spot.facing}, {"options", opts},
                  {"on_tree", ans.on_tree}, {"resolved", ans.resolved}, {"floored", ans.floored},
                  {"chose", vars["action"]}, {"big", spec.big}, {"model", plan.model}, {"solver", plan.solver}});
            r = tr->call(act, vars);
            fields = act_fields;
            keep_token(r);
            if (!platform_error(r).empty()) {
                ++sum.fallbacks;
                vars["action"] = spot.facing ? "c" : "k";
                r = tr->call(act, vars);
                fields = act_fields;
                keep_token(r);
            }
        }
        if (failed) {
            if (sum.errors >= plan.max_errors) {
                sum.stopped = "too many platform errors";
                break;
            }
            continue;
        }
        double won = at(r, fields, "result").get<double>() / double(spec.big);
        results.push_back(won);
        json line = {{"connector", spec.name},
                     {"model", plan.model},
                     {"solver", plan.solver},
                     {"pos", at(r, fields, "my_seat")},
                     {"hole", at(r, fields, "hole")},
                     {"board", at(r, fields, "board")},
                     {"action", at(r, fields, "action")},
                     {"result_bb", won}};
        if (fields.contains("their_hole")) line["their_hole"] = at(r, fields, "their_hole");
        if (plan.opponents) {  // a heads-up platform's opponent: the username its responses give, else the platform
            json who = at(r, fields, "opponent");
            const std::string them = who.is_null() ? spec.name : id_of(who), me = "kestrel";
            std::map<std::string, std::vector<std::string>> shown;
            if (line.contains("their_hole") && !line["their_hole"].is_null()) shown[them] = cards(line["their_hole"]);
            plan.opponents->record(spec.name, {me, them}, acpc_actions(id_of(line["action"]), line["pos"].is_number() ? line["pos"].get<int>() : 0, me, them),
                                   shown, cards(line["board"]), me);
        }
        if (spec.codec == "acpc") {
            try {
                LuckHand lh;
                if (luck_hand(line, spec.big, spec.stack, lh)) luck.push_back(lh);
            } catch (const std::exception&) {
            }
        }
        log << line.dump() << "\n" << std::flush;
        db.hand(line);
        if (on_hand) on_hand(line);
        json ev = line;
        ev["type"] = "hand";
        ev["hand"] = h + 1;
        emit(ev);
    }
    sum.hands = int(results.size());
    double m, lo, hi;
    bootstrap(results, m, lo, hi);
    sum.bb100 = m * 100, sum.ci_lo = lo * 100, sum.ci_hi = hi * 100;
    if (luck.size() >= 2) sum.luck = luck_estimate(luck).to_json();
    std::ofstream(plan.out_dir + "/summary.json") << sum.to_json().dump(1) << "\n";
    db.end(sum.to_json());
    return sum;
}


// ---- 2 to 9 handed tables (the structured codec): poll the table state, answer when it is our turn ----
SessionSummary run_ninemax_session(const ConnectorSpec& spec, const SessionPlan& plan, const Net& net,
                                   const std::function<void(const json&)>& on_hand, const std::atomic<bool>* stop,
                                   const std::function<void(const json&)>& on_event) {
    auto emit = [&](json e) {
        if (on_event) on_event(std::move(e));
    };
    if (spec.codec != "structured") throw std::runtime_error(spec.name + ": 9-max models play structured connectors");
    const json& routes = spec.raw["routes"];
    const json& flow = spec.raw["flow"];
    auto tr = make_transport(spec);
    const bool stream = tr->streaming();
    if (!stream && !flow.contains("state")) throw std::runtime_error(spec.name + ": flow.state (the table state route) is required");
    if (stream && !spec.raw.contains("events")) throw std::runtime_error(spec.name + ": a WebSocket connector needs events");
    const json& state_route = stream ? routes[flow["act"].get<std::string>()] : routes[flow["state"].get<std::string>()];
    const json& act_route = routes[flow["act"].get<std::string>()];
    const json fields = spec.raw["fields"];
    const json answer = spec.raw.value("answer", json::object());
    const json amap = answer.value("actions", json::object());
    const bool send_add = answer.value("amount", "to") == "add";
    const double ascale = answer.value("scale", 1.0);
    const double unit = std::pow(10.0, -answer.value("decimals", 0));  // whole chips unless the platform says otherwise
    const auto poll = std::chrono::milliseconds(spec.raw["transport"].value("poll_ms", 250));
    const bool ante_each = spec.raw.contains("game") && spec.raw["game"].value("ante", "big_blind") == "each";
    // values carried from every response into the next requests, e.g. {turn_id: /turn_id} for ${turn_id}
    const json carry = spec.raw.contains("session") ? spec.raw["session"].value("vars", json::object()) : json::object();
    const json* status_route = flow.contains("status") ? &routes[flow["status"].get<std::string>()] : nullptr;
    const json status_fields = status_route && status_route->contains("response") && (*status_route)["response"].contains("fields")
                                   ? (*status_route)["response"]["fields"]
                                   : fields;
    auto last_status = std::chrono::steady_clock::now();
    std::string token_ptr;
    if (spec.raw.contains("session") && spec.raw["session"].contains("token_field"))
        token_ptr = spec.raw["session"]["token_field"].get<std::string>();

    std::filesystem::create_directories(plan.out_dir);
    std::ofstream log(plan.out_dir + "/hands.jsonl", std::ios::app);
    DbLog db(plan, spec.name);
    std::mt19937_64 rng(plan.seed);
    SessionSummary sum;
    sum.connector = spec.name;
    sum.tournaments = json::array();
    std::vector<double> results;
    std::map<std::string, std::string> vars;
    auto keep_token = [&](const json& r) {
        if (!token_ptr.empty()) {
            json::json_pointer p(token_ptr);
            if (r.contains(p) && r[p].is_string()) vars["token"] = r[p].get<std::string>();
        }
        for (auto& [name, ptr] : carry.items()) {
            json::json_pointer p(ptr.get<std::string>());
            if (r.contains(p) && !r[p].is_null()) vars[name] = id_of(r[p]);
        }
    };
    auto platform_error = [&](const json& r) {
        json e = at(r, fields, "error");
        return e.is_null() || (e.is_string() && e.get<std::string>().empty()) ? std::string() : id_of(e);
    };
    auto call = [&](const std::string& step) {
        json r = tr->call(routes[flow[step].get<std::string>()], vars);
        keep_token(r);
        return r;
    };
    if (flow.contains("login"))
        if (auto e = platform_error(call("login")); !e.empty()) throw std::runtime_error(spec.name + ": login refused: " + e);
    if (auto e = platform_error(call("start")); !e.empty()) throw std::runtime_error(spec.name + ": could not join: " + e);

    std::string cur_hand, acted_at;
    double cur_start = 0, cur_bb = 0;
    json cur_line;
    std::vector<Store::Spot> cur_spots;  // the hand's decisions, with their embeddings
    auto my_start = [&](const NMSpot& sp) {
        for (auto& p : sp.players)
            if (p.id == sp.me) return p.stack;
        return 0.0;
    };
    std::vector<std::string> seen_names, seen_board;  // the current hand as last seen: players by username, actions
    std::vector<SeenAction> seen_acts;
    std::string seen_me;
    std::map<std::string, std::string> seen_name_of;  // player id -> username, for the hand being closed
    auto close_hand = [&](double now_chips, const json& r) {
        if (cur_hand.empty() || cur_bb <= 0) return;
        std::map<std::string, std::vector<std::string>> shown;
        if (plan.opponents && !seen_names.empty()) {
            // the finished hand's whole history, when the platform gives it (last_actions): what Kestrel saw stops at
            // its own last move, and folds after it, to a continuation bet say, would be missed
            json la = at(r, fields, "last_actions");
            if (la.is_array() && (!fields.contains("last_hand") || id_of(at(r, fields, "last_hand")) == cur_hand)) {
                auto user = [&](const std::string& id) { return seen_name_of.count(id) ? seen_name_of[id] : id; };
                seen_acts.clear();
                for (auto& a : la) {
                    const std::string kind = id_of(a["kind"]);
                    if (kind != "fold" && kind != "check" && kind != "call" && kind != "bet" && kind != "raise" && kind != "allin")
                        continue;
                    seen_acts.push_back({user(id_of(a["player"])), a.contains("street") && !a["street"].is_null() ? int(num(a["street"])) : 0, kind});
                }
                if (json ls = at(r, fields, "last_shown"); ls.is_array())
                    for (auto& x : ls) shown[user(id_of(x["id"]))] = cards(x["cards"]);
                if (json lb = at(r, fields, "last_board"); !lb.is_null()) seen_board = cards(lb);
            }
            plan.opponents->record(spec.name, seen_names, seen_acts, shown, seen_board, seen_me);
        }
        if (!seen_names.empty()) {  // the whole table, for analyses of the other players (tilt, collusion)
            json acts = json::array(), stacks = json::object();
            for (auto& a : seen_acts) acts.push_back({a.player, a.street, a.kind});
            const json start_stacks = cur_line.value("stacks_bb", json::object());  // named: items() of a temporary dangles
            for (auto& [id, st] : start_stacks.items())
                stacks[seen_name_of.count(id) ? seen_name_of[id] : id] = st;
            cur_line["table"] = {{"players", seen_names}, {"me", seen_me}, {"bb", cur_bb}, {"stacks_bb", stacks},
                                 {"actions", acts}, {"shown", shown}};
            cur_line.erase("stacks_bb");
        }
        seen_names.clear(), seen_acts.clear(), seen_board.clear();
        double won = (now_chips - cur_start) / cur_bb;
        results.push_back(won);
        cur_line["result_bb"] = won;
        if (json b = at(r, fields, "last_board"); !b.is_null()) cur_line["board"] = cards(b);
        log << cur_line.dump() << "\n" << std::flush;
        db.spots(db.hand(cur_line), model_name(plan.model), cur_spots);
        cur_spots.clear();
        if (on_hand) on_hand(cur_line);
        json ev = cur_line;
        ev["type"] = "hand";
        ev["table9"] = true;
        ev["hand"] = int(results.size());
        emit(ev);
    };
    int idle = 0, t_start = 0, translated = 0, decisions = 0;
    // streams: each event updates a table document that `fields` read, as they read a polled platform's reply
    json doc = json::object();
    const json events = spec.raw.value("events", json::object());
    const std::string type_ptr = events.value("type", "/type");
    int fallback_stage = 0;  // after a move: 1 when a rejection should get a check, 2 a fold
    uint64_t nonce = rng();
    auto path_of = [&](std::string path, const json& e) {  // "/stacks/{seat}": fields of the event fill the braces
        for (size_t a; (a = path.find('{')) != std::string::npos;) {
            size_t b = path.find('}', a);
            if (b == std::string::npos) break;
            std::string k = path.substr(a + 1, b - a - 1);
            std::string v = e.contains(k) ? id_of(e[k]) : "";
            if (v.empty() || v.size() > 32 ||
                v.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.") != std::string::npos)
                throw std::runtime_error("event field " + k + " is not a short identifier");
            path.replace(a, b - a + 1, v);
        }
        return json::json_pointer(path);
    };
    // the slot a path names in the table document, making objects (never arrays) along the way
    auto slot = [&](const json::json_pointer& p) -> json& {
        json* cur = &doc;
        std::string rest = p.to_string();
        for (size_t at_ = 0; at_ < rest.size();) {
            size_t next = rest.find('/', at_ + 1);
            std::string tok = rest.substr(at_ + 1, next == std::string::npos ? std::string::npos : next - at_ - 1);
            for (size_t k; (k = tok.find("~1")) != std::string::npos;) tok.replace(k, 2, "/");
            for (size_t k; (k = tok.find("~0")) != std::string::npos;) tok.replace(k, 2, "~");
            if (!cur->is_object()) *cur = json::object();
            cur = &(*cur)[tok];
            if (next == std::string::npos) break;
            at_ = next;
        }
        return *cur;
    };
    auto send_route = [&](const std::string& name) {
        vars["nonce"] = std::to_string(++nonce);
        tr->call(routes.at(name), vars);
    };
    // returns true when the event opens our turn
    auto apply_event = [&](const json& e) {
        json::json_pointer tp(type_ptr);
        if (!e.contains(tp)) return false;
        std::string type = id_of(e[tp]);
        if (!events.contains("apply") || !events["apply"].contains(type)) return false;
        const json& rule = events["apply"][type];
        const json ops_reset = rule.value("reset", json::array());
        for (auto& path : ops_reset) slot(path_of(path.get<std::string>(), e)) = nullptr;
        const json ops_unset = rule.value("unset", json::array());
        for (auto& path : ops_unset) {
            auto p = path_of(path.get<std::string>(), e);
            if (doc.contains(p) && doc[p.parent_pointer()].is_object()) doc[p.parent_pointer()].erase(p.back());
        }
        const json ops_copy = rule.value("copy", json::object());
        for (auto& [path, src] : ops_copy.items()) {
            json::json_pointer from(src.get<std::string>());
            json v = doc.contains(from) ? doc[from] : json();
            slot(path_of(path, e)) = std::move(v);
        }
        const json ops_set = rule.value("set", json::object());
        for (auto& [path, how] : ops_set.items()) {
            json one = {{"v", how}};
            slot(path_of(path, e)) = at(e, one, "v");
        }
        const json ops_append = rule.value("append", json::object());
        for (auto& [path, how] : ops_append.items()) {
            json one = {{"v", how}};
            json v = at(e, one, "v");
            json& dst = slot(path_of(path, e));
            if (!dst.is_array()) dst = json::array();
            if (v.is_array())
                for (auto& x : v) dst.push_back(x);
            else if (!v.is_null())
                dst.push_back(v);
        }
        const json ops_push = rule.value("push", json::object());
        for (auto& [path, spec_] : ops_push.items()) {
            json o = json::object();
            for (auto& [k, how] : spec_.items()) o[k] = at(e, spec_, k.c_str());
            json& dst = slot(path_of(path, e));
            if (!dst.is_array()) dst = json::array();
            dst.push_back(o);
        }
        const json ops_merge = rule.value("merge", json::object());
        for (auto& [path, how] : ops_merge.items()) {
            json one = {{"v", how}};
            json v = at(e, one, "v");
            json& dst = slot(path_of(path, e));
            if (!dst.is_object()) dst = json::object();
            if (v.is_object()) dst.update(v);
        }
        const json ops_index = rule.value("index", json::object());
        for (auto& [path, idx] : ops_index.items()) {  // a list of objects into a map
            json::json_pointer from(idx.at("at").get<std::string>());
            json& dst = slot(path_of(path, e));
            if (!dst.is_object()) dst = json::object();
            if (e.contains(from) && e[from].is_array())
                for (auto& x : e[from]) {
                    json::json_pointer kp(idx.at("key").get<std::string>()), vp(idx.at("value").get<std::string>());
                    if (x.contains(kp) && x.contains(vp)) dst[id_of(x[kp])] = x[vp];
                }
        }
        if (rule.contains("send")) send_route(rule["send"].get<std::string>());
        if (rule.value("reject", false) && fallback_stage > 0) {  // our move was refused: check, then fold
            ++sum.fallbacks;
            std::ofstream(plan.out_dir + "/refused.jsonl", std::ios::app) << json{{"event", e}, {"doc", doc}}.dump() << "\n";
            vars["action"] = amap.contains(fallback_stage == 1 ? "check" : "fold") ? id_of(amap[fallback_stage == 1 ? "check" : "fold"])
                                                                                     : (fallback_stage == 1 ? "check" : "fold");
            vars["amount"] = "";
            send_route(flow["act"].get<std::string>());
            fallback_stage = fallback_stage == 1 ? 2 : 0;
        }
        if (rule.value("accept", false)) fallback_stage = 0;
        return rule.value("turn", false);
    };
    while (true) {
        if (stop && *stop) {
            sum.stopped = "stopped";
            break;
        }
        json r;
        bool turn_event = true;
        if (stream) {
            json e = tr->next(500);
            if (e.is_null()) continue;
            if (e.contains("$transport")) {
                emit({{"type", "transport"}, {"state", e["$transport"]}, {"reason", e.value("reason", "")}});
                if (e["$transport"] == "open" && flow.contains("start")) send_route(flow["start"].get<std::string>());
                continue;
            }
            keep_token(e);
            json::json_pointer tp(type_ptr);
            const std::string etype = e.contains(tp) ? id_of(e[tp]) : "";
            const bool refusal = events.contains("apply") && events["apply"].contains(etype) &&
                                 events["apply"][etype].value("reject", false);  // counted as a fallback, not an error
            if (auto why = refusal ? std::string() : platform_error(e); !why.empty()) {
                emit({{"type", "platform_error"}, {"error", why}});
                if (++sum.errors >= plan.max_errors) {
                    sum.stopped = "too many platform errors: " + why;
                    break;
                }
            }
            try {
                turn_event = apply_event(e);
            } catch (const std::exception& x) {
                emit({{"type", "platform_error"}, {"error", std::string("event not applied: ") + x.what()}});
                if (++sum.errors >= plan.max_errors) {
                    sum.stopped = std::string("too many platform errors: ") + x.what();
                    break;
                }
                continue;
            }
            r = doc;
        } else {
            r = tr->call(state_route, vars);
            keep_token(r);
        }
        if (auto e = stream ? std::string() : platform_error(r); !e.empty()) {
            if (++sum.errors >= plan.max_errors) {
                sum.stopped = "too many platform errors: " + e;
                break;
            }
            std::this_thread::sleep_for(poll);
            continue;
        }
        NMSpot sp;
        sp.me = id_of(at(r, fields, "me"));
        sp.ante_each = ante_each;
        if (!stream && status_route && at(r, fields, "hand").is_null() &&
            std::chrono::steady_clock::now() - last_status > std::chrono::seconds(2)) {
            last_status = std::chrono::steady_clock::now();
            json st = tr->call(*status_route, vars);
            for (const char* k : {"finished", "place", "prize", "chips"})
                if (status_fields.contains(k)) r[k] = at(st, status_fields, k);
        }
        json fin = status_route ? r.value("finished", json()) : at(r, fields, "finished");
        if (truthy(fin)) {
            auto pick = [&](const char* k) { return status_route ? r.value(k, json()) : at(r, fields, k); };
            if (json c = pick("chips"); !c.is_null()) close_hand(num(c), r);
            json t = {{"place", int(num(pick("place")))}, {"prize", num(pick("prize"))},
                      {"hands", int(results.size()) - t_start}};
            t_start = int(results.size());
            cur_hand.clear(), acted_at.clear();  // a new tournament numbers its hands from 1 again
            if (stream) doc = json::object();
            sum.tournaments.push_back(t);
            emit({{"type", "tournament"}, {"table9", true}, {"place", t["place"]}, {"prize", t["prize"]},
                  {"hands", t["hands"]}, {"tournaments", sum.tournaments.size()}});
            if (int(sum.tournaments.size()) >= plan.tournaments || int(results.size()) >= plan.hands) break;
            if (auto e = platform_error(call("start")); !e.empty()) {
                sum.stopped = "could not join again: " + e;
                break;
            }
            continue;
        }
        json players = at(r, fields, "players");
        std::string hand = id_of(at(r, fields, "hand"));
        if (!players.is_array() || hand.empty()) {
            if (stream) continue;
            if (++idle > 2400) {
                sum.stopped = "no table state for too long";
                break;
            }
            std::this_thread::sleep_for(poll);
            continue;
        }
        idle = 0;
        for (auto& p : players)
            if (num(p["stack"]) > 0) sp.players.push_back({id_of(p["id"]), num(p["stack"])});  // busted seats sit out
        sp.button = id_of(at(r, fields, "button"));
        sp.sb = num(at(r, fields, "sb")), sp.bb = num(at(r, fields, "bb")), sp.ante = num(at(r, fields, "ante"));
        if (hand != cur_hand) {
            close_hand(my_start(sp), r);
            if (int(results.size()) >= plan.hands) break;
            cur_hand = hand, cur_start = my_start(sp), cur_bb = sp.bb;
            cur_line = {{"connector", spec.name}, {"model", model_name(plan.model)}, {"solver", plan.solver}, {"hand_id", hand}, {"players", sp.players.size()},
                        {"start_bb", sp.bb > 0 ? cur_start / sp.bb : 0.0}, {"decisions", json::array()}};
            for (auto& p : sp.players)  // every player's chips at the start, in big blinds (by id; named at the end)
                if (sp.bb > 0) cur_line["stacks_bb"][p.id] = p.stack / sp.bb;
        }
        if (stream ? !turn_event
                   : id_of(at(r, fields, "to_act")) != sp.me && !(at(r, fields, "to_act").is_boolean() && truthy(at(r, fields, "to_act")))) {
            if (stream) continue;
            std::this_thread::sleep_for(poll);
            continue;
        }
        auto hole = cards(at(r, fields, "hole"));
        if (hole.size() != 2) {
            std::this_thread::sleep_for(poll);
            continue;
        }
        sp.hole = {hole[0], hole[1]};
        sp.board = cards(at(r, fields, "board"));
        json acts = at(r, fields, "actions");
        if (acts.is_array())
            for (auto& a : acts) {
                std::string kind = id_of(a["kind"]);
                if (kind != "fold" && kind != "check" && kind != "call" && kind != "bet" && kind != "raise" && kind != "allin")
                    continue;  // blinds, antes, shows
                sp.actions.push_back({id_of(a["player"]), a.contains("street") && !a["street"].is_null() ? int(num(a["street"])) : -1,
                                      kind, num(a["to"]), a.contains("add") && !a["add"].is_null() ? num(a["add"]) : -1.0,
                                      a.contains("left") && !a["left"].is_null() ? num(a["left"]) : -1.0});
            }
        if (json pay = at(r, fields, "payouts"); pay.is_array())
            for (auto& x : pay) sp.payouts.push_back(num(x));
        // usernames: a connector's players may name a `name` field; else the player id is the name
        std::map<std::string, std::string> name_of;
        for (auto& p : players) name_of[id_of(p["id"])] = p.contains("name") && !p["name"].is_null() ? id_of(p["name"]) : id_of(p["id"]);
        auto username = [&](const std::string& id) { return name_of.count(id) ? name_of[id] : id; };
        seen_names.clear();
        for (auto& p : sp.players) seen_names.push_back(username(p.id));
        seen_acts.clear();
        for (auto& a : sp.actions) seen_acts.push_back({username(a.player), std::max(a.street, 0), a.kind});
        seen_board = sp.board, seen_me = username(sp.me), seen_name_of = name_of;
        // a state the platform has not yet updated after our move: wait, never answer twice
        std::string key = hand + "/" + std::to_string(sp.actions.size()) + (stream ? "/" + std::to_string(nonce) : "");
        if (key == acted_at) {
            std::this_thread::sleep_for(poll);
            continue;
        }
        SearchConfig sc;
        sc.samples = plan.search_samples, sc.robust = plan.search_robust, sc.seed = plan.seed * 1000003 + rng();
        if (plan.search_exploit && plan.opponents) {  // each opponent as its profile says, weighted by the evidence
            sc.exploit_weight = 0.5;
            for (auto& p : sp.players) {
                const std::string u = name_of.count(p.id) ? name_of[p.id] : p.id;
                if (u == seen_me) continue;
                json stats = plan.opponents->player(spec.name, u);
                if (!stats.is_null()) sp.tilts[p.id] = opponent_tilt(stats);
            }
        }
        NMAdvice adv = ninemax_decide(net, sp, plan.search_samples > 0 ? &sc : nullptr);
        NMOption pick{"check", 0, 0, 1.0, 1, "check"};
        if (!adv.error.empty() || adv.options.empty()) {
            ++sum.off_tree;
            emit({{"type", "replay_error"}, {"hand", hand}, {"error", adv.error}});
            std::ofstream(plan.out_dir + "/replay_errors.jsonl", std::ios::app)
                << json{{"error", adv.error}, {"state", r}}.dump() << "\n";
        } else {
            std::vector<double> w;
            for (auto& o : adv.options) w.push_back(o.prob);
            pick = adv.options[std::discrete_distribution<size_t>(w.begin(), w.end())(rng)];
        }
        auto send = [&](const NMOption& o) {
            vars["action"] = amap.contains(o.action) ? id_of(amap[o.action]) : o.action;
            double amt = (send_add ? o.add : o.to) * ascale / unit;
            amt = (o.action == "allin" ? std::floor(amt + 1e-6) : std::round(amt)) * unit;  // never past the stack
            vars["amount"] = o.action == "raise" || o.action == "allin" || answer.value("amount_always", false)
                                 ? (unit >= 1 ? json(int64_t(amt)) : json(amt)).dump()
                                 : "";
            vars["nonce"] = std::to_string(++nonce);
            return tr->call(act_route, vars);
        };
        json ar = send(pick);
        fallback_stage = stream ? 1 : 0;
        keep_token(ar);
        if (auto why = platform_error(ar); !why.empty()) {
            ++sum.fallbacks;
            std::ofstream(plan.out_dir + "/refused.jsonl", std::ios::app)
                << json{{"sent", {{"action", vars["action"]}, {"amount", vars["amount"]}}}, {"why", why},
                        {"options", adv.to_json()["options"]}, {"state", r}}.dump() << "\n";
            ar = send({"check", 0, 0, 1.0, 1, "check"});
            keep_token(ar);
            if (!platform_error(ar).empty()) ar = send({"fold", 0, 0, 1.0, 0, "fold"});
        }
        acted_at = key;
        ++decisions, translated += adv.translated;
        json opts = json::array();
        for (auto& o : adv.options)
            opts.push_back({{"label", o.label}, {"action", o.action}, {"to", o.to}, {"prob", o.prob},
                            {"chosen", o.slot == pick.slot && !adv.options.empty()}});
        const int street = sp.board.size() >= 5 ? 3 : sp.board.size() >= 4 ? 2 : sp.board.size() >= 3 ? 1 : 0;
        std::string chose = pick.label.empty() ? pick.action : pick.label;
        cur_line["hole"] = hole;
        cur_line["decisions"].push_back({{"street", street}, {"chose", chose}});
        if (plan.store && !adv.embedding.empty())  // the spot as the network saw it, for finding similar spots later
            cur_spots.push_back({street, adv.view.value("pos", ""), chose, adv.embedding});
        std::string summary;
        for (auto& d : cur_line["decisions"]) summary += (summary.empty() ? "" : ", ") + d["chose"].get<std::string>();
        cur_line["action"] = summary;
        json ev = adv.view.is_object() ? adv.view : json::object();
        if (plan.opponents && ev.contains("seats"))  // each seat's statistics so far, and its username
            for (auto& seat : ev["seats"]) {
                const std::string u = name_of.count(id_of(seat["name"])) ? name_of[id_of(seat["name"])] : id_of(seat["name"]);
                seat["name"] = u;
                if (u != seen_me) seat["hud"] = plan.opponents->hud(spec.name, u);
            }
        ev.update({{"type", "decision"}, {"table9", true}, {"hand", hand}, {"hole", hole}, {"options", opts},
                   {"chose", chose}, {"to", pick.to}, {"translated", adv.translated}, {"value_bb", adv.value_bb},
                   {"model", plan.model}, {"error", adv.error}, {"street", street}, {"board", sp.board},
                   {"tournament", int(sum.tournaments.size()) + 1}});
        ev["places"] = json::array();
        for (auto& x : sum.tournaments) ev["places"].push_back(x["place"]);
        emit(ev);
    }
    sum.hands = int(results.size());
    double m, lo, hi;
    bootstrap(results, m, lo, hi);
    sum.bb100 = m * 100, sum.ci_lo = lo * 100, sum.ci_hi = hi * 100;
    json j = sum.to_json();
    j["decisions"] = decisions, j["translated_sizes"] = translated;
    std::ofstream(plan.out_dir + "/summary.json") << j.dump(1) << "\n";
    db.end(j);
    return sum;
}

}
