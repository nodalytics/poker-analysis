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

namespace kestrel {

json SessionSummary::to_json() const {
    json j = {{"connector", connector}, {"hands", hands},      {"bb100", bb100},         {"ci95", {ci_lo, ci_hi}},
              {"errors", errors},       {"off_tree", off_tree}, {"fallbacks", fallbacks}, {"stopped", stopped}};
    if (!luck.is_null()) j["luck_adjusted"] = luck;
    if (!tournaments.is_null()) j["tournaments"] = tournaments;
    return j;
}

namespace {

json at(const json& r, const json& fields, const char* key) {
    if (!fields.contains(key)) return nullptr;
    const json& f = fields[key];
    if (f.is_object() && f.contains("value")) return render(f["value"], {});
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
        if (spec.codec == "acpc") {
            try {
                LuckHand lh;
                if (luck_hand(line, spec.big, spec.stack, lh)) luck.push_back(lh);
            } catch (const std::exception&) {
            }
        }
        log << line.dump() << "\n" << std::flush;
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
    if (!flow.contains("state")) throw std::runtime_error(spec.name + ": flow.state (the table state route) is required");
    auto tr = make_transport(spec);
    const json& state_route = routes[flow["state"].get<std::string>()];
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
    auto my_start = [&](const NMSpot& sp) {
        for (auto& p : sp.players)
            if (p.id == sp.me) return p.stack;
        return 0.0;
    };
    auto close_hand = [&](double now_chips, const json& r) {
        if (cur_hand.empty() || cur_bb <= 0) return;
        double won = (now_chips - cur_start) / cur_bb;
        results.push_back(won);
        cur_line["result_bb"] = won;
        if (json b = at(r, fields, "last_board"); !b.is_null()) cur_line["board"] = cards(b);
        log << cur_line.dump() << "\n" << std::flush;
        if (on_hand) on_hand(cur_line);
        json ev = cur_line;
        ev["type"] = "hand";
        ev["table9"] = true;
        ev["hand"] = int(results.size());
        emit(ev);
    };
    int idle = 0, t_start = 0, translated = 0, decisions = 0;
    while (true) {
        if (stop && *stop) {
            sum.stopped = "stopped";
            break;
        }
        json r = tr->call(state_route, vars);
        keep_token(r);
        if (auto e = platform_error(r); !e.empty()) {
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
        if (status_route && at(r, fields, "hand").is_null() &&
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
            if (++idle > 2400) {
                sum.stopped = "no table state for too long";
                break;
            }
            std::this_thread::sleep_for(poll);
            continue;
        }
        idle = 0;
        for (auto& p : players) sp.players.push_back({id_of(p["id"]), num(p["stack"])});
        sp.button = id_of(at(r, fields, "button"));
        sp.sb = num(at(r, fields, "sb")), sp.bb = num(at(r, fields, "bb")), sp.ante = num(at(r, fields, "ante"));
        if (hand != cur_hand) {
            close_hand(my_start(sp), r);
            if (int(results.size()) >= plan.hands) break;
            cur_hand = hand, cur_start = my_start(sp), cur_bb = sp.bb;
            cur_line = {{"connector", spec.name}, {"model", plan.model}, {"hand_id", hand}, {"players", sp.players.size()},
                        {"start_bb", sp.bb > 0 ? cur_start / sp.bb : 0.0}, {"decisions", json::array()}};
        }
        if (id_of(at(r, fields, "to_act")) != sp.me && !(at(r, fields, "to_act").is_boolean() && truthy(at(r, fields, "to_act")))) {
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
                                      kind, num(a["to"])});
            }
        if (json pay = at(r, fields, "payouts"); pay.is_array())
            for (auto& x : pay) sp.payouts.push_back(num(x));
        // a state the platform has not yet updated after our move: wait, never answer twice
        std::string key = hand + "/" + std::to_string(sp.actions.size());
        if (key == acted_at) {
            std::this_thread::sleep_for(poll);
            continue;
        }
        NMAdvice adv = ninemax_decide(net, sp);
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
            return tr->call(act_route, vars);
        };
        json ar = send(pick);
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
        std::string summary;
        for (auto& d : cur_line["decisions"]) summary += (summary.empty() ? "" : ", ") + d["chose"].get<std::string>();
        cur_line["action"] = summary;
        json ev = adv.view.is_object() ? adv.view : json::object();
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
    return sum;
}

}
