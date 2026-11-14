#include "kestrel/session.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "kestrel/codec_acpc.hpp"

namespace kestrel {

json SessionSummary::to_json() const {
    return {{"connector", connector}, {"hands", hands},      {"bb100", bb100},         {"ci95", {ci_lo, ci_hi}},
            {"errors", errors},       {"off_tree", off_tree}, {"fallbacks", fallbacks}, {"stopped", stopped}};
}

namespace {

json at(const json& r, const json& fields, const char* key) {
    if (!fields.contains(key)) return nullptr;
    const json& f = fields[key];
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
                  {"on_tree", ans.on_tree}, {"chose", vars["action"]}, {"big", spec.big}});
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
                     {"pos", at(r, fields, "my_seat")},
                     {"hole", at(r, fields, "hole")},
                     {"board", at(r, fields, "board")},
                     {"action", at(r, fields, "action")},
                     {"result_bb", won}};
        if (fields.contains("their_hole")) line["their_hole"] = at(r, fields, "their_hole");
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
    std::ofstream(plan.out_dir + "/summary.json") << sum.to_json().dump(1) << "\n";
    return sum;
}

}
