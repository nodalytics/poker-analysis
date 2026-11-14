// A session: play hands on a platform through its connector until the plan is done, logging every hand.
#pragma once
#include <atomic>
#include <functional>
#include <string>

#include "kestrel/connector.hpp"
#include "kestrel/headsup.hpp"

namespace kestrel {

struct SessionPlan {
    int hands = 100;
    std::string out_dir;  // hands.jsonl, summary.json
    int max_errors = 20;  // platform errors before the session stops
    uint64_t seed = 1;
};

struct SessionSummary {
    std::string connector;
    int hands = 0, errors = 0, off_tree = 0, fallbacks = 0;
    double bb100 = 0, ci_lo = 0, ci_hi = 0;  // bb per 100 hands, bootstrap 95% interval
    std::string stopped;                      // why it ended early, if it did
    json to_json() const;
};

// Plays the plan. on_hand (optional) sees every finished hand's log line; stop (optional) ends it between hands.
SessionSummary run_session(const ConnectorSpec& spec, const SessionPlan& plan, HUBlueprint& engine,
                           const std::function<void(const json&)>& on_hand = nullptr,
                           const std::atomic<bool>* stop = nullptr);

}  // namespace kestrel
