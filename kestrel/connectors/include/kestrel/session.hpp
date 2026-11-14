#pragma once
#include <atomic>
#include <functional>
#include <string>

#include "kestrel/connector.hpp"
#include "kestrel/headsup.hpp"

namespace kestrel {

struct SessionPlan {
    int hands = 100;
    std::string out_dir;
    int max_errors = 20;
    uint64_t seed = 1;
    std::string model = "hu-blueprint", solver = "none";
};

struct SessionSummary {
    std::string connector;
    int hands = 0, errors = 0, off_tree = 0, fallbacks = 0;
    double bb100 = 0, ci_lo = 0, ci_hi = 0;
    std::string stopped;
    json luck;  // the luck-adjusted estimate (kestrel/luck.hpp), when the platform shows the opponent's cards
    json to_json() const;
};

SessionSummary run_session(const ConnectorSpec& spec, const SessionPlan& plan, HUBlueprint& engine,
                           const std::function<void(const json&)>& on_hand = nullptr,
                           const std::atomic<bool>* stop = nullptr,
                           const std::function<void(const json&)>& on_event = nullptr);

}
