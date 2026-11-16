#pragma once
#include <atomic>
#include <functional>
#include <string>

#include "kestrel/connector.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/net.hpp"
#include "kestrel/opponents.hpp"

namespace kestrel {

class Store;

struct SessionPlan {
    int hands = 100;
    std::string out_dir;
    int max_errors = 20;
    uint64_t seed = 1;
    std::string model = "hu-blueprint", solver = "none";
    int tournaments = 1;  // structured connectors: tournaments to play (the hands cap still applies)
    OpponentBook* opponents = nullptr;
    Store* store = nullptr;  // the database every hand is also written to (none: the files only)
    int search_samples = 0;  // structured connectors: search at decision time (0: the network alone)
    bool search_robust = false;  // where the opponents' statistics are kept (none: not tracked)
};

struct SessionSummary {
    std::string connector;
    int hands = 0, errors = 0, off_tree = 0, fallbacks = 0;
    double bb100 = 0, ci_lo = 0, ci_hi = 0;
    std::string stopped;
    json tournaments;  // structured connectors: each tournament's place and prize
    json luck;  // the luck-adjusted estimate (kestrel/luck.hpp), when the platform shows the opponent's cards
    json to_json() const;
};

SessionSummary run_session(const ConnectorSpec& spec, const SessionPlan& plan, HUBlueprint& engine,
                           const std::function<void(const json&)>& on_hand = nullptr,
                           const std::atomic<bool>* stop = nullptr,
                           const std::function<void(const json&)>& on_event = nullptr);

// a 2-9 handed table (codec structured) played by a 9-max tournament network
SessionSummary run_ninemax_session(const ConnectorSpec& spec, const SessionPlan& plan, const Net& net,
                                   const std::function<void(const json&)>& on_hand = nullptr,
                                   const std::atomic<bool>* stop = nullptr,
                                   const std::function<void(const json&)>& on_event = nullptr);

}
