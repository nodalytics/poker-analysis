#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kestrel {

using json = nlohmann::json;

struct ConnectorSpec {
    std::string name, codec, path;
    json raw;
    std::string variant = "nlhe";
    int players = 2;
    int64_t big = 100, stack = 20000;
    std::vector<std::string> env;
    std::vector<std::string> hosts;
};

ConnectorSpec load_connector(const std::string& path);
std::vector<ConnectorSpec> load_connectors(const std::string& dir, std::vector<std::string>* errors = nullptr);

json render(const json& tmpl, const std::map<std::string, std::string>& vars);

class Transport {
  public:
    virtual ~Transport() = default;
    // request and response (HTTP)
    virtual json call(const json& route, const std::map<std::string, std::string>& vars) = 0;
    // a stream of messages (WebSocket): the platform pushes, Kestrel sends a route's message template
    virtual bool streaming() const { return false; }
    virtual void send(const json&, const std::map<std::string, std::string>&) {}
    virtual json next(int /*timeout_ms*/) { return nullptr; }  // the next message, null if none came in time
};

std::unique_ptr<Transport> make_transport(const ConnectorSpec& spec);

// a request that still failed after the transport's own retries (about 2 minutes): the platform could not be reached,
// or kept answering 5xx. Anything else (a platform's refusal, bad JSON) is not one.
struct TransportError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// waits out an outage instead of ending the session: a request that throws TransportError is sent again after a
// pause that doubles from first_s to max_s, until it gets an answer, *stop is set, or the platform has been
// unreachable for limit_s in a row (then the TransportError is thrown). on_change hears {"state": "down"|"up",
// "reason", "down_s"}. A heads-up session lost one hand to a 2-minute network blip on the lab, and with it the
// whole session (2026-11-20); polls are safe to repeat, and a repeated move is refused as not our turn.
std::unique_ptr<Transport> patient(std::unique_ptr<Transport> inner, const std::atomic<bool>* stop, double limit_s,
                                   std::function<void(const json&)> on_change, double first_s = 15, double max_s = 300);

}
