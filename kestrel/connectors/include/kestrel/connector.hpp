// Connectors: a platform described in a file (YAML or JSON), loaded at run time, no rebuild. See docs/design.md.
// A connector lists the platform's API routes as the user writes them (method, path, query, headers, body), says
// which route plays each step of a hand (flow), where the answers are in each response (fields, JSON pointers), and
// names the codec that reads the platform's notation. Any file a user writes is accepted if it is complete.
#pragma once
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kestrel {

using json = nlohmann::json;

struct ConnectorSpec {
    std::string name, codec, path;
    json raw;  // the whole file, as JSON
    // game
    std::string variant = "nlhe";
    int players = 2;
    int64_t big = 100, stack = 20000;
    // what the file reaches, for the user to see before running it
    std::vector<std::string> env;    // environment variables it reads (KESTREL_* only)
    std::vector<std::string> hosts;  // base URLs it talks to
};

// Reads a .yaml/.yml/.json connector file; throws with a reason when it is malformed or not allowed.
ConnectorSpec load_connector(const std::string& path);
// every connector file (.yaml, .yml, .json) in a folder; files that do not load are reported in errors
std::vector<ConnectorSpec> load_connectors(const std::string& dir, std::vector<std::string>* errors = nullptr);

// Replaces ${name} (from vars) and ${env:NAME} in every string of a request template. A string that is exactly one
// placeholder whose value is empty is dropped from its object (e.g. no token on the first request).
json render(const json& tmpl, const std::map<std::string, std::string>& vars);

// One request-response transport (HTTP/HTTPS; WebSocket, TCP and MCP to follow behind the same interface).
class Transport {
  public:
    virtual ~Transport() = default;
    // one route as the connector lists it: {method, path, query, headers, body, content_type}
    virtual json call(const json& route, const std::map<std::string, std::string>& vars) = 0;
};

std::unique_ptr<Transport> make_transport(const ConnectorSpec& spec);

}  // namespace kestrel
