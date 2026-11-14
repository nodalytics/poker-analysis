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
    virtual json call(const json& route, const std::map<std::string, std::string>& vars) = 0;
};

std::unique_ptr<Transport> make_transport(const ConnectorSpec& spec);

}
