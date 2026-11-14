#pragma once
#include <string>
#include <vector>

namespace kestrel {

struct ServeOptions {
    int port = 8790;
    std::string pack;
    std::vector<std::string> connectors;
    std::string web, assets;
    std::string out = "sessions";
};

int serve(const ServeOptions& o);

}
