#pragma once
#include <string>
#include <vector>

namespace kestrel {

struct ServeOptions {
    int port = 8790;
    std::string pack;
    std::string models;  // models/ninemax: the 9-max networks, listed for the tables that can use them
    int resolve_iters = 500;
    std::vector<std::string> connectors;
    std::string web, assets;
    std::string out = "sessions";
};

int serve(const ServeOptions& o);

}
