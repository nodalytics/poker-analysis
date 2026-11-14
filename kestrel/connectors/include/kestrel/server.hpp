#pragma once
#include <string>
#include <vector>

namespace kestrel {

struct ServeOptions {
    int port = 8790;
    std::string pack;
    std::string models;  // models/ninemax: the 9-max networks, listed for the tables that can use them
    int resolve_iters = 500;
    std::string ts_bin, ts_resources;  // TexasSolver's console_solver and resources/, if installed
    int ts_threads = 8;
    std::vector<std::string> connectors;
    std::string web, assets;
    std::string out = "sessions";
    bool open_browser = false;  // open the web app once listening (a double-click start)
    bool lan = false;           // also listen on the network; other devices need the access token
    std::string token_file;     // where the access token is kept (created on first use, readable by the user only)
};

int serve(const ServeOptions& o);

}
