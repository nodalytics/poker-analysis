#pragma once
#include <string>
#include <vector>

namespace kestrel {

struct ServeOptions {
    int port = 8790;
    std::string pack;
    std::string models;                     // installed model packages (kestreld models install)
    std::vector<std::string> extra_models;  // more folders of models, e.g. the ones shipped beside the program
    int resolve_iters = 500;
    int search_samples = 1024;  // 9-max search (solvers search, search-robust): deals of the unknown cards a decision
    std::string ts_bin, ts_resources;  // TexasSolver's console_solver and resources/, if installed
    int ts_threads = 8;
    std::vector<std::string> connectors;
    std::string web, assets;
    std::string out = "sessions";
    bool open_browser = false;  // open the web app once listening (a double-click start)
    bool lan = false;                      // also listen on the network (0.0.0.0); other devices need the access token. The app and a double-click turn it on
    int ws_port = 8791, tcp_port = 8792;  // JSON-RPC over WebSocket and TCP (127.0.0.1); 0 turns one off
    bool quiet = false;                    // print nothing on stdout (MCP uses it)
    std::string nine_max_default = "nm25";  // the 9-max model a table plays unless another is chosen
    std::string db;  // the database (kestrel.db): sessions, hands and opponents; empty: files only
    std::string token_file;     // where the access token is kept (created on first use, readable by the user only)
};

int serve(const ServeOptions& o);

}
