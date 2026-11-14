// kestreld: the Kestrel daemon. Today: play a session through a connector, or check a connector file.
//   kestreld list [DIR]                      every connector in a folder (default: ~/.config/kestrel/connectors)
//   kestreld check CONNECTOR                 load and validate a connector file (any file a user writes)
//   kestreld play CONNECTOR --hands N --out DIR [--pack DIR] [--seed S] [--max-errors E]
// The local API (REST, WebSocket, TCP, MCP) and the web page come next (docs/design.md).
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "kestrel/connector.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/session.hpp"

using namespace kestrel;

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int usage() {
    std::fprintf(stderr,
                 "usage: kestreld list [DIR]\n"
                 "       kestreld check CONNECTOR\n"
                 "       kestreld play CONNECTOR --hands N --out DIR [--pack DIR] [--seed S] [--max-errors E]\n");
    return 2;
}
}  // namespace

std::string default_dir() {
    const char* x = std::getenv("XDG_CONFIG_HOME");
    const char* h = std::getenv("HOME");
    if (!h) h = std::getenv("USERPROFILE");
    return x ? std::string(x) + "/kestrel/connectors" : std::string(h ? h : ".") + "/.config/kestrel/connectors";
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "list") {
        std::string dir = argc >= 3 ? argv[2] : default_dir();
        std::vector<std::string> errors;
        auto all = load_connectors(dir, &errors);
        std::printf("%zu connectors in %s\n", all.size(), dir.c_str());
        for (auto& c : all)
            std::printf("  %-20s codec %-8s %d players  %s\n", c.name.c_str(), c.codec.c_str(), c.players, c.path.c_str());
        for (auto& e : errors) std::printf("  not loaded: %s\n", e.c_str());
        return errors.empty() ? 0 : 1;
    }
    if (argc < 3) return usage();
    std::string cmd = argv[1], file = argv[2];
    try {
        ConnectorSpec spec = load_connector(file);
        if (cmd == "check") {
            std::printf("%s: ok (codec %s, %d players, blinds %lld, stack %lld)\n", spec.name.c_str(),
                        spec.codec.c_str(), spec.players, (long long)spec.big, (long long)spec.stack);
            for (auto& h : spec.hosts) std::printf("  talks to   %s\n", h.c_str());
            for (auto& e : spec.env) std::printf("  reads env  %s\n", e.c_str());
            return 0;
        }
        if (cmd != "play") return usage();
        SessionPlan plan;
        std::string pack = "blueprint/pack";
        for (int k = 3; k + 1 < argc; k += 2) {
            std::string a = argv[k], v = argv[k + 1];
            if (a == "--hands") plan.hands = std::stoi(v);
            else if (a == "--out") plan.out_dir = v;
            else if (a == "--pack") pack = v;
            else if (a == "--seed") plan.seed = std::stoull(v);
            else if (a == "--max-errors") plan.max_errors = std::stoi(v);
            else return usage();
        }
        if (plan.out_dir.empty()) return usage();
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        HUBlueprint engine(pack);
        int n = 0;
        double total = 0;
        auto s = run_session(
            spec, plan, engine,
            [&](const json& h) {
                total += h["result_bb"].get<double>();
                if (++n % 50 == 0)
                    std::printf("%d hands, %+.1f bb/100 so far\n", n, total / n * 100), std::fflush(stdout);
            },
            &g_stop);
        std::printf("%s\n", s.to_json().dump().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kestreld: %s\n", e.what());
        return 1;
    }
}
