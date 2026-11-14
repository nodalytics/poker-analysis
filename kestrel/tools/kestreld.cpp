#include <atomic>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>

#include "kestrel/connector.hpp"
#include <httplib.h>
#include <thread>

#include "kestrel/gateway.hpp"
#include "kestrel/buckets.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/luck.hpp"
#include "kestrel/models.hpp"
#include "kestrel/server.hpp"
#include "kestrel/session.hpp"

using namespace kestrel;

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int usage() {
    std::fprintf(stderr,
                 "usage: kestreld list [DIR]\n"
                 "       kestreld luck DIR... [--big 100 --stack 20000]   luck-adjusted results of session logs\n"
                 "       kestreld models list|install PACKAGE.tar.gz|verify NAME [--dir DIR]\n"
                 "       kestreld mcp [--port 8790]        MCP over stdio for LLMs and agents\n"
                 "       kestreld check CONNECTOR\n"
                 "       kestreld play CONNECTOR --hands N --out DIR [--pack DIR] [--model M]\n"
                 "                     [--solver none|kestrel-river|texassolver] [--resolve-iters N]\n"
                 "                     [--texassolver BIN --texassolver-resources DIR] [--seed S] [--max-errors E]\n"
                 "                     table connectors (codec structured): --model <9-max model folder> [--tournaments N]\n"
                 "       kestreld                         (no arguments: start the app and open it, as a double-click does)\n"
                 "       kestreld serve [--port P] [--lan on|off] [--pack DIR] [--models DIR] [--connectors DIR]...\n"
                 "                      [--web DIR] [--assets DIR]\n"
                 "                      [--out DIR] [--resolve-iters N] [--texassolver BIN --texassolver-resources DIR]\n");
    return 2;
}
}

std::string config_dir() {
    const char* x = std::getenv("XDG_CONFIG_HOME");
    const char* h = std::getenv("HOME");
    if (!h) h = std::getenv("USERPROFILE");
    return x ? std::string(x) + "/kestrel" : std::string(h ? h : ".") + "/.config/kestrel";
}

std::string default_dir() {
    const char* x = std::getenv("XDG_CONFIG_HOME");
    const char* h = std::getenv("HOME");
    if (!h) h = std::getenv("USERPROFILE");
    return x ? std::string(x) + "/kestrel/connectors" : std::string(h ? h : ".") + "/.config/kestrel/connectors";
}

int luck_command(int argc, char** argv) {
    int64_t big = 100, stack = 20000;
    std::vector<std::string> dirs;
    for (int k = 2; k < argc; ++k) {
        std::string a = argv[k];
        if (a == "--big" && k + 1 < argc) big = std::stoll(argv[++k]);
        else if (a == "--stack" && k + 1 < argc) stack = std::stoll(argv[++k]);
        else dirs.push_back(a);
    }
    if (dirs.empty()) return usage();
    std::vector<LuckHand> all;
    for (auto& d : dirs) {
        std::vector<nlohmann::json> lines;
        for (auto& e : std::filesystem::directory_iterator(d)) {
            auto f = e.path().filename().string();
            if (f.rfind("hands", 0) != 0 || e.path().extension() != ".jsonl") continue;
            std::ifstream in(e.path());
            for (std::string l; std::getline(in, l);)
                if (!l.empty()) lines.push_back(nlohmann::json::parse(l));
        }
        std::vector<LuckHand> hs(lines.size());
        std::vector<char> ok(lines.size());
#pragma omp parallel for schedule(dynamic, 16)
        for (long k = 0; k < long(lines.size()); ++k) ok[k] = luck_hand(lines[k], big, stack, hs[k]);
        std::vector<LuckHand> got;
        for (size_t k = 0; k < hs.size(); ++k)
            if (ok[k]) got.push_back(hs[k]);
        auto e = luck_estimate(got);
        std::printf("%s: %s\n", d.c_str(), e.to_json().dump().c_str());
        all.insert(all.end(), got.begin(), got.end());
    }
    if (dirs.size() > 1) std::printf("pooled: %s\n", luck_estimate(all).to_json().dump().c_str());
    return 0;
}

int models_command(int argc, char** argv) {
    std::string sub = argc >= 3 ? argv[2] : "list", dir = default_models_dir(), arg;
    for (int k = 3; k < argc; ++k) {
        std::string a = argv[k];
        if (a == "--dir" && k + 1 < argc) dir = argv[++k];
        else arg = a;
    }
    try {
        if (sub == "list") {
            auto all = list_models(dir);
            std::printf("%zu models in %s\n", all.size(), dir.c_str());
            for (auto& m : all)
                std::printf("  %-12s %-6s %-10s %s%s\n", m.name.c_str(), m.version.c_str(), m.game.c_str(),
                            m.verified ? "verified" : "NOT VERIFIED: ", m.verified ? "" : m.problem.c_str());
            return 0;
        }
        if (sub == "install" && !arg.empty()) {
            auto m = install_model(arg, dir);
            std::printf("installed %s %s (%s) in %s; every file matches model.json\n", m.name.c_str(),
                        m.version.c_str(), m.game.c_str(), m.dir.c_str());
            return 0;
        }
        if (sub == "verify" && !arg.empty()) {
            auto m = verify_model(dir + "/" + arg);
            std::printf("%s: %s\n", arg.c_str(), m.verified ? "verified" : m.problem.c_str());
            return m.verified ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kestreld: %s\n", e.what());
        return 1;
    }
    return usage();
}

std::filesystem::path exe_dir(const char* argv0) {
    std::error_code ec;
#ifdef _WIN32
    wchar_t buf[4096];
    if (GetModuleFileNameW(nullptr, buf, 4096)) return std::filesystem::path(buf).parent_path();
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n) == 0) return std::filesystem::canonical(buf, ec).parent_path();
#else
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return p.parent_path();
#endif
    return std::filesystem::absolute(argv0, ec).parent_path();
}

// a double-click: serve with everything found beside the program (the release layout: bin/ and share/kestrel/) and
// in the user's folders, and open the app in the browser
ServeOptions defaults(const char* argv0) {
    namespace fs = std::filesystem;
    fs::path share = exe_dir(argv0).parent_path() / "share" / "kestrel";
    std::string home = std::getenv("HOME") ? std::getenv("HOME") : (std::getenv("USERPROFILE") ? std::getenv("USERPROFILE") : ".");
    fs::path data = std::getenv("XDG_DATA_HOME") ? fs::path(std::getenv("XDG_DATA_HOME")) / "kestrel"
                                                  : fs::path(home) / ".local" / "share" / "kestrel";
    ServeOptions o;
    o.web = (share / "web").string(), o.assets = (share / "assets").string();
    o.connectors = {(share / "connectors").string(), default_dir()};
    o.models = default_models_dir();
    o.extra_models = {(share / "models").string(), (share / "models" / "ninemax").string()};
    o.out = (data / "sessions").string();
    o.pack = std::getenv("KESTREL_PACK") ? std::getenv("KESTREL_PACK")
             : fs::exists(share / "pack" / "tree.tsv") ? (share / "pack").string()
                                                         : (data / "pack").string();
    if (const char* b = std::getenv("TEXASSOLVER_BIN")) o.ts_bin = b;
    if (const char* r = std::getenv("TEXASSOLVER_RESOURCES")) o.ts_resources = r;
    o.token_file = config_dir() + "/token";
    {  // settings.json: {"lan": true} to reach the app from other devices on the network
        std::ifstream sf(config_dir() + "/settings.json");
        auto st = nlohmann::json::parse(sf, nullptr, false);
        if (!st.is_discarded() && st.is_object()) o.lan = st.value("lan", false), o.port = st.value("port", 8790);
    }
    return o;
}

int launch(const char* argv0) {
    namespace fs = std::filesystem;
    ServeOptions o = defaults(argv0);
    o.open_browser = true;
    if (!fs::exists(fs::path(o.web) / "index.html")) {
        std::fprintf(stderr, "kestreld: the web app is not beside the program (%s)\n", o.web.c_str());
        return 1;
    }
    try {
        return serve(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kestreld: %s\n", e.what());
        return 1;
    }
}

// MCP over stdio for LLMs and agents: joins a running Kestrel, or starts one quietly in this process
int mcp_command(int argc, char** argv) {
    ServeOptions o = defaults(argv[0]);
    for (int k = 2; k + 1 < argc; k += 2)
        if (std::string(argv[k]) == "--port") o.port = std::stoi(argv[k + 1]);
    auto running = [&] {
        httplib::Client c("127.0.0.1", o.port);
        c.set_connection_timeout(1);
        auto r = c.Get("/v1/health");
        return r && r->status == 200;
    };
    if (!running()) {
        o.quiet = true, o.open_browser = false;
        std::thread([o] {
            try {
                serve(o);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "kestreld: %s\n", e.what());
            }
        }).detach();
        for (int k = 0; k < 100 && !running(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return mcp_stdio(o.port, "0.2.0");
}

int main(int argc, char** argv) {
    if (argc == 1) return launch(argv[0]);
    if (std::string(argv[1]) == "mcp") return mcp_command(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "luck") return luck_command(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "models") return models_command(argc, argv);
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
    if (argc >= 2 && std::string(argv[1]) == "serve") {
        ServeOptions o;
        o.pack = "blueprint/pack";
        for (int k = 2; k + 1 < argc; k += 2) {
            std::string a = argv[k], v = argv[k + 1];
            if (a == "--port") o.port = std::stoi(v);
            else if (a == "--pack") o.pack = v;
            else if (a == "--connectors") o.connectors.push_back(v);
            else if (a == "--web") o.web = v;
            else if (a == "--assets") o.assets = v;
            else if (a == "--out") o.out = v;
            else if (a == "--models") o.extra_models.push_back(v);
            else if (a == "--resolve-iters") o.resolve_iters = std::stoi(v);
            else if (a == "--texassolver") o.ts_bin = v;
            else if (a == "--texassolver-resources") o.ts_resources = v;
            else if (a == "--lan") o.lan = v == "on" || v == "1" || v == "true";
            else if (a == "--ws-port") o.ws_port = std::stoi(v);
            else if (a == "--tcp-port") o.tcp_port = std::stoi(v);
            else return usage();
        }
        o.connectors.push_back(default_dir());
        o.token_file = config_dir() + "/token";
        try {
            return serve(o);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "kestreld: %s\n", e.what());
            return 1;
        }
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
        int iters = 500;
        TexasSolverConfig ts;
        for (int k = 3; k + 1 < argc; k += 2) {
            std::string a = argv[k], v = argv[k + 1];
            if (a == "--hands") plan.hands = std::stoi(v);
            else if (a == "--out") plan.out_dir = v;
            else if (a == "--pack") pack = v;
            else if (a == "--seed") plan.seed = std::stoull(v);
            else if (a == "--max-errors") plan.max_errors = std::stoi(v);
            else if (a == "--model") plan.model = v;
            else if (a == "--tournaments") plan.tournaments = std::stoi(v);
            else if (a == "--solver") plan.solver = v;
            else if (a == "--resolve-iters") iters = std::stoi(v);
            else if (a == "--texassolver") ts.bin = v;
            else if (a == "--texassolver-resources") ts.resources = v;
            else return usage();
        }
        if (plan.out_dir.empty()) return usage();
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        int n = 0;
        double total = 0;
        auto progress = [&](const json& h) {
            total += h["result_bb"].get<double>();
            if (++n % 50 == 0) std::printf("%d hands, %+.1f bb/100 so far\n", n, total / n * 100), std::fflush(stdout);
        };
        if (spec.codec == "structured") {  // a 2-9 handed table: --model is a 9-max model folder (net.json, net.bin)
            if (plan.model == "hu-blueprint") throw std::runtime_error("this connector needs --model <9-max model folder>");
            buckets_init(pack);
            Net net(plan.model);
            auto s = run_ninemax_session(spec, plan, net, progress, &g_stop, [](const json& e) {
                if (e["type"] == "replay_error")
                    std::fprintf(stderr, "hand %s: could not replay: %s\n", e["hand"].dump().c_str(), e["error"].get<std::string>().c_str());
                if (e["type"] == "tournament")
                    std::printf("tournament %zu: place %s\n", e["tournaments"].get<size_t>(), e["place"].dump().c_str()),
                        std::fflush(stdout);
            });
            std::printf("%s\n", s.to_json().dump().c_str());
            return 0;
        }
        if (plan.model != "hu-blueprint") throw std::runtime_error("model " + plan.model + " cannot play this connector");
        std::string why;
        if (plan.solver == "texassolver" && !ts.usable(&why)) throw std::runtime_error(why);
        if (plan.solver != "none" && plan.solver != "kestrel-river" && plan.solver != "texassolver")
            throw std::runtime_error("solver " + plan.solver + " is not available");
        HUBlueprint engine(pack);
        if (plan.solver == "kestrel-river") engine.set_resolve(iters);
        if (plan.solver == "texassolver") engine.set_texassolver(ts);
        auto s = run_session(spec, plan, engine, progress, &g_stop);
        std::printf("%s\n", s.to_json().dump().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kestreld: %s\n", e.what());
        return 1;
    }
}
