#include "kestrel/server.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <memory>
#include <mutex>
#include <thread>

#include <httplib.h>
#ifndef _WIN32
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#endif

#include "kestrel/connector.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/luck.hpp"
#include "kestrel/models.hpp"
#include "kestrel/session.hpp"

namespace kestrel {

namespace {

constexpr const char* VERSION = "0.1.0";

// the models and solvers a table can play with; heads-up connectors take the heads-up blueprint
TexasSolverConfig ts_config(const ServeOptions& o) {
    TexasSolverConfig c;
    c.bin = o.ts_bin, c.resources = o.ts_resources, c.threads = o.ts_threads;
    return c;
}

json solvers_json(const ServeOptions& o) {
    std::string why;
    bool ts = ts_config(o).usable(&why);
    return json::array({{{"name", "none"}, {"about", "the model's own policy"}, {"available", true}},
                        {{"name", "kestrel-river"}, {"about", "safe river re-solve (CFR+ with the gadget), heads-up"},
                         {"available", true}},
                        {{"name", "texassolver"},
                         {"about", ts ? "TexasSolver on the turn and river from the blueprint's ranges, heads-up"
                                      : "TexasSolver: " + why},
                         {"available", ts}}});
}

struct Engines {
    std::mutex mu;
    std::map<std::string, std::shared_ptr<HUBlueprint>> by_key;
    std::shared_ptr<HUBlueprint> get(const ServeOptions& o, const std::string& model, const std::string& solver) {
        std::lock_guard<std::mutex> lk(mu);
        auto& e = by_key[model + "+" + solver];
        if (!e) {
            e = std::make_shared<HUBlueprint>(o.pack);
            if (solver == "kestrel-river") e->set_resolve(o.resolve_iters);
            if (solver == "texassolver") e->set_texassolver(ts_config(o));
        }
        return e;
    }
};

json models_json(const ServeOptions& o) {
    json a = json::array();
    bool pack = std::filesystem::exists(std::filesystem::path(o.pack) / "tree.tsv");
    a.push_back({{"name", "hu-blueprint"}, {"game", "heads-up"}, {"players", 2}, {"available", pack},
                 {"about", pack ? "heads-up blueprint, 200bb (the data pack)"
                                : "heads-up blueprint: the data pack is not installed (" + o.pack + ")"}});
    for (auto& m : list_models(o.models.empty() ? default_models_dir() : o.models))
        if (m.verified)
            a.push_back({{"name", m.name}, {"version", m.version}, {"game", m.game}, {"players", m.players},
                         {"available", false}, {"about", m.about + " (no connector for this game yet)"}});
    std::error_code ec;
    if (!o.models.empty())
        for (auto& d : std::filesystem::directory_iterator(o.models, ec))
            if (std::filesystem::exists(d.path() / "net.json"))
                a.push_back({{"name", d.path().filename().string()}, {"game", "9-max tournament"}, {"players", 9},
                             {"available", false}, {"about", "9-max tournament network; no 9-max connector yet"}});
    return a;
}

struct Table {
    int id = 0;
    ConnectorSpec spec;
    std::string model, solver;
    int hands = 0;
    std::atomic<bool> stop{false};
    std::thread th;
    std::string status = "starting";
    json state;
    std::deque<json> recent;
    int done = 0;
    double total_bb = 0;
    json summary;
    std::vector<LuckHand> luck;
};

struct Hub {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<long, json>> events;
    long seq = 0;
    std::map<int, std::shared_ptr<Table>> tables;
    int next_id = 1;
    std::atomic<bool> quit{false};

    void publish(json e) {
        {
            std::lock_guard<std::mutex> lk(mu);
            e["seq"] = ++seq;
            events.emplace_back(seq, std::move(e));
            if (events.size() > 2000) events.pop_front();
        }
        cv.notify_all();
    }
};

Hub* g_hub = nullptr;
httplib::Server* g_srv = nullptr;
void on_signal(int) {
    if (g_hub) g_hub->quit = true;
    if (g_srv) g_srv->stop();
}

std::string access_token(const std::string& file) {
    namespace fs = std::filesystem;
    std::ifstream in(file);
    std::string t;
    if (in >> t && t.size() >= 32) return t;
    std::random_device rd;
    const char* hex = "0123456789abcdef";
    t.clear();
    for (int k = 0; k < 40; ++k) t += hex[rd() & 15];
    fs::create_directories(fs::path(file).parent_path());
    std::ofstream(file) << t << "\n";
    std::error_code ec;
    fs::permissions(file, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    return t;
}

std::vector<std::string> lan_addresses() {
    std::vector<std::string> out;
#ifndef _WIN32
    ifaddrs* ifa = nullptr;
    if (getifaddrs(&ifa) == 0) {
        for (auto* i = ifa; i; i = i->ifa_next) {
            if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET) continue;
            char buf[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(i->ifa_addr)->sin_addr, buf, sizeof buf);
            std::string ip = buf;
            if (ip.rfind("127.", 0) != 0 && ip.rfind("169.254.", 0) != 0) out.push_back(ip);
        }
        freeifaddrs(ifa);
    }
#else
    char name[256];
    if (gethostname(name, sizeof name) == 0) {
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        if (getaddrinfo(name, nullptr, &hints, &res) == 0) {
            for (auto* r = res; r; r = r->ai_next) {
                char buf[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(r->ai_addr)->sin_addr, buf, sizeof buf);
                if (std::string(buf).rfind("127.", 0) != 0) out.push_back(buf);
            }
            freeaddrinfo(res);
        }
    }
#endif
    return out;
}

void open_url(const std::string& url) {
#ifdef _WIN32
    std::string cmd = "start \"\" \"" + url + "\"";
#elif defined(__APPLE__)
    std::string cmd = "open '" + url + "' >/dev/null 2>&1 &";
#else
    std::string cmd = "xdg-open '" + url + "' >/dev/null 2>&1 &";
#endif
    if (std::system(cmd.c_str()) != 0) std::printf("open %s in your browser\n", url.c_str());
}

json table_json(const Table& t) {
    double bb100 = t.done ? t.total_bb / t.done * 100 : 0;
    json j = {{"id", t.id},           {"connector", t.spec.name}, {"status", t.status}, {"model", t.model},
              {"solver", t.solver},
              {"hands_target", t.hands}, {"hands", t.done},       {"bb100", bb100},
              {"won_bb", t.total_bb}};
    if (!t.summary.is_null()) j["summary"] = t.summary;
    if (t.luck.size() >= 20) j["luck"] = luck_estimate(t.luck).to_json();
    return j;
}

std::vector<ConnectorSpec> all_connectors(const ServeOptions& o, std::vector<std::string>* errors) {
    std::vector<ConnectorSpec> out;
    for (auto& d : o.connectors)
        for (auto& c : load_connectors(d, errors))
            if (std::none_of(out.begin(), out.end(), [&](auto& x) { return x.name == c.name; })) out.push_back(c);
    return out;
}

void reply(httplib::Response& res, const json& j, int status = 200) {
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

}

int serve(const ServeOptions& o) {
    Hub hub;
    g_hub = &hub;
    Engines engines;
    std::string pack_problem;
    if (!std::filesystem::exists(std::filesystem::path(o.pack) / "tree.tsv"))
        pack_problem = "the heads-up data pack is not installed (looked in " + o.pack + ")";
    httplib::Server srv;
    g_srv = &srv;

    srv.Get("/v1/health", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(hub.mu);
        int running = 0;
        for (auto& [id, t] : hub.tables) running += t->status == "playing";
        reply(res, {{"ok", true}, {"version", VERSION}, {"tables", hub.tables.size()}, {"playing", running}});
    });

    srv.Get("/v1/connectors", [&](const httplib::Request&, httplib::Response& res) {
        std::vector<std::string> errors;
        json a = json::array();
        for (auto& c : all_connectors(o, &errors))
            a.push_back({{"name", c.name}, {"codec", c.codec}, {"players", c.players}, {"path", c.path},
                         {"hosts", c.hosts}, {"env", c.env}, {"big", c.big}, {"stack", c.stack}});
        reply(res, {{"connectors", a}, {"not_loaded", errors}});
    });

    srv.Get("/v1/engines", [&](const httplib::Request&, httplib::Response& res) {
        reply(res, {{"models", models_json(o)}, {"solvers", solvers_json(o)},
                    {"default", {{"model", "hu-blueprint"}, {"solver", "none"}}}});
    });

    srv.Post("/v1/sessions", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("connector")) return reply(res, {{"error", "need connector"}}, 400);
        std::string name = body["connector"];
        int hands = std::clamp(body.value("hands", 100), 1, 1000000);
        std::string model = body.value("model", "hu-blueprint"), solver = body.value("solver", "none");
        if (model != "hu-blueprint") return reply(res, {{"error", "model " + model + " cannot play this connector"}}, 400);
        bool known = false;
        for (auto& x : solvers_json(o))
            if (x["name"] == solver && x["available"] == true) known = true;
        if (!known) return reply(res, {{"error", "solver " + solver + " is not available"}}, 400);
        std::vector<std::string> errors;
        auto cs = all_connectors(o, &errors);
        auto it = std::find_if(cs.begin(), cs.end(), [&](auto& c) { return c.name == name; });
        if (it == cs.end()) return reply(res, {{"error", "no connector " + name}}, 404);
        auto t = std::make_shared<Table>();
        {
            std::lock_guard<std::mutex> lk(hub.mu);
            t->id = hub.next_id++;
            t->spec = *it;
            t->hands = hands;
            t->model = model, t->solver = solver;
            hub.tables[t->id] = t;
        }
        hub.publish({{"type", "table"}, {"table", t->id}, {"connector", name}, {"status", "starting"}, {"model", model},
                     {"solver", solver}});
        if (!pack_problem.empty()) return reply(res, {{"error", pack_problem}}, 400);
        std::shared_ptr<HUBlueprint> engine;
        try {
            engine = engines.get(o, model, solver);
        } catch (const std::exception& e) {
            return reply(res, {{"error", e.what()}}, 400);
        }
        t->th = std::thread([&hub, engine, &o, t] {
            SessionPlan plan;
            plan.model = t->model, plan.solver = t->solver;
            plan.hands = t->hands;
            plan.out_dir = o.out + "/" + std::to_string(t->id) + "-" + t->spec.name;
            plan.seed = uint64_t(t->id) * 7919;
            auto set_status = [&](const std::string& s) {
                {
                    std::lock_guard<std::mutex> lk(hub.mu);
                    t->status = s;
                }
                hub.publish({{"type", "table"}, {"table", t->id}, {"connector", t->spec.name}, {"status", s}});
            };
            set_status("playing");
            try {
                auto sum = run_session(
                    t->spec, plan, *engine, nullptr, &t->stop, [&](const json& e) {
                        json ev = e;
                        ev["table"] = t->id;
                        ev["connector"] = t->spec.name;
                        LuckHand lh;
                        bool lucky = false;
                        if (e["type"] == "hand" && t->spec.codec == "acpc") {
                            try {
                                lucky = luck_hand(e, t->spec.big, t->spec.stack, lh);
                            } catch (const std::exception&) {
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lk(hub.mu);
                            if (lucky) t->luck.push_back(lh);
                            if (e["type"] == "hand") {
                                ++t->done;
                                t->total_bb += e["result_bb"].get<double>();
                                t->recent.push_back(e);
                                if (t->recent.size() > 50) t->recent.pop_front();
                                ev["hands"] = t->done;
                                ev["bb100"] = t->total_bb / t->done * 100;
                                ev["won_bb"] = t->total_bb;
                                if (t->luck.size() >= 20) ev["luck"] = luck_estimate(t->luck).to_json();
                            }
                            t->state = ev;
                        }
                        hub.publish(ev);
                    });
                {
                    std::lock_guard<std::mutex> lk(hub.mu);
                    t->summary = sum.to_json();
                }
                set_status(t->stop ? "stopped" : "finished");
            } catch (const std::exception& e) {
                {
                    std::lock_guard<std::mutex> lk(hub.mu);
                    t->summary = {{"error", e.what()}};
                }
                set_status("failed");
            }
        });
        reply(res, {{"id", t->id}}, 201);
    });

    srv.Get("/v1/sessions", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(hub.mu);
        json a = json::array();
        for (auto& [id, t] : hub.tables) a.push_back(table_json(*t));
        reply(res, {{"sessions", a}, {"seq", hub.seq}});
    });

    srv.Get(R"(/v1/sessions/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(hub.mu);
        auto it = hub.tables.find(std::stoi(req.matches[1]));
        if (it == hub.tables.end()) return reply(res, {{"error", "no such table"}}, 404);
        json j = table_json(*it->second);
        j["state"] = it->second->state;
        j["recent"] = json(it->second->recent);
        reply(res, j);
    });

    srv.Delete(R"(/v1/sessions/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(hub.mu);
        auto it = hub.tables.find(std::stoi(req.matches[1]));
        if (it == hub.tables.end()) return reply(res, {{"error", "no such table"}}, 404);
        it->second->stop = true;
        reply(res, {{"stopping", it->first}});
    });

    srv.Get("/v1/events", [&](const httplib::Request& req, httplib::Response& res) {
        long from = 0;
        if (req.has_header("Last-Event-ID")) from = std::stol(req.get_header_value("Last-Event-ID"));
        else if (req.has_param("from")) from = std::stol(req.get_param_value("from"));
        else {
            std::lock_guard<std::mutex> lk(hub.mu);
            from = hub.seq;
        }
        auto last = std::make_shared<long>(from);
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [&hub, last](size_t, httplib::DataSink& sink) {
            std::vector<std::pair<long, json>> out;
            {
                std::unique_lock<std::mutex> lk(hub.mu);
                hub.cv.wait_for(lk, std::chrono::seconds(15),
                                [&] { return hub.quit || (!hub.events.empty() && hub.events.back().first > *last); });
                for (auto& e : hub.events)
                    if (e.first > *last) out.push_back(e);
            }
            if (hub.quit) {
                sink.done();
                return false;
            }
            std::string chunk;
            if (out.empty()) chunk = ": keep-alive\n\n";
            for (auto& [s, e] : out) {
                chunk += "id: " + std::to_string(s) + "\ndata: " + e.dump() + "\n\n";
                *last = s;
            }
            return sink.write(chunk.data(), chunk.size());
        });
    });

    const std::string p = std::to_string(o.port);
    // who may use the app: this computer always; other devices on the network only with --lan and the access token
    // (a cookie set by opening the link with ?token=, or an X-Kestrel-Token header). Writes must come from this app's
    // own pages (Origin), POST bodies must be JSON, and on this computer the Host must be a local name (DNS
    // rebinding).
    const std::string token = o.lan ? access_token(o.token_file) : "";
    const auto addrs = o.lan ? lan_addresses() : std::vector<std::string>{};
    srv.set_pre_routing_handler([p, token, addrs](const httplib::Request& req, httplib::Response& res) {
        auto deny = [&](const char* why) {
            res.status = 403;
            res.set_content(std::string(R"({"error":"refused: )") + why + "\"}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        };
        auto host = req.get_header_value("Host");
        bool loopback = req.remote_addr == "127.0.0.1" || req.remote_addr == "::1";
        bool local_host = host == "127.0.0.1:" + p || host == "localhost:" + p;
        for (auto& ip : addrs) local_host = local_host || host == ip + ":" + p;
        if (!local_host) return deny("unknown host name");
        if (!loopback) {
            if (token.empty()) return deny("this app accepts requests from this computer only");
            std::string got = req.get_header_value("X-Kestrel-Token");
            if (got.empty() && req.has_param("token")) got = req.get_param_value("token");
            if (got.empty()) {
                auto c = req.get_header_value("Cookie");
                auto k = c.find("kestrel_token=");
                if (k != std::string::npos) got = c.substr(k + 14, c.find(';', k) == std::string::npos ? std::string::npos : c.find(';', k) - k - 14);
            }
            if (got != token) return deny("this device needs the access link (with its token)");
            if (req.has_param("token")) {  // remember the device, and drop the token from the address bar
                res.set_header("Set-Cookie", "kestrel_token=" + token + "; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000");
                res.set_redirect(req.path);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        bool write = req.method != "GET" && req.method != "HEAD";
        auto origin = req.get_header_value("Origin");
        if (write && !origin.empty() && origin != "http://" + host) return deny("cross-site request");
        if (req.method == "POST" && req.get_header_value("Content-Type").rfind("application/json", 0) != 0)
            return deny("JSON bodies only");
        return httplib::Server::HandlerResponse::Unhandled;
    });
    srv.Get("/v1/network", [&, token, addrs](const httplib::Request& req, httplib::Response& res) {
        bool loopback = req.remote_addr == "127.0.0.1" || req.remote_addr == "::1";
        json urls = json::array();
        if (loopback)  // the access links are shown on this computer only
            for (auto& ip : addrs) urls.push_back("http://" + ip + ":" + std::to_string(o.port) + "/?token=" + token);
        reply(res, {{"lan", !token.empty()}, {"links", urls}});
    });
    if (!o.assets.empty()) srv.set_mount_point("/assets", o.assets);
    if (!o.web.empty()) srv.set_mount_point("/", o.web);
    srv.new_task_queue = [] { return new httplib::ThreadPool(16); };

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    const std::string url = "http://127.0.0.1:" + std::to_string(o.port) + "/";
    // loopback only: reach it from elsewhere through an SSH tunnel
    {  // one Kestrel per port: if it is already running, open it instead
        httplib::Client c("127.0.0.1", o.port);
        c.set_connection_timeout(1);
        auto r = c.Get("/v1/health");
        if (r && r->status == 200 && r->body.find("\"version\"") != std::string::npos) {
            std::printf("kestreld is already running: %s\n", url.c_str());
            if (o.open_browser) open_url(url);
            return 0;
        }
    }
    srv.set_socket_options([](socket_t sock) {  // no port sharing: a second server must fail to bind
        int yes = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&yes), sizeof yes);
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
#endif
    });
    if (!srv.bind_to_port(o.lan ? "0.0.0.0" : "127.0.0.1", o.port))
        throw std::runtime_error("port " + std::to_string(o.port) + " is in use by another program");
    std::printf("kestreld %s: %s  (Ctrl-C to stop)\n", VERSION, url.c_str());
    if (!pack_problem.empty()) std::printf("  note: %s; heads-up tables are unavailable\n", pack_problem.c_str());
    if (o.lan) {
        std::printf("  on your network (open once per device; the link carries the access token):\n");
        for (auto& ip : addrs) std::printf("    http://%s:%d/?token=%s\n", ip.c_str(), o.port, token.c_str());
    }
    std::fflush(stdout);
    if (o.open_browser) open_url(url);
    bool ok = srv.listen_after_bind();
    hub.quit = true;
    hub.cv.notify_all();
    std::vector<std::shared_ptr<Table>> ts;
    {
        std::lock_guard<std::mutex> lk(hub.mu);
        for (auto& [id, t] : hub.tables) t->stop = true, ts.push_back(t);
    }
    std::printf("kestreld: finishing the hands in play\n");
    for (auto& t : ts)
        if (t->th.joinable()) t->th.join();
    return ok ? 0 : 1;
}

}
