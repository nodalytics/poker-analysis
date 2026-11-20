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
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#endif

#include "kestrel/connector.hpp"
#include "kestrel/codec_acpc.hpp"
#include "kestrel/gateway.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/buckets.hpp"
#include "kestrel/luck.hpp"
#include "kestrel/net.hpp"
#include "kestrel/icm.hpp"
#include "kestrel/store.hpp"
#include "kestrel/opponents.hpp"
#include "kestrel/ninemax_spot.hpp"
#include "kestrel/practice.hpp"
#include "kestrel/models.hpp"
#include "kestrel/session.hpp"

namespace kestrel {

namespace {

constexpr const char* VERSION = "0.4.0";

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
                         {"available", ts}},
                        {{"name", "search"},
                         {"about", "9-max: search at decision time, " + std::to_string(o.search_samples) +
                                       " deals each, opponents' ranges from their actions. For strong, balanced opponents: +4 to +6 bb/100 over "
                                       "the network against nm25-like play, but it assumes everyone plays like the network and lost "
                                       "-96 bb/100 to calling stations in tests (search-exploit or search-robust there)"},
                         {"available", true}, {"players", 9}},
                        {{"name", "search-robust"},
                         {"about", "9-max: search that assumes the opponents may play tighter, looser or more aggressive "
                                   "(after Pluribus); never below the network in tests, about 3x slower than search"},
                         {"available", true}, {"players", 9}},
                        {{"name", "search-exploit"},
                         {"about", "9-max: search that also plays each opponent as its profile says (fold, call and "
                                   "raise tendencies from its statistics), weighted by the evidence: safe exploitation"},
                         {"available", true}, {"players", 9}}});
}

// 9-max solvers: none, search, search-robust or search-exploit; false for anything else
bool nine_max_solver(const std::string& s) {
    return s == "none" || s == "search" || s == "search-robust" || s == "search-exploit";
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
    std::map<std::string, std::shared_ptr<Net>> nets;
    std::shared_ptr<Net> net(const std::string& dir) {
        std::lock_guard<std::mutex> lk(mu);
        auto& n = nets[dir];
        if (!n) n = std::make_shared<Net>(dir);
        return n;
    }
};

// the 9-max tournament networks: installed packages (verified) and plain model folders with net.json
std::vector<std::pair<std::string, std::string>> nine_max_models(const ServeOptions& o) {
    std::vector<std::pair<std::string, std::string>> out;
    for (auto& m : list_models(o.models.empty() ? default_models_dir() : o.models))
        if (m.verified && m.players == 9 && std::filesystem::exists(std::filesystem::path(m.dir) / "net.json"))
            out.push_back({m.name, m.dir});
    std::vector<std::string> dirs = o.extra_models;
    if (!o.models.empty()) dirs.insert(dirs.begin(), o.models);
    for (auto& dir : dirs) {
        std::error_code ec;
        for (auto& d : std::filesystem::directory_iterator(dir, ec)) {
            if (!std::filesystem::exists(d.path() / "net.json")) continue;
            if (std::filesystem::exists(d.path() / "model.json") && !verify_model(d.path().string()).verified) continue;
            std::string n = d.path().filename().string();
            if (std::none_of(out.begin(), out.end(), [&](auto& x) { return x.first == n; }))
                out.push_back({n, d.path().string()});
        }
    }
    // the default first (new tables, practice and advice take the first), the others by name
    std::sort(out.begin(), out.end(), [&](auto& a, auto& b) {
        bool da = a.first == o.nine_max_default, db = b.first == o.nine_max_default;
        return da != db ? da : a.first < b.first;
    });
    return out;
}

json models_json(const ServeOptions& o) {
    json a = json::array();
    bool pack = std::filesystem::exists(std::filesystem::path(o.pack) / "tree.tsv");
    a.push_back({{"name", "hu-blueprint"}, {"game", "heads-up"}, {"players", 2}, {"available", pack},
                 {"practice", false},
                 {"about", pack ? "heads-up blueprint, 200bb (the data pack)"
                                : "heads-up blueprint: the data pack is not installed (" + o.pack + ")"}});
    for (auto& [name, dir] : nine_max_models(o)) {
        // the game it was trained for, from its net.json: tournament (ICM), bounty tournament (PKO) or cash
        std::ifstream f(std::filesystem::path(dir) / "net.json");
        json nj = json::parse(f, nullptr, false);
        const json feat = nj.is_object() ? nj.value("features", json::object()) : json::object();
        const bool tourney = feat.value("tourney", true), pko = feat.value("bounty", 0.0) > 0;
        const int seats = nj.is_object() ? nj.value("seats", 9) : 9;
        std::string game = pko ? "9-max bounty tournament" : tourney ? "9-max tournament" : std::to_string(seats) + "-max cash";
        std::string about = pko ? "progressive knockout (PKO) tournaments, 10-25bb: practice tournaments with bounties, advice and platforms"
                          : tourney ? "9-max tournament network, 10-25bb with ICM: practice tournaments, advice (/v1/decide) and platforms"
                                    : std::to_string(seats) + "-max cash network, 50-250bb: advice and cash platforms";
        if (nj.is_object() && nj.contains("note") && nj["note"].is_string()) about += " (" + nj["note"].get<std::string>() + ")";
        a.push_back({{"name", name}, {"game", game}, {"players", 9}, {"seats", seats}, {"available", pack},
                     {"practice", pack && tourney}, {"about", about}});
    }
    return a;
}

struct TableRun {
    int id = 0;
    std::string kind = "session";  // or "practice"
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
    // practice with a human seat: the table waits on the user's move
    std::mutex wait_mu;
    std::condition_variable wait_cv;
    json your_turn;       // the pending question, or null
    int answer = -2;      // the chosen slot (-2: none yet)
    int players = 9;
    bool human = false;
};

struct Hub {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<long, json>> events;
    long seq = 0;
    std::map<int, std::shared_ptr<TableRun>> tables;
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

// the access token for other devices: 40 hex digits in a file only its owner can read. Created atomically with mode
// 0600 in a 0700 folder; a token file others could read is replaced, since the token may have leaked
std::string access_token(const std::string& file) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir = fs::path(file).parent_path();
    fs::create_directories(dir, ec);
#ifndef _WIN32
    ::chmod(dir.c_str(), 0700);
    struct stat st {};
    if (::stat(file.c_str(), &st) == 0) {
        if ((st.st_mode & 077) == 0) {
            std::ifstream in(file);
            std::string t;
            if (in >> t && t.size() >= 32) return t;
        }
        fs::remove(file, ec);
    }
#else
    {
        std::ifstream in(file);
        std::string t;
        if (in >> t && t.size() >= 32) return t;
    }
#endif
    std::random_device rd;
    const char* hex = "0123456789abcdef";
    std::string t;
    for (int k = 0; k < 40; ++k) t += hex[rd() & 15];
#ifndef _WIN32
    int fd = ::open(file.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0) throw std::runtime_error("cannot create the access token file " + file);
    std::string line = t + "\n";
    bool ok = ::write(fd, line.data(), line.size()) == ssize_t(line.size());
    ::close(fd);
    if (!ok) throw std::runtime_error("cannot write the access token file " + file);
#else
    std::ofstream(file) << t << "\n";
#endif
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

json table_json(const TableRun& t) {
    double bb100 = t.done ? t.total_bb / t.done * 100 : 0;
    json j = {{"id", t.id},           {"kind", t.kind}, {"players", t.players}, {"human", t.human}, {"connector", t.spec.name}, {"status", t.status}, {"model", t.model},
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

    // the opponents' statistics, by platform and username, kept in the sessions folder across sessions
    static OpponentBook opponents(o.out + "/opponents.json");
    // the database: every session's and practice table's hands, and the opponents (the files are still written)
    static std::unique_ptr<Store> store;
    if (!o.db.empty()) {
        try {
            store = std::make_unique<Store>(o.db);
            opponents.set_store(store.get());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "kestreld: %s; continuing with files only\n", e.what());
        }
    }
    Store* db = store.get();
    auto int_param = [](const httplib::Request& req, const char* k, int dflt, int lo, int hi) {
        if (!req.has_param(k)) return dflt;
        try {
            return std::clamp(std::stoi(req.get_param_value(k)), lo, hi);
        } catch (const std::exception&) {
            return dflt;
        }
    };
    // GET /v1/db: where the database is and how much it holds
    srv.Get("/v1/db", [&](const httplib::Request&, httplib::Response& res) {
        if (!db) return reply(res, {{"enabled", false}});
        try {
            json j = db->stats();
            j["enabled"] = true;
            reply(res, j);
        } catch (const std::exception& e) {
            reply(res, {{"error", e.what()}}, 500);
        }
    });
    // GET /v1/db/sessions?limit=&offset=: sessions newest first, with their hands and result
    srv.Get("/v1/db/sessions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!db) return reply(res, {{"error", "no database"}}, 404);
        try {
            reply(res, {{"sessions", db->sessions(int_param(req, "limit", 100, 1, 1000), int_param(req, "offset", 0, 0, 1 << 30))}});
        } catch (const std::exception& e) {
            reply(res, {{"error", e.what()}}, 500);
        }
    });
    // GET /v1/spots/similar?spot=ID&k=10: the stored spots most like a stored one (same model, cosine similarity of
    // the network's embeddings), each with its hand; POST /v1/decide with "similar": k does the same for a new spot
    srv.Get("/v1/spots/similar", [&](const httplib::Request& req, httplib::Response& res) {
        if (!db) return reply(res, {{"error", "no database"}}, 404);
        try {
            const int64_t id = int_param(req, "spot", 0, 0, 1 << 30);
            std::string model;
            auto v = db->spot_vec(id, &model);
            if (v.empty()) return reply(res, {{"error", "no spot " + std::to_string(id)}}, 404);
            reply(res, {{"spot", id}, {"model", model}, {"similar", db->similar(model, v, int_param(req, "k", 10, 1, 100), id)}});
        } catch (const std::exception& e) {
            reply(res, {{"error", e.what()}}, 500);
        }
    });
    // GET /v1/hands?session=&connector=&limit=&offset=: hands newest first, each with its whole record
    srv.Get("/v1/hands", [&](const httplib::Request& req, httplib::Response& res) {
        if (!db) return reply(res, {{"error", "no database"}}, 404);
        try {
            reply(res, {{"hands", db->hands(int_param(req, "session", 0, 0, 1 << 30),
                                            req.has_param("connector") ? req.get_param_value("connector") : "",
                                            int_param(req, "limit", 100, 1, 1000), int_param(req, "offset", 0, 0, 1 << 30))}});
        } catch (const std::exception& e) {
            reply(res, {{"error", e.what()}}, 500);
        }
    });
    srv.Get("/v1/opponents", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.has_param("platform") && req.has_param("player")) {  // one player: statistics, profile, look-alikes
            const std::string p = req.get_param_value("platform"), who = req.get_param_value("player");
            json all = opponents.platform(p);
            if (!all.contains(who)) return reply(res, {{"error", "no player " + who + " on " + p}}, 404);
            return reply(res, {{"platform", p}, {"player", who}, {"stats", all[who]},
                               {"like", opponents.similar(p, who, 5, req.has_param("all"))}});
        }
        if (req.has_param("platform")) return reply(res, {{"platform", req.get_param_value("platform")},
                                                          {"players", opponents.platform(req.get_param_value("platform"))}});
        reply(res, {{"platforms", opponents.platforms()}});
    });

    // the desktop app restarts the server after changing settings; only from this computer
    srv.Post("/v1/admin/shutdown", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.remote_addr != "127.0.0.1" && req.remote_addr != "::1")
            return reply(res, {{"error", "only from this computer"}}, 403);
        reply(res, {{"stopping", true}});
        std::thread([&srv] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            srv.stop();
        }).detach();
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
        std::vector<std::string> errors;
        auto cs = all_connectors(o, &errors);
        auto it = std::find_if(cs.begin(), cs.end(), [&](auto& c) { return c.name == name; });
        if (it == cs.end()) return reply(res, {{"error", "no connector " + name}}, 404);
        const bool table9 = it->codec == "structured";
        auto ms = nine_max_models(o);
        std::string model = body.value("model", table9 ? (ms.empty() ? "" : ms.front().first) : "hu-blueprint");
        std::string solver = body.value("solver", "none");
        std::string net_dir;
        if (table9) {
            auto m = std::find_if(ms.begin(), ms.end(), [&](auto& x) { return x.first == model; });
            if (m == ms.end()) return reply(res, {{"error", "no 9-max model " + model + " for this connector"}}, 400);
            if (!nine_max_solver(solver))
                return reply(res, {{"error", "9-max tables take solver none, search, search-robust or search-exploit"}}, 400);
            net_dir = m->second;
        } else {
            if (model != "hu-blueprint") return reply(res, {{"error", "model " + model + " cannot play this connector"}}, 400);
            if (solver == "search" || solver == "search-robust" || solver == "search-exploit")
                return reply(res, {{"error", "search is for 9-max tables"}}, 400);
            bool known = false;
            for (auto& x : solvers_json(o))
                if (x["name"] == solver && x["available"] == true) known = true;
            if (!known) return reply(res, {{"error", "solver " + solver + " is not available"}}, 400);
        }
        const int tournaments = std::clamp(body.value("tournaments", 1), 1, 10000);
        auto t = std::make_shared<TableRun>();
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
        std::shared_ptr<Net> net;
        try {
            if (table9) {
                buckets_init(o.pack);
                net = engines.net(net_dir);
            } else {
                engine = engines.get(o, model, solver);
            }
        } catch (const std::exception& e) {
            return reply(res, {{"error", e.what()}}, 400);
        }
        t->th = std::thread([&hub, engine, net, tournaments, &o, t, db] {
            SessionPlan plan;
            plan.model = t->model, plan.solver = t->solver;
            plan.hands = t->hands;
            plan.tournaments = tournaments;
            plan.opponents = &opponents;
            plan.store = db;
            if (t->solver == "search" || t->solver == "search-robust" || t->solver == "search-exploit")
                plan.search_samples = o.search_samples, plan.search_robust = t->solver == "search-robust",
                plan.search_exploit = t->solver == "search-exploit";
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
                auto on_event = [&](const json& e) {
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
                    };
                auto sum = net ? run_ninemax_session(t->spec, plan, *net, nullptr, &t->stop, on_event)
                               : run_session(t->spec, plan, *engine, nullptr, &t->stop, on_event);
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

    // a practice tournament: a 9-max network in every seat, played at a watchable pace
    srv.Post("/v1/practice", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return reply(res, {{"error", "bad JSON"}}, 400);
        std::string model = body.value("model", "");
        auto ms = nine_max_models(o);
        auto it = std::find_if(ms.begin(), ms.end(), [&](auto& m) { return m.first == model; });
        if (it == ms.end()) return reply(res, {{"error", "no 9-max model " + model}}, 404);
        if (!pack_problem.empty()) return reply(res, {{"error", pack_problem}}, 400);
        std::shared_ptr<Net> net;
        try {
            buckets_init(o.pack);
            net = std::make_shared<Net>(it->second);
        } catch (const std::exception& e) {
            return reply(res, {{"error", e.what()}}, 400);
        }
        PracticeConfig pc;
        pc.players = std::clamp(body.value("players", 9), 2, 9);
        pc.avg_bb = std::clamp(body.value("avg_bb", 20.0), 5.0, 60.0);
        int hands = std::clamp(body.value("hands", 300), 1, 100000);
        int pace = std::clamp(body.value("pace_ms", 700), 0, 5000);
        pc.human = body.value("human", false);
        // knockout bounties: a PKO model plays with the share it trained with unless told otherwise; others never
        pc.bounty = net->info().feat_bounty > 0 ? std::clamp(body.value("bounty", net->info().feat_bounty), 0.0, 0.9) : 0.0;
        auto t = std::make_shared<TableRun>();
        {
            std::lock_guard<std::mutex> lk(hub.mu);
            t->id = hub.next_id++;
            t->kind = "practice", t->model = model, t->solver = "none", t->hands = hands;
            t->players = pc.players, t->human = pc.human;
            t->spec.name = "practice";
            hub.tables[t->id] = t;
        }
        hub.publish({{"type", "table"}, {"table", t->id}, {"connector", "practice"}, {"kind", "practice"},
                     {"status", "playing"}, {"model", model}, {"solver", "none"}, {"players", pc.players},
                     {"human", pc.human}});
        const std::string logdir = o.out + "/" + std::to_string(t->id) + "-practice";
        t->th = std::thread([&hub, t, net, pc, pace, logdir, db, model] {
            {
                std::lock_guard<std::mutex> lk(hub.mu);
                t->status = "playing";
            }
            std::error_code ec;
            std::filesystem::create_directories(logdir, ec);
            std::ofstream log(logdir + "/hands.jsonl", std::ios::app);  // every hand, for study afterwards
            int64_t sid = 0;
            auto in_db = [&](auto f) {  // the database is a copy of the file: an error never stops the table
                if (!db) return;
                try {
                    f();
                } catch (const std::exception&) {
                }
            };
            in_db([&] { sid = db->begin_session("practice", "practice", model, "none", logdir); });
            uint64_t round = 0;
            while (!t->stop && t->done < t->hands) {
                PracticeConfig c = pc;
                c.seed = uint64_t(t->id) * 1000003 + round++;
                PracticeTournament tour(*net, c, "Kestrel");
                while (!tour.over() && !t->stop && t->done < t->hands) {
                    auto ask = [&](const json& q) {
                        json ev = q;
                        ev["table"] = t->id, ev["connector"] = "practice", ev["kind"] = "practice";
                        {
                            std::lock_guard<std::mutex> lk(t->wait_mu);
                            t->your_turn = ev;
                            t->answer = -2;
                        }
                        {
                            std::lock_guard<std::mutex> lk(hub.mu);
                            t->state = ev;
                        }
                        hub.publish(ev);
                        std::unique_lock<std::mutex> lk(t->wait_mu);
                        t->wait_cv.wait(lk, [&] { return t->answer != -2 || t->stop; });
                        t->your_turn = nullptr;
                        return t->stop ? -1 : t->answer;
                    };
                    tour.play_hand([&](const json& e) {
                        json ev = e;
                        ev["table"] = t->id, ev["connector"] = "practice", ev["kind"] = "practice";
                        ev["tournament"] = round;
                        {
                            std::lock_guard<std::mutex> lk(hub.mu);
                            if (e["type"] == "practice_hand") ev["hands"] = ++t->done, t->summary = e["standings"];
                            t->state = ev;
                        }
                        hub.publish(ev);
                        if (e["type"] == "practice_hand" && log) log << ev.dump() << "\n" << std::flush;
                        if (e["type"] == "practice_hand" && sid) {
                            json row = ev;
                            row["model"] = model;
                            in_db([&] { db->add_hand(sid, row); });
                        }
                        if (pace && e["type"] == "practice_decision") std::this_thread::sleep_for(std::chrono::milliseconds(pace));
                    }, pc.human ? std::function<int(const json&)>(ask) : nullptr);
                    if (pace) std::this_thread::sleep_for(std::chrono::milliseconds(pace * 2));
                }
            }
            {
                std::lock_guard<std::mutex> lk(hub.mu);
                t->status = t->stop ? "stopped" : "finished";
            }
            if (sid) {
                json standings;
                {
                    std::lock_guard<std::mutex> lk(hub.mu);
                    standings = t->summary;
                }
                in_db([&] { db->end_session(sid, standings); });
            }
            hub.publish({{"type", "table"}, {"table", t->id}, {"connector", "practice"}, {"kind", "practice"},
                         {"status", t->stop ? "stopped" : "finished"}});
        });
        reply(res, {{"id", t->id}}, 201);
    });

    // the user's move at a practice table: {"slot": n}, one of the choices of the pending practice_your_turn
    srv.Post(R"(/v1/practice/(\d+)/act)", [&](const httplib::Request& req, httplib::Response& res) {
        std::shared_ptr<TableRun> t;
        {
            std::lock_guard<std::mutex> lk(hub.mu);
            auto it = hub.tables.find(std::stoi(req.matches[1]));
            if (it == hub.tables.end()) return reply(res, {{"error", "no such table"}}, 404);
            t = it->second;
        }
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("slot") || !body["slot"].is_number_integer())
            return reply(res, {{"error", "need slot"}}, 400);
        int slot = body["slot"];
        {
            std::lock_guard<std::mutex> lk(t->wait_mu);
            if (t->your_turn.is_null()) return reply(res, {{"error", "not your turn"}}, 409);
            bool ok = false;
            for (auto& c : t->your_turn["choices"]) ok = ok || c["slot"] == slot;
            if (!ok) return reply(res, {{"error", "not a legal move"}}, 400);
            t->answer = slot;
        }
        t->wait_cv.notify_all();
        reply(res, {{"ok", true}});
    });

    // advice for one spot. Heads-up: {"game": "headsup", "action": ACPC string, "pos": 0|1, "hole": [..], "board": [..],
    // "solver": none|kestrel-river|texassolver}
    // prize equity for stacks and payouts (place 1 first): model mh (Malmuth-Harville, what the networks read) or mw
    // (Malmuth-Weitzman), for analysis
    srv.Post("/v1/icm", [&](const httplib::Request& req, httplib::Response& res) {
        json b = json::parse(req.body, nullptr, false);
        if (!b.is_object()) return reply(res, {{"error", "bad JSON"}}, 400);
        try {
            auto st = b.at("stacks").get<std::vector<double>>();
            auto pay = b.at("payouts").get<std::vector<double>>();
            if (st.empty() || st.size() > 20) return reply(res, {{"error", "1 to 20 stacks"}}, 400);
            std::string model = b.value("model", "mh");
            if (model != "mh" && model != "mw") return reply(res, {{"error", "model is mh or mw"}}, 400);
            reply(res, {{"equity", model == "mw" ? icm_mw(st, pay) : icm(st, pay)}, {"model", model}});
        } catch (const std::exception& e) {
            reply(res, {{"error", e.what()}}, 400);
        }
    });

    srv.Post("/v1/decide", [&](const httplib::Request& req, httplib::Response& res) {
        json b = json::parse(req.body, nullptr, false);
        if (b.is_discarded()) return reply(res, {{"error", "bad JSON"}}, 400);
        std::string game = b.value("game", "");
        if (game != "headsup" && game != "ninemax") return reply(res, {{"error", "game must be headsup or ninemax"}}, 400);
        if (!pack_problem.empty()) return reply(res, {{"error", pack_problem}}, 400);
        if (game == "ninemax") {
            auto ms = nine_max_models(o);
            if (ms.empty()) return reply(res, {{"error", "no 9-max model is installed"}}, 404);
            std::string model = b.value("model", ms.front().first);
            auto it = std::find_if(ms.begin(), ms.end(), [&](auto& m) { return m.first == model; });
            if (it == ms.end()) return reply(res, {{"error", "no 9-max model " + model}}, 404);
            try {
                const std::string solver = b.value("solver", "none");
                if (!nine_max_solver(solver))
                    return reply(res, {{"error", "solver is none, search, search-robust or search-exploit"}}, 400);
                SearchConfig sc;
                sc.samples = std::clamp(b.value("samples", o.search_samples), 1024, 4096), sc.robust = solver == "search-robust";
                if (solver == "search-exploit") sc.exploit_weight = 0.5;  // tilts: the request's "exploit" {id: {fold, call, raise}}
                sc.seed = std::random_device{}();
                buckets_init(o.pack);
                auto adv = ninemax_decide(*engines.net(it->second), NMSpot::from_json(b), solver == "none" ? nullptr : &sc);
                json j = adv.to_json();
                j["model"] = model, j["solver"] = solver;
                if (const int k = std::clamp(b.value("similar", 0), 0, 100); k > 0 && db && adv.error.empty())
                    j["similar"] = db->similar(model, adv.embedding, k);
                return reply(res, j, adv.error.empty() ? 200 : 400);
            } catch (const std::exception& e) {
                return reply(res, {{"error", e.what()}}, 400);
            }
        }
        try {
            auto eng = engines.get(o, "hu-blueprint", b.value("solver", "none"));
            HUSpot sp;
            acpc::Game g{b.value("big", 100), b.value("stack", 20000)};
            if (!acpc::spot(b.value("action", ""), b.value("pos", 0), b.value("hole", std::vector<std::string>{}),
                            b.value("board", std::vector<std::string>{}), g, sp))
                return reply(res, {{"error", "it is not this position's turn in that action string"}}, 400);
            auto ans = eng->decide(sp);
            json opts = json::array();
            for (auto& x : ans.options)
                opts.push_back({{"action", x.kind == 'f' ? "fold" : x.kind == 'k' ? (sp.facing ? "call" : "check") : "raise"},
                                {"to", x.to}, {"prob", x.prob}, {"send", acpc::encode(x, sp)}});
            reply(res, {{"on_tree", ans.on_tree}, {"resolved", ans.resolved}, {"floored", ans.floored},
                        {"options", opts}});
        } catch (const std::exception& e) {
            reply(res, {{"error", e.what()}}, 400);
        }
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
        it->second->wait_cv.notify_all();
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
    FILE* out = o.quiet ? stderr : stdout;
    std::fprintf(out, "kestreld %s: %s  (Ctrl-C to stop)\n", VERSION, url.c_str());
    if (o.ws_port) start_websocket(o.port, o.ws_port, hub.quit), std::fprintf(out, "  WebSocket ws://127.0.0.1:%d/\n", o.ws_port);
    if (o.tcp_port) start_tcp(o.port, o.tcp_port, hub.quit), std::fprintf(out, "  TCP 127.0.0.1:%d (JSON-RPC, a line each)\n", o.tcp_port);
    if (!pack_problem.empty()) std::fprintf(out, "  note: %s; heads-up tables are unavailable\n", pack_problem.c_str());
    if (o.lan) {
        std::fprintf(out, "  on your network (open once per device; the link carries the access token):\n");
        for (auto& ip : addrs) std::fprintf(out, "    http://%s:%d/?token=%s\n", ip.c_str(), o.port, token.c_str());
    }
    std::fflush(out);
    if (o.open_browser) open_url(url);
    bool ok = srv.listen_after_bind();
    opponents.flush();
    hub.quit = true;
    hub.cv.notify_all();
    std::vector<std::shared_ptr<TableRun>> ts;
    {
        std::lock_guard<std::mutex> lk(hub.mu);
        for (auto& [id, t] : hub.tables) t->stop = true, t->wait_cv.notify_all(), ts.push_back(t);
    }
    std::printf("kestreld: finishing the hands in play\n");
    for (auto& t : ts)
        if (t->th.joinable()) t->th.join();
    return ok ? 0 : 1;
}

}
