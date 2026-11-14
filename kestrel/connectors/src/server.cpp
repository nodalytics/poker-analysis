#include "kestrel/server.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include <httplib.h>

#include "kestrel/connector.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/session.hpp"

namespace kestrel {

namespace {

constexpr const char* VERSION = "0.1.0";

struct Table {
    int id = 0;
    ConnectorSpec spec;
    int hands = 0;
    std::atomic<bool> stop{false};
    std::thread th;
    std::string status = "starting";
    json state;
    std::deque<json> recent;
    int done = 0;
    double total_bb = 0;
    json summary;
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

json table_json(const Table& t) {
    double bb100 = t.done ? t.total_bb / t.done * 100 : 0;
    json j = {{"id", t.id},           {"connector", t.spec.name}, {"status", t.status},
              {"hands_target", t.hands}, {"hands", t.done},       {"bb100", bb100},
              {"won_bb", t.total_bb}};
    if (!t.summary.is_null()) j["summary"] = t.summary;
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
    HUBlueprint engine(o.pack);
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

    srv.Post("/v1/sessions", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("connector")) return reply(res, {{"error", "need connector"}}, 400);
        std::string name = body["connector"];
        int hands = std::clamp(body.value("hands", 100), 1, 1000000);
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
            hub.tables[t->id] = t;
        }
        hub.publish({{"type", "table"}, {"table", t->id}, {"connector", name}, {"status", "starting"}});
        t->th = std::thread([&hub, &engine, &o, t] {
            SessionPlan plan;
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
                    t->spec, plan, engine, nullptr, &t->stop, [&](const json& e) {
                        json ev = e;
                        ev["table"] = t->id;
                        ev["connector"] = t->spec.name;
                        {
                            std::lock_guard<std::mutex> lk(hub.mu);
                            if (e["type"] == "hand") {
                                ++t->done;
                                t->total_bb += e["result_bb"].get<double>();
                                t->recent.push_back(e);
                                if (t->recent.size() > 50) t->recent.pop_front();
                                ev["hands"] = t->done;
                                ev["bb100"] = t->total_bb / t->done * 100;
                                ev["won_bb"] = t->total_bb;
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
    srv.set_pre_routing_handler([p](const httplib::Request& req, httplib::Response& res) {
        auto host = req.get_header_value("Host");
        bool local = host == "127.0.0.1:" + p || host == "localhost:" + p;
        bool write = req.method != "GET" && req.method != "HEAD";
        auto origin = req.get_header_value("Origin");
        bool same = origin.empty() || origin == "http://127.0.0.1:" + p || origin == "http://localhost:" + p;
        bool json_body = req.method != "POST" || req.get_header_value("Content-Type").rfind("application/json", 0) == 0;
        if (!local || (write && !same) || !json_body) {
            res.status = 403;
            res.set_content(R"({"error":"refused: local requests from this app only"})", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    if (!o.assets.empty()) srv.set_mount_point("/assets", o.assets);
    if (!o.web.empty()) srv.set_mount_point("/", o.web);
    srv.new_task_queue = [] { return new httplib::ThreadPool(16); };

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::printf("kestreld %s: http://127.0.0.1:%d/  (Ctrl-C to stop)\n", VERSION, o.port);
    std::fflush(stdout);
    bool ok = srv.listen("127.0.0.1", o.port);  // loopback only: reach it from elsewhere through an SSH tunnel
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
