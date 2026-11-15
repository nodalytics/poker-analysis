#include "kestrel/gateway.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include <httplib.h>
#include <ixwebsocket/IXWebSocketServer.h>

#include "kestrel/icm.hpp"

namespace kestrel {

using json = nlohmann::json;

namespace {

json obj_schema(json props, json required = json::array()) {
    return {{"type", "object"}, {"properties", std::move(props)}, {"required", std::move(required)}};
}

struct Method {
    const char* name;
    const char* about;
    const char* http;  // GET POST DELETE, or "" for local
    const char* path;  // {id} is filled from params.id
    json schema;
};

const std::vector<Method>& methods() {
    static const std::vector<Method> m = {
        {"health", "Kestrel's version and how many tables are playing", "GET", "/v1/health", obj_schema(json::object())},
        {"list_connectors", "the platforms Kestrel can play (connector files), what each talks to", "GET",
         "/v1/connectors", obj_schema(json::object())},
        {"list_engines", "the models and solvers a table can play with", "GET", "/v1/engines",
         obj_schema(json::object())},
        {"start_session", "start a table on a platform: connector, hands, model, solver", "POST", "/v1/sessions",
         obj_schema({{"connector", {{"type", "string"}}}, {"hands", {{"type", "integer"}}},
                     {"model", {{"type", "string"}}}, {"solver", {{"type", "string"}}}},
                    {"connector"})},
        {"list_sessions", "every table: kind, status, hands, results", "GET", "/v1/sessions",
         obj_schema(json::object())},
        {"session_status", "one table: its live state and last hands", "GET", "/v1/sessions/{id}",
         obj_schema({{"id", {{"type", "integer"}}}}, {"id"})},
        {"stop_session", "stop a table after the hand in play", "DELETE", "/v1/sessions/{id}",
         obj_schema({{"id", {{"type", "integer"}}}}, {"id"})},
        {"start_practice", "a practice tournament with a 9-max model: watch it, or play (human: true)", "POST",
         "/v1/practice",
         obj_schema({{"model", {{"type", "string"}}}, {"players", {{"type", "integer"}}},
                     {"avg_bb", {{"type", "number"}}}, {"hands", {{"type", "integer"}}},
                     {"pace_ms", {{"type", "integer"}}}, {"human", {{"type", "boolean"}}}},
                    {"model"})},
        {"practice_act", "your move at a practice table: one of the slots it offered", "POST",
         "/v1/practice/{id}/act", obj_schema({{"id", {{"type", "integer"}}}, {"slot", {{"type", "integer"}}}}, {"id", "slot"})},
        {"decide", "Kestrel's advice for a spot: game headsup (ACPC action string) or ninemax (table state)", "POST",
         "/v1/decide", obj_schema({{"game", {{"type", "string"}}}}, {"game"})},
        {"icm_equity", "ICM prize equity for stacks and payouts (place 1 first); model mh (Malmuth-Harville, default) or mw (Malmuth-Weitzman)", "", "",
         obj_schema({{"stacks", {{"type", "array"}, {"items", {{"type", "number"}}}}},
                     {"payouts", {{"type", "array"}, {"items", {{"type", "number"}}}}},
                     {"model", {{"type", "string"}, {"enum", {"mh", "mw"}}}}},
                    {"stacks", "payouts"})},
    };
    return m;
}

json rpc_error(const json& id, int code, const std::string& msg) {
    return {{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", msg}}}};
}

}  // namespace

json rpc_methods() {
    json a = json::array();
    for (auto& m : methods()) a.push_back({{"name", m.name}, {"description", m.about}, {"inputSchema", m.schema}});
    return a;
}

namespace {
json dispatch(int api_port, const json& req) {
    json id = req.is_object() && req.contains("id") ? req["id"] : json(nullptr);
    bool notify = !req.contains("id");
    if (!req.is_object() || !req.contains("method") || !req["method"].is_string()) return rpc_error(id, -32600, "invalid request");
    std::string name = req["method"];
    json params = req.value("params", json::object());
    if (!params.is_object()) return rpc_error(id, -32602, "params must be an object");
    const Method* m = nullptr;
    for (auto& x : methods())
        if (name == x.name) m = &x;
    if (!m) return notify ? json(nullptr) : rpc_error(id, -32601, "no method " + name);
    json result;
    if (name == "icm_equity") {
        try {
                auto st = params.at("stacks").get<std::vector<double>>();
            auto pay = params.at("payouts").get<std::vector<double>>();
            if (st.empty() || st.size() > 20) return rpc_error(id, -32602, "1 to 20 stacks");
            const std::string model = params.value("model", "mh");
            if (model != "mh" && model != "mw") return rpc_error(id, -32602, "model is mh or mw");
            result = {{"equity", model == "mw" ? icm_mw(st, pay) : icm(st, pay)}, {"model", model}};
        } catch (const std::exception& e) {
            return rpc_error(id, -32602, e.what());
        }
    } else {
        std::string path = m->path;
        auto k = path.find("{id}");
        if (k != std::string::npos) {
            if (!params.contains("id") || !params["id"].is_number_integer()) return rpc_error(id, -32602, "needs an integer id");
            path.replace(k, 4, std::to_string(params["id"].get<long long>()));
        }
        httplib::Client c("127.0.0.1", api_port);
        c.set_read_timeout(120);
        httplib::Headers h{{"Origin", "http://127.0.0.1:" + std::to_string(api_port)}};
        std::string body = params.dump();
        httplib::Result r = std::string(m->http) == "GET"      ? c.Get(path, h)
                            : std::string(m->http) == "DELETE" ? c.Delete(path, h)
                                                               : c.Post(path, h, body, "application/json");
        if (!r) return rpc_error(id, -32000, "Kestrel is not answering on port " + std::to_string(api_port));
        json j = json::parse(r->body, nullptr, false);
        if (j.is_discarded()) return rpc_error(id, -32000, "unreadable answer");
        if (r->status >= 400) return rpc_error(id, -32000, j.value("error", "HTTP " + std::to_string(r->status)));
        result = j;
    }
    if (notify) return nullptr;
    return {{"jsonrpc", "2.0"}, {"id", id}, {"result", result}};
}

// a browser page may only use the gateway from Kestrel's own pages; programs send no Origin
bool origin_ok(const std::string& origin, int api_port) {
    if (origin.empty()) return true;
    for (const char* h : {"127.0.0.1", "localhost", "[::1]"})
        if (origin == std::string("http://") + h + ":" + std::to_string(api_port)) return true;
    return false;
}
}  // namespace

json rpc_dispatch(int api_port, const json& req) {
    try {
        return dispatch(api_port, req);
    } catch (const std::exception& e) {
        return rpc_error(nullptr, -32603, e.what());
    }
}

// ---- WebSocket: JSON-RPC, plus {"method": "subscribe"} for the live events of every table ----
void start_websocket(int api_port, int ws_port, const std::atomic<bool>& quit) {
    std::thread([api_port, ws_port, &quit] {
        ix::WebSocketServer server(ws_port, "127.0.0.1");
        std::mutex mu;
        std::set<std::shared_ptr<ix::WebSocket>> subscribers;
        std::set<const ix::WebSocket*> accepted;
        server.setOnClientMessageCallback([&](std::shared_ptr<ix::ConnectionState>, ix::WebSocket& ws,
                                              const ix::WebSocketMessagePtr& msg) {
            if (msg->type == ix::WebSocketMessageType::Open) {
                auto it = msg->openInfo.headers.find("Origin");
                if (origin_ok(it == msg->openInfo.headers.end() ? "" : it->second, api_port)) {
                    std::lock_guard<std::mutex> lk(mu);
                    accepted.insert(&ws);
                } else {
                    ws.close(1008, "this origin may not use Kestrel");
                }
                return;
            }
            if (msg->type == ix::WebSocketMessageType::Close) {
                std::lock_guard<std::mutex> lk(mu);
                accepted.erase(&ws);
                for (auto it = subscribers.begin(); it != subscribers.end();)
                    it = it->get() == &ws ? subscribers.erase(it) : std::next(it);
                return;
            }
            if (msg->type != ix::WebSocketMessageType::Message) return;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!accepted.count(&ws)) return;
            }
            json req = json::parse(msg->str, nullptr, false);
            if (req.is_discarded()) {
                ws.send(rpc_error(nullptr, -32700, "parse error").dump());
                return;
            }
            if (req.is_object() && req.value("method", "") == "subscribe") {
                for (auto& c : server.getClients())
                    if (c.get() == &ws) {
                        std::lock_guard<std::mutex> lk(mu);
                        subscribers.insert(c);
                    }
                if (req.contains("id")) ws.send(json{{"jsonrpc", "2.0"}, {"id", req["id"]}, {"result", {{"subscribed", true}}}}.dump());
                return;
            }
            json resp = rpc_dispatch(api_port, req);
            if (!resp.is_null()) ws.send(resp.dump());
        });
        if (!server.listen().first) {
            std::fprintf(stderr, "kestreld: WebSocket port %d is busy; WebSocket is off\n", ws_port);
            return;
        }
        server.start();
        // the live events: read the server-sent event stream and push each event to the subscribers
        std::thread([&] {
            while (!quit) {
                httplib::Client c("127.0.0.1", api_port);
                c.set_read_timeout(30);
                std::string buf;
                c.Get("/v1/events", [&](const char* d, size_t n) {
                    buf.append(d, n);
                    size_t k;
                    while ((k = buf.find("\n\n")) != std::string::npos) {
                        std::string block = buf.substr(0, k);
                        buf.erase(0, k + 2);
                        auto p = block.find("data: ");
                        if (p == std::string::npos) continue;
                        std::string ev = json{{"jsonrpc", "2.0"}, {"method", "event"},
                                              {"params", json::parse(block.substr(p + 6), nullptr, false)}}.dump();
                        std::lock_guard<std::mutex> lk(mu);
                        for (auto& s : subscribers) s->send(ev);
                    }
                    return !quit.load();
                });
                if (!quit) std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }).detach();
        while (!quit) std::this_thread::sleep_for(std::chrono::milliseconds(200));
        server.stop();
    }).detach();
}

// ---- TCP: one JSON-RPC request per line, one response per line ----
void start_tcp(int api_port, int tcp_port, const std::atomic<bool>& quit) {
    std::thread([api_port, tcp_port, &quit] {
        socket_t sock = ::socket(AF_INET, SOCK_STREAM, 0);
        int yes = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof yes);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(uint16_t(tcp_port));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (sock == INVALID_SOCKET || ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 ||
            ::listen(sock, 16) != 0) {
            std::fprintf(stderr, "kestreld: TCP port %d is busy; TCP is off\n", tcp_port);
            if (sock != INVALID_SOCKET) httplib::detail::close_socket(sock);
            return;
        }
        while (!quit) {
            if (httplib::detail::select_read(sock, 0, 200000) <= 0) continue;
            socket_t c = ::accept(sock, nullptr, nullptr);
            if (c == INVALID_SOCKET) continue;
            std::thread([c, api_port] {
                std::string buf;
                bool first = true;
                char tmp[4096];
                while (true) {
                    auto n = ::recv(c, tmp, sizeof tmp, 0);
                    if (n <= 0) break;
                    buf.append(tmp, size_t(n));
                    if (buf.size() > (1 << 20)) break;  // a line longer than 1 MB: not a client of ours
                    size_t k;
                    while ((k = buf.find('\n')) != std::string::npos) {
                        std::string line = buf.substr(0, k);
                        buf.erase(0, k + 1);
                        if (line.empty()) continue;
                        json req = json::parse(line, nullptr, false);
                        if (first && !req.is_object()) {
                            buf.clear();
                            goto done;
                        }
                        first = false;
                        json resp = req.is_discarded() ? rpc_error(nullptr, -32700, "parse error") : rpc_dispatch(api_port, req);
                        if (resp.is_null()) continue;
                        std::string out = resp.dump() + "\n";
                        ::send(c, out.data(), out.size(), 0);
                    }
                }
            done:
                httplib::detail::close_socket(c);
            }).detach();
        }
        httplib::detail::close_socket(sock);
    }).detach();
}

// ---- MCP over stdio ----
int mcp_stdio(int api_port, const std::string& version) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json req = json::parse(line, nullptr, false);
        if (req.is_discarded()) {
            std::cout << rpc_error(nullptr, -32700, "parse error").dump() << std::endl;
            continue;
        }
        if (!req.is_object()) {
            std::cout << rpc_error(nullptr, -32600, "invalid request").dump() << std::endl;
            continue;
        }
        if (!req.contains("id")) continue;  // notifications (initialized, cancelled): nothing to answer
        std::string m = req.value("method", "");
        json id = req["id"];
        json result;
        try {
        if (m == "initialize") {
            std::string pv = req.contains("params") ? req["params"].value("protocolVersion", "2025-06-18") : "2025-06-18";
            result = {{"protocolVersion", pv},
                      {"capabilities", {{"tools", json::object()}}},
                      {"serverInfo", {{"name", "kestrel"}, {"version", version}}},
                      {"instructions", "Kestrel plays poker (heads-up against platforms, 9-max tournament practice) "
                                       "and gives advice. Use list_engines and list_connectors first."}};
        } else if (m == "ping") {
            result = json::object();
        } else if (m == "tools/list") {
            result = {{"tools", rpc_methods()}};
        } else if (m == "tools/call") {
            json p = req.value("params", json::object());
            json r = rpc_dispatch(api_port, {{"jsonrpc", "2.0"}, {"id", 1}, {"method", p.value("name", "")},
                                             {"params", p.value("arguments", json::object())}});
            bool err = r.contains("error");
            result = {{"content", {{{"type", "text"}, {"text", (err ? r["error"]["message"] : r["result"]).dump(1)}}}},
                      {"isError", err}};
        } else {
            std::cout << rpc_error(id, -32601, "no method " + m).dump() << std::endl;
            continue;
        }
        } catch (const std::exception& e) {
            std::cout << rpc_error(id, -32602, e.what()).dump() << std::endl;
            continue;
        }
        std::cout << json{{"jsonrpc", "2.0"}, {"id", id}, {"result", result}}.dump() << std::endl;
    }
    return 0;
}

}  // namespace kestrel
