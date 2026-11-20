#include "kestrel/connector.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <stdexcept>
#include <thread>

#include <yaml-cpp/yaml.h>

#include <httplib.h>
#include <ixwebsocket/IXWebSocket.h>

#include <condition_variable>
#include <deque>
#include <mutex>

namespace kestrel {

namespace {

json from_yaml(const YAML::Node& n) {
    switch (n.Type()) {
        case YAML::NodeType::Map: {
            json o = json::object();
            for (auto it = n.begin(); it != n.end(); ++it) o[it->first.as<std::string>()] = from_yaml(it->second);
            return o;
        }
        case YAML::NodeType::Sequence: {
            json a = json::array();
            for (const auto& x : n) a.push_back(from_yaml(x));
            return a;
        }
        case YAML::NodeType::Scalar: {
            const std::string& s = n.Scalar();
            if (n.Tag() == "!") return s;
            if (s == "true" || s == "false") return s == "true";
            if (s == "null" || s == "~") return nullptr;
            try {
                size_t k;
                long long v = std::stoll(s, &k);
                if (k == s.size()) return v;
                double d = std::stod(s, &k);
                if (k == s.size()) return d;
            } catch (...) {
            }
            return s;
        }
        default:
            return nullptr;
    }
}

bool ends_with(const std::string& s, const std::string& e) {
    return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}

}

ConnectorSpec load_connector(const std::string& path) {
    ConnectorSpec c;
    c.path = path;
    if (ends_with(path, ".json")) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot read " + path);
        c.raw = json::parse(f);
    } else {
        c.raw = from_yaml(YAML::LoadFile(path));
    }
    const json& r = c.raw;
    c.name = r.value("name", "");
    if (c.name.empty()) throw std::runtime_error(path + ": no name");
    if (c.name.size() > 64 || c.name.find("..") != std::string::npos ||
        c.name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-") != std::string::npos)
        throw std::runtime_error(path + ": name may use letters, digits, '.', '_' and '-' only");
    c.codec = r.value("codec", "");
    if (c.codec.empty()) throw std::runtime_error(c.name + ": no codec");
    if (r.contains("game")) {
        const json& g = r["game"];
        c.variant = g.value("variant", c.variant);
        c.players = g.value("players", c.players);
        if (g.contains("blinds") && g["blinds"].size() == 2) c.big = g["blinds"][1].get<int64_t>();
        c.stack = g.value("stack", c.stack);
    }
    if (c.variant != "nlhe") throw std::runtime_error(c.name + ": Texas hold'em (nlhe) only");
    for (const char* k : {"transport", "routes", "flow", "fields"})
        if (!r.contains(k)) throw std::runtime_error(c.name + ": no " + k);
    for (auto& [step, route] : r["flow"].items()) {
        if (!route.is_string() || !r["routes"].contains(route.get<std::string>()))
            throw std::runtime_error(c.name + ": flow." + step + " names no route in routes");
    }
    for (const char* k : {"start", "act"})
        if (!r["flow"].contains(k)) throw std::runtime_error(c.name + ": flow." + k + " is required");
    for (auto& [name, rt] : r["routes"].items()) {
        if (!rt.contains("path") && !rt.contains("message"))
            throw std::runtime_error(c.name + ": routes." + name + " has no path (HTTP) or message (WebSocket)");
        std::string m = rt.value("method", "POST");
        if (m != "GET" && m != "POST" && m != "PUT" && m != "PATCH" && m != "DELETE")
            throw std::runtime_error(c.name + ": routes." + name + ": method " + m);
    }
    std::function<void(const json&)> scan = [&](const json& j) {
        if (j.is_string()) {
            const std::string s = j.get<std::string>();
            for (size_t a = s.find("${env:"); a != std::string::npos; a = s.find("${env:", a + 1)) {
                size_t b = s.find('}', a);
                std::string name = s.substr(a + 6, b == std::string::npos ? std::string::npos : b - a - 6);
                if (name.rfind("KESTREL_", 0) != 0)
                    throw std::runtime_error(c.name + ": ${env:" + name + "}: connectors may read KESTREL_* variables only");
                if (std::find(c.env.begin(), c.env.end(), name) == c.env.end()) c.env.push_back(name);
            }
        } else if (j.is_object() || j.is_array()) {
            for (auto& v : j) scan(v);
        }
    };
    scan(r);
    c.hosts.push_back(r["transport"].value("base_url", r["transport"].value("websocket", "")));
    for (auto& [name, rt] : r["routes"].items()) {
        std::string p = rt.value("path", "");
        if (p.find("://") != std::string::npos) c.hosts.push_back(p);
    }
    return c;
}

std::vector<ConnectorSpec> load_connectors(const std::string& dir, std::vector<std::string>* errors) {
    std::vector<ConnectorSpec> out;
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir, ec)) {
        std::string p = e.path().string();
        if (!(ends_with(p, ".yaml") || ends_with(p, ".yml") || ends_with(p, ".json"))) continue;
        try {
            out.push_back(load_connector(p));
        } catch (const std::exception& x) {
            if (errors) errors->push_back(p + ": " + x.what());
        }
    }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.name < b.name; });
    return out;
}

json render(const json& t, const std::map<std::string, std::string>& vars) {
    if (t.is_string()) {
        std::string s = t.get<std::string>(), out;
        // "${num:name}" alone: sent as a number
        if (s.rfind("${num:", 0) == 0 && s.back() == '}' && s.find('}') == s.size() - 1) {
            auto it = vars.find(s.substr(6, s.size() - 7));
            if (it == vars.end() || it->second.empty()) return "";
            json v = json::parse(it->second, nullptr, false);
            return v.is_number() ? v : json(it->second);
        }
        for (size_t k = 0; k < s.size();) {
            size_t a = s.find("${", k);
            if (a == std::string::npos) {
                out += s.substr(k);
                break;
            }
            size_t b = s.find('}', a);
            if (b == std::string::npos) throw std::runtime_error("unclosed ${ in " + s);
            out += s.substr(k, a - k);
            std::string key = s.substr(a + 2, b - a - 2), val;
            if (key.rfind("env:", 0) == 0) {
                std::string name = key.substr(4);
                if (name.rfind("KESTREL_", 0) != 0)
                    throw std::runtime_error("${env:" + name + "}: connectors may read KESTREL_* variables only");
                const char* e = std::getenv(name.c_str());
                val = e ? e : "";
            } else if (auto it = vars.find(key); it != vars.end()) {
                val = it->second;
            }
            out += val;
            k = b + 1;
        }
        return out;
    }
    if (t.is_object()) {
        json o = json::object();
        for (auto& [k, v] : t.items()) {
            if (v.is_string()) {
                std::string s = v.get<std::string>();
                if (s.size() > 3 && s.rfind("${", 0) == 0 && s.back() == '}' && s.find("${", 2) == std::string::npos) {
                    json r = render(v, vars);
                    if (r.is_string() && r.get<std::string>().empty()) continue;
                }
            }
            o[k] = render(v, vars);
        }
        return o;
    }
    if (t.is_array()) {
        json a = json::array();
        for (auto& v : t) a.push_back(render(v, vars));
        return a;
    }
    return t;
}

namespace {

class HttpTransport : public Transport {
  public:
    explicit HttpTransport(const json& tr) {
        std::string base = tr.at("base_url");
        size_t p = base.find("://");
        if (p == std::string::npos) throw std::runtime_error("transport.base_url needs a scheme: " + base);
        size_t q = base.find('/', p + 3);
        origin_ = q == std::string::npos ? base : base.substr(0, q);
        prefix_ = q == std::string::npos ? "/" : base.substr(q);
        if (prefix_.back() != '/') prefix_ += '/';
        tries_ = tr.contains("retry") ? tr["retry"].value("tries", 6) : 6;
        backoff_ = tr.contains("retry") ? tr["retry"].value("backoff_s", 2.0) : 2.0;
        double rate = tr.value("rate_limit_per_s", 2.0);
        gap_ = std::chrono::milliseconds(rate > 0 ? int(1000.0 / rate) : 0);
        if (tr.contains("headers")) headers_ = tr["headers"];
        content_type_ = tr.value("content_type", "application/json");
        cli_ = std::make_unique<httplib::Client>(origin_);
        cli_->set_connection_timeout(10);
        cli_->set_read_timeout(30);
        cli_->set_follow_location(true);
    }

    json call(const json& route_in, const std::map<std::string, std::string>& vars) override {
        json route = route_in;
        if (route_in.contains("request"))
            for (auto& [k, v] : route_in["request"].items()) route[k] = v;
        std::string path = render(route.at("path"), vars).get<std::string>();
        if (path.empty() || path[0] != '/') path = prefix_ + path;
        std::string method = route.value("method", "POST");
        std::string body = route.contains("body") ? render(route["body"], vars).dump() : "";
        httplib::Headers headers;
        auto add_headers = [&](const json& src) {
            if (!src.is_object()) return;
            json h = render(src, vars);
            for (auto it = h.begin(); it != h.end(); ++it) headers.emplace(it.key(), it.value().get<std::string>());
        };
        add_headers(headers_);
        if (route.contains("headers")) add_headers(route["headers"]);
        httplib::Params query;
        if (route.contains("query"))
        {
            json q = render(route["query"], vars);
            for (auto it = q.begin(); it != q.end(); ++it)
                query.emplace(it.key(), it.value().is_string() ? it.value().get<std::string>() : it.value().dump());
        }
        std::string type = route.value("content_type", content_type_);
        std::string err;
        double wait = backoff_;
        for (int k = 0; k < tries_; ++k) {
            auto now = std::chrono::steady_clock::now();
            if (now < next_) std::this_thread::sleep_until(next_);
            next_ = std::chrono::steady_clock::now() + gap_;
            const std::string url = query.empty() ? path : httplib::append_query_params(path, query);
            httplib::Result res = method == "GET"      ? cli_->Get(url, headers)
                                  : method == "DELETE" ? cli_->Delete(url, headers, body, type)
                                  : method == "PUT"    ? cli_->Put(url, headers, body, type)
                                  : method == "PATCH"  ? cli_->Patch(url, headers, body, type)
                                                       : cli_->Post(url, headers, body, type);
            if (res && res->status < 500) {
                if (res->body.find_first_not_of(" \t\r\n") == std::string::npos) return json::object();  // 204: nothing yet
                try {
                    return json::parse(res->body);
                } catch (const std::exception& e) {
                    err = "not JSON (HTTP " + std::to_string(res->status) + "): " + res->body.substr(0, 200);
                }
            } else {
                err = res ? "HTTP " + std::to_string(res->status) : httplib::to_string(res.error());
            }
            std::this_thread::sleep_for(std::chrono::duration<double>(wait));
            wait *= 2;
        }
        throw TransportError(origin_ + path + " failed: " + err);
    }

  private:
    std::string origin_, prefix_;
    int tries_ = 6;
    double backoff_ = 2.0;
    std::chrono::milliseconds gap_{0};
    json headers_;
    std::string content_type_;
    std::chrono::steady_clock::time_point next_{};
    std::unique_ptr<httplib::Client> cli_;
};

// a WebSocket: messages in a queue as they arrive; reconnects on its own, no faster than reconnect_s
class WsTransport : public Transport {
  public:
    explicit WsTransport(const json& tr) {
        url_ = tr.at("websocket").get<std::string>();
        if (tr.contains("headers")) headers_ = tr["headers"];
        double rate = tr.value("rate_limit_per_s", 10.0);
        gap_ = std::chrono::milliseconds(rate > 0 ? int(1000.0 / rate) : 0);
        ws_.setUrl(url_);
        ix::WebSocketHttpHeaders h;
        json rh = render(headers_, {});
        for (auto it = rh.begin(); it != rh.end(); ++it) h[it.key()] = it.value().get<std::string>();
        ws_.setExtraHeaders(h);
        int back = int(tr.value("reconnect_s", 6.0) * 1000);
        ws_.setMinWaitBetweenReconnectionRetries(back);
        ws_.setMaxWaitBetweenReconnectionRetries(std::max(back, 60000));
        ws_.setPingInterval(30);
        ws_.setOnMessageCallback([this](const ix::WebSocketMessagePtr& m) {
            json ev;
            if (m->type == ix::WebSocketMessageType::Message) {
                ev = json::parse(m->str, nullptr, false);
                if (ev.is_discarded()) return;
            } else if (m->type == ix::WebSocketMessageType::Open) {
                ev = {{"$transport", "open"}};
            } else if (m->type == ix::WebSocketMessageType::Close) {
                ev = {{"$transport", "closed"}, {"code", m->closeInfo.code}, {"reason", m->closeInfo.reason}};
            } else if (m->type == ix::WebSocketMessageType::Error) {
                ev = {{"$transport", "error"}, {"reason", m->errorInfo.reason}};
            } else {
                return;
            }
            std::lock_guard<std::mutex> lk(mu_);
            q_.push_back(std::move(ev));
            if (q_.size() > 10000) q_.pop_front();
            cv_.notify_one();
        });
        ws_.start();
    }
    ~WsTransport() override { ws_.stop(); }

    bool streaming() const override { return true; }
    json call(const json& route, const std::map<std::string, std::string>& vars) override {
        send(route, vars);
        return json::object();
    }
    void send(const json& route, const std::map<std::string, std::string>& vars) override {
        if (!route.contains("message")) throw std::runtime_error("a WebSocket route needs a message");
        auto now = std::chrono::steady_clock::now();
        if (now < next_) std::this_thread::sleep_until(next_);
        next_ = std::chrono::steady_clock::now() + gap_;
        ws_.send(render(route["message"], vars).dump());
    }
    json next(int timeout_ms) override {
        std::unique_lock<std::mutex> lk(mu_);
        if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !q_.empty(); })) return nullptr;
        json e = std::move(q_.front());
        q_.pop_front();
        return e;
    }

  private:
    std::string url_;
    json headers_;
    ix::WebSocket ws_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<json> q_;
    std::chrono::milliseconds gap_{0};
    std::chrono::steady_clock::time_point next_{};
};

}

namespace {
class PatientTransport : public Transport {
  public:
    PatientTransport(std::unique_ptr<Transport> inner, const std::atomic<bool>* stop, double limit_s,
                     std::function<void(const json&)> on_change, double first_s, double max_s)
        : inner_(std::move(inner)), stop_(stop), limit_s_(limit_s), on_change_(std::move(on_change)),
          first_s_(first_s), max_s_(max_s) {}
    json call(const json& route, const std::map<std::string, std::string>& vars) override {
        using clock = std::chrono::steady_clock;
        clock::time_point down_since{};
        double wait = first_s_;
        while (true) {
            try {
                json r = inner_->call(route, vars);
                if (down_since != clock::time_point{} && on_change_)
                    on_change_({{"state", "up"}, {"down_s", std::chrono::duration<double>(clock::now() - down_since).count()}});
                return r;
            } catch (const TransportError& e) {
                const auto now = clock::now();
                if (down_since == clock::time_point{}) {
                    down_since = now;
                    if (on_change_) on_change_({{"state", "down"}, {"reason", e.what()}});
                }
                const double down = std::chrono::duration<double>(now - down_since).count();
                if (down >= limit_s_ || (stop_ && *stop_)) throw;
                for (double t = 0; t < wait && !(stop_ && *stop_); t += 1) std::this_thread::sleep_for(std::chrono::seconds(1));
                if (stop_ && *stop_) throw;
                wait = std::min(wait * 2, max_s_);
            }
        }
    }
    bool streaming() const override { return inner_->streaming(); }
    void send(const json& r, const std::map<std::string, std::string>& v) override { inner_->send(r, v); }
    json next(int timeout_ms) override { return inner_->next(timeout_ms); }

  private:
    std::unique_ptr<Transport> inner_;
    const std::atomic<bool>* stop_;
    double limit_s_;
    std::function<void(const json&)> on_change_;
    double first_s_, max_s_;
};
}  // namespace

std::unique_ptr<Transport> patient(std::unique_ptr<Transport> inner, const std::atomic<bool>* stop, double limit_s,
                                   std::function<void(const json&)> on_change, double first_s, double max_s) {
    return std::make_unique<PatientTransport>(std::move(inner), stop, limit_s, std::move(on_change), first_s, max_s);
}

std::unique_ptr<Transport> make_transport(const ConnectorSpec& spec) {
    const json& tr = spec.raw.at("transport");
    if (tr.contains("base_url")) return std::make_unique<HttpTransport>(tr);
    if (tr.contains("websocket")) return std::make_unique<WsTransport>(tr);
    if (tr.contains("tcp") || tr.contains("mcp"))
        throw std::runtime_error(spec.name + ": tcp and mcp transports are not built yet");
    throw std::runtime_error(spec.name + ": transport needs base_url (http/https)");
}

}
