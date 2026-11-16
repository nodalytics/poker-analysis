#include "kestrel/store.hpp"

#include <sqlite3.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace kestrel {

using json = nlohmann::json;

namespace {

std::string now_utc() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

// a prepared statement, finalised on scope exit; parameters bound by position from 1
struct Stmt {
    sqlite3_stmt* s = nullptr;
    Stmt(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &s, nullptr) != SQLITE_OK)
            throw std::runtime_error(std::string("database: ") + sqlite3_errmsg(db));
    }
    ~Stmt() { sqlite3_finalize(s); }
    Stmt& text(int k, const std::string& v) {
        sqlite3_bind_text(s, k, v.c_str(), int(v.size()), SQLITE_TRANSIENT);
        return *this;
    }
    Stmt& num(int k, double v) {
        sqlite3_bind_double(s, k, v);
        return *this;
    }
    Stmt& i64(int k, int64_t v) {
        sqlite3_bind_int64(s, k, v);
        return *this;
    }
    Stmt& null(int k) {
        sqlite3_bind_null(s, k);
        return *this;
    }
    bool step() {
        int rc = sqlite3_step(s);
        if (rc == SQLITE_ROW) return true;
        if (rc == SQLITE_DONE) return false;
        throw std::runtime_error(std::string("database: ") + sqlite3_errmsg(sqlite3_db_handle(s)));
    }
    std::string col_text(int k) const {
        auto p = sqlite3_column_text(s, k);
        return p ? reinterpret_cast<const char*>(p) : "";
    }
    json col(int k) const {  // a column as JSON: null, a number or text
        switch (sqlite3_column_type(s, k)) {
            case SQLITE_NULL: return nullptr;
            case SQLITE_INTEGER: return sqlite3_column_int64(s, k);
            case SQLITE_FLOAT: return sqlite3_column_double(s, k);
            default: return col_text(k);
        }
    }
};

std::string str_of(const json& j, const char* k) {
    if (!j.contains(k) || j[k].is_null()) return "";
    return j[k].is_string() ? j[k].get<std::string>() : j[k].dump();
}

}  // namespace

Store::Store(const std::string& path) : path_(path) {
    std::error_code ec;
    if (auto dir = std::filesystem::path(path).parent_path(); !dir.empty()) std::filesystem::create_directories(dir, ec);
    if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
        std::string why = db_ ? sqlite3_errmsg(db_) : "out of memory";
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("database " + path + ": " + why);
    }
    sqlite3_busy_timeout(db_, 5000);
    exec("PRAGMA journal_mode = WAL");
    exec("PRAGMA foreign_keys = ON");
    exec("PRAGMA synchronous = NORMAL");
    exec(R"(
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,          -- session (a platform) or practice
  connector TEXT NOT NULL,
  model TEXT, solver TEXT,
  out_dir TEXT UNIQUE,         -- the session's folder of files
  started TEXT NOT NULL, ended TEXT,
  summary TEXT                 -- summary.json
);
CREATE TABLE IF NOT EXISTS hands (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,        -- 1, 2, ... within the session
  recorded TEXT NOT NULL,
  connector TEXT, hand TEXT,   -- the platform's hand id when it gives one
  model TEXT, solver TEXT,
  result_bb REAL,              -- Kestrel's result in big blinds (null when the record has none)
  data TEXT NOT NULL           -- the whole record, JSON
);
CREATE INDEX IF NOT EXISTS hands_by_session ON hands(session_id, seq);
CREATE INDEX IF NOT EXISTS hands_by_connector ON hands(connector, id);
CREATE TABLE IF NOT EXISTS opponents (
  platform TEXT NOT NULL, player TEXT NOT NULL,
  hands INTEGER, stats TEXT NOT NULL, updated TEXT NOT NULL,
  PRIMARY KEY (platform, player)
);
INSERT OR IGNORE INTO meta VALUES ('schema', '1');
)");
}

Store::~Store() { sqlite3_close(db_); }

void Store::exec(const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string why = err ? err : "error";
        sqlite3_free(err);
        throw std::runtime_error("database: " + why);
    }
}

int64_t Store::begin_session(const std::string& kind, const std::string& connector, const std::string& model,
                             const std::string& solver, const std::string& out_dir) {
    std::lock_guard<std::mutex> lk(mu_);
    return begin_session_locked(kind, connector, model, solver, out_dir);
}

int64_t Store::begin_session_locked(const std::string& kind, const std::string& connector, const std::string& model,
                                    const std::string& solver, const std::string& out_dir) {
    if (!out_dir.empty()) {  // the same folder again (a resumed session): its row
        Stmt q(db_, "SELECT id FROM sessions WHERE out_dir = ?");
        if (q.text(1, out_dir).step()) return sqlite3_column_int64(q.s, 0);
    }
    Stmt s(db_, "INSERT INTO sessions (kind, connector, model, solver, out_dir, started) VALUES (?, ?, ?, ?, ?, ?)");
    s.text(1, kind).text(2, connector).text(3, model).text(4, solver).text(6, now_utc());
    if (out_dir.empty()) s.null(5);
    else s.text(5, out_dir);
    s.step();
    return sqlite3_last_insert_rowid(db_);
}

void Store::end_session(int64_t session, const json& summary) {
    std::lock_guard<std::mutex> lk(mu_);
    end_session_locked(session, summary);
}

void Store::end_session_locked(int64_t session, const json& summary) {
    Stmt s(db_, "UPDATE sessions SET ended = ?, summary = ? WHERE id = ?");
    s.text(1, now_utc()).text(2, summary.dump()).i64(3, session).step();
}

void Store::add_hand(int64_t session, const json& h) {
    std::lock_guard<std::mutex> lk(mu_);
    add_hand_locked(session, h);
}

void Store::add_hand_locked(int64_t session, const json& h) {
    Stmt n(db_, "SELECT COALESCE(MAX(seq), 0) + 1 FROM hands WHERE session_id = ?");
    n.i64(1, session).step();
    const int64_t seq = sqlite3_column_int64(n.s, 0);
    Stmt s(db_, "INSERT INTO hands (session_id, seq, recorded, connector, hand, model, solver, result_bb, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
    s.i64(1, session).i64(2, seq).text(3, now_utc()).text(4, str_of(h, "connector"));
    std::string hand = str_of(h, "hand_id");
    if (hand.empty()) hand = str_of(h, "hand");
    s.text(5, hand).text(6, str_of(h, "model")).text(7, str_of(h, "solver"));
    if (h.contains("result_bb") && h["result_bb"].is_number()) s.num(8, h["result_bb"].get<double>());
    else s.null(8);
    s.text(9, h.dump()).step();
}

void Store::put_opponents(const json& book) {
    std::lock_guard<std::mutex> lk(mu_);
    exec("BEGIN");
    try {
        const std::string t = now_utc();
        Stmt s(db_, "INSERT INTO opponents (platform, player, hands, stats, updated) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (platform, player) DO UPDATE SET hands = excluded.hands, stats = excluded.stats, "
                    "updated = excluded.updated");
        for (auto& [plat, players] : book.items())
            for (auto& [name, st] : players.items()) {
                sqlite3_reset(s.s);
                s.text(1, plat).text(2, name).text(4, st.dump()).text(5, t);
                if (st.contains("hands") && st["hands"].is_number()) s.i64(3, st["hands"].get<int64_t>());
                else s.null(3);
                s.step();
            }
        exec("COMMIT");
    } catch (...) {
        exec("ROLLBACK");
        throw;
    }
}

json Store::stats() {
    std::lock_guard<std::mutex> lk(mu_);
    json j = {{"path", path_}};
    for (const char* t : {"sessions", "hands", "opponents"}) {
        Stmt s(db_, (std::string("SELECT COUNT(*) FROM ") + t).c_str());
        s.step();
        j[t] = sqlite3_column_int64(s.s, 0);
    }
    return j;
}

json Store::sessions(int limit, int offset) {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt s(db_, "SELECT s.id, s.kind, s.connector, s.model, s.solver, s.out_dir, s.started, s.ended, "
                "COUNT(h.id), SUM(h.result_bb) FROM sessions s LEFT JOIN hands h ON h.session_id = s.id "
                "GROUP BY s.id ORDER BY s.id DESC LIMIT ? OFFSET ?");
    s.i64(1, limit).i64(2, offset);
    json out = json::array();
    while (s.step()) {
        const int64_t n = sqlite3_column_int64(s.s, 8);
        json total = s.col(9);
        out.push_back({{"id", s.col(0)}, {"kind", s.col(1)}, {"connector", s.col(2)}, {"model", s.col(3)},
                       {"solver", s.col(4)}, {"out_dir", s.col(5)}, {"started", s.col(6)}, {"ended", s.col(7)},
                       {"hands", n}, {"result_bb", total},
                       {"bb100", n && total.is_number() ? json(total.get<double>() / double(n) * 100) : json(nullptr)}});
    }
    return out;
}

json Store::hands(int64_t session, const std::string& connector, int limit, int offset) {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt s(db_, "SELECT id, session_id, seq, recorded, data FROM hands WHERE (?1 = 0 OR session_id = ?1) "
                "AND (?2 = '' OR connector = ?2) ORDER BY id DESC LIMIT ?3 OFFSET ?4");
    s.i64(1, session).text(2, connector).i64(3, limit).i64(4, offset);
    json out = json::array();
    while (s.step()) {
        json h = json::parse(s.col_text(4), nullptr, false);
        out.push_back({{"id", s.col(0)}, {"session", s.col(1)}, {"seq", s.col(2)}, {"recorded", s.col(3)},
                       {"hand", h.is_discarded() ? json(nullptr) : h}});
    }
    return out;
}

int Store::import_dir(const std::string& dir) {
    namespace fs = std::filesystem;
    const fs::path d(dir);
    std::ifstream f(d / "hands.jsonl");
    if (!f) return 0;
    const std::string key = fs::absolute(d).lexically_normal().string();
    std::lock_guard<std::mutex> lk(mu_);  // the whole import, so no other write lands inside its transaction
    {
        Stmt q(db_, "SELECT id FROM sessions WHERE out_dir = ?");
        if (q.text(1, key).step()) return 0;  // already in the database
    }
    std::vector<json> lines;
    for (std::string line; std::getline(f, line);) {
        json h = json::parse(line, nullptr, false);
        if (h.is_object()) lines.push_back(std::move(h));
    }
    const bool practice = d.filename().string().find("practice") != std::string::npos;
    std::string connector = practice ? "practice" : "", model, solver;
    for (auto& h : lines) {
        if (connector.empty()) connector = str_of(h, "connector");
        if (model.empty()) model = str_of(h, "model");
        if (solver.empty()) solver = str_of(h, "solver");
    }
    if (connector.empty()) connector = d.filename().string();
    exec("BEGIN");
    try {
        const int64_t sid = begin_session_locked(practice ? "practice" : "session", connector, model, solver, key);
        for (auto& h : lines) {
            if (practice && !h.contains("connector")) h["connector"] = "practice";
            add_hand_locked(sid, h);
        }
        std::ifstream sf(d / "summary.json");
        json sum = json::parse(sf, nullptr, false);
        if (sf && !sum.is_discarded()) end_session_locked(sid, sum);
        exec("COMMIT");
    } catch (...) {
        exec("ROLLBACK");
        throw;
    }
    return int(lines.size());
}

}  // namespace kestrel
