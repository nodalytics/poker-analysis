// Kestrel's database: one SQLite file (kestrel.db beside settings.json by default) with every session, every hand and
// every opponent's statistics, so they can be reviewed and queried across sessions and platforms. Each hand keeps the
// key fields in columns (for filters and sums) and its whole record as JSON in `data` (the same line as hands.jsonl,
// which is still written as an export); SQLite's JSON functions read the rest, e.g.
//   SELECT json_extract(data, '$.hole'), result_bb FROM hands WHERE connector = 'slumbot' ORDER BY result_bb LIMIT 10;
// Thread-safe: one connection behind a mutex, write-ahead logging so readers (the sqlite3 shell, a notebook) never
// block the tables.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

struct sqlite3;

namespace kestrel {

class Store {
  public:
    explicit Store(const std::string& path);  // creates the file and its tables if needed; throws if it cannot
    ~Store();
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;

    const std::string& path() const { return path_; }

    // kind: session (a platform), practice; out_dir: the session's folder (unique: importing it again reuses the row)
    int64_t begin_session(const std::string& kind, const std::string& connector, const std::string& model,
                          const std::string& solver, const std::string& out_dir);
    void end_session(int64_t session, const nlohmann::json& summary);
    void add_hand(int64_t session, const nlohmann::json& hand);  // one line of hands.jsonl
    // the whole opponent book, {platform: {username: stats}}: every player's row replaced in one transaction
    void put_opponents(const nlohmann::json& book);

    nlohmann::json stats();  // {path, sessions, hands, opponents}
    nlohmann::json sessions(int limit = 100, int offset = 0);  // newest first, with each one's hands and result
    // hands, newest first; session 0: any; connector "": any
    nlohmann::json hands(int64_t session, const std::string& connector, int limit = 100, int offset = 0);
    // a session folder's hands.jsonl (and summary.json), for history recorded before the database; returns the hands
    // added (0 when that folder is already in the database)
    int import_dir(const std::string& dir);

  private:
    void exec(const char* sql);
    int64_t begin_session_locked(const std::string& kind, const std::string& connector, const std::string& model,
                                 const std::string& solver, const std::string& out_dir);
    void end_session_locked(int64_t session, const nlohmann::json& summary);
    void add_hand_locked(int64_t session, const nlohmann::json& hand);
    std::string path_;
    sqlite3* db_ = nullptr;
    std::mutex mu_;
};

}  // namespace kestrel
