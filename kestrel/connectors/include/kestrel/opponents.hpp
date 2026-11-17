// Opponent tracking: each platform's players by username, with the usual statistics, kept across sessions. The
// statistics layer only: Kestrel does not change its play on them (bounded exploitation comes later, once there is
// data: docs/design.md, Status).
#pragma once
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kestrel {

struct SeenAction {
    std::string player;  // username
    int street = 0;      // 0 pre-flop .. 3 river
    std::string kind;    // fold check call bet raise allin
};

struct OpponentStats {
    int64_t hands = 0;
    int64_t vpip = 0, pfr = 0;                  // put money in voluntarily, raised, before the flop
    int64_t threebet = 0, threebet_chance = 0;  // re-raised a single raise before the flop / faced one
    int64_t cbet_folds = 0, cbet_faced = 0;     // folded to the pre-flop raiser's first flop bet / faced one
    int64_t aggressive = 0, passive = 0;        // after the flop: bets and raises / calls
    int64_t saw_flop = 0, showdown = 0;         // saw the flop / showed cards
    std::deque<nlohmann::json> shown;           // the last hands shown: cards and board
    nlohmann::json to_json() const;
};

class OpponentBook {
  public:
    explicit OpponentBook(std::string path);  // a JSON file; created on the first save
    // one hand: everything Kestrel saw (players dealt in by username, the actions in order, shown cards, board)
    void record(const std::string& platform, const std::vector<std::string>& players, const std::vector<SeenAction>& acts,
                const std::map<std::string, std::vector<std::string>>& shown, const std::vector<std::string>& board,
                const std::string& me);
    nlohmann::json platform(const std::string& platform) const;  // {username: stats}
    nlohmann::json player(const std::string& platform, const std::string& name) const;
    nlohmann::json platforms() const;
    // a short line for a seat: "VPIP/PFR/3B · hands", or "" when there is nothing yet
    std::string hud(const std::string& platform, const std::string& name) const;
    // the k players most like this one (cosine similarity of their profiles, profile.hpp), on this platform or (all)
    // every platform, among those with at least min_hands
    nlohmann::json similar(const std::string& platform, const std::string& name, int k = 5, bool all = false,
                           int min_hands = 30) const;
    void flush();  // write the book now (it also saves every 10 hands)
    void set_store(class Store* s);  // also save the book into the database

  private:
    void save_locked();
    std::string path_;
    mutable std::mutex mu_;
    std::map<std::string, std::map<std::string, OpponentStats>> book_;
    int unsaved_ = 0;
    class Store* store_ = nullptr;
};

}  // namespace kestrel
