#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace kestrel {

constexpr int64_t BB = 100;
constexpr int64_t SB = 50;
constexpr int64_t STACK = 100 * BB;

struct Rules {
    double ante_bb = 0.0;
    bool open_limp = true;
    bool short_stacks = false;  // platform replays: stacks under 2bb play, and a blind that cannot be covered is all-in
    int64_t extra_dead = 0;     // dead money already paid from the stacks (antes posted by every player)
    // first in before the flop with at least 6 players still to act behind (UTG, UTG+1 at a full table) and at least
    // this many big blinds: no all-in, open small instead (the partner's rule, 2026-11-19; 0: off)
    double early_shove_bb = 0.0;
};

struct Menu {
    std::vector<float> opens;
    std::vector<std::vector<float>> rer;
    int max_pre = 4;
    std::vector<float> hb, hr, mb, mr;
    int max_post = 3;
    int S = 0;
    int A = 0;

    static Menu named(const std::string& name);
    void finish();
};

class Table {
  public:
    Table(const std::vector<int64_t>& stacks, const Menu& menu, const Rules& rules);

    int P() const { return P_; }
    int actor() const { return actor_; }
    bool done() const { return done_; }
    bool showdown() const { return showdown_; }
    int street() const { return street_; }

    std::vector<uint8_t> legal(std::vector<float>* targets = nullptr) const;
    void step(int action);
    void step_to(int64_t total);  // a bet or raise of any size, to this hand total (platform replays)
    std::vector<double> payoffs(const std::vector<int>& ranks) const;

    std::vector<int64_t> stack, avail, contrib;
    std::vector<uint8_t> seated, folded, allin, acted;
    int64_t dead = 0, bet = BB, last_raise = BB;
    int raises = 0, pre_raises = 0, aggressor = -1;
    bool hu = false;
    std::vector<uint8_t> needs() const;
    std::vector<float> targets() const;
    const Menu& menu() const { return menu_; }
    // the actor's moves and sizes come from the menu set here: a table of networks on different menus sets each
    // actor's own before its decision (the chips and betting do not depend on it)
    void set_menu(const Menu& m) { menu_ = m; }
    const Rules& rules() const { return rules_; }
    void set_rules(const Rules& r) { rules_ = r; }

  private:
    void advance();
    void commit(int i, bool fold, bool call, int64_t total);
    int P_;
    Menu menu_;
    Rules rules_;
    int street_ = 0, to_act_ = 1, actor_ = -1;
    bool done_ = false, showdown_ = false;
};

}
