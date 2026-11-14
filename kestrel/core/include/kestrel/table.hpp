// One no-limit hold'em table: a port of scripts/ninemax_engine.py's Tables for a single table, so the advisor plays
// exactly the game the networks were trained on. Seats: 0 small blind, 1 big blind, 2 UTG ... P-1 button. A stack of
// 0 is an empty seat (the early seats empty first; heads-up the small blind is the button). Action slots: 0 fold,
// 1 check/call, 2..2+S-1 the menu's raise sizes, A-1 all-in. Chips: 1 bb = 100.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace kestrel {

constexpr int64_t BB = 100;
constexpr int64_t SB = 50;
constexpr int64_t STACK = 100 * BB;  // the feature scale (100bb), as in training

struct Rules {
    double ante_bb = 0.0;    // big-blind ante in big blinds (dead money posted by the big blind)
    bool open_limp = true;   // false: first in from UTG to the button may not just call the big blind
};

struct Menu {
    std::vector<float> opens;                // pre-flop open sizes, in chips
    std::vector<std::vector<float>> rer;     // re-raise multiples of the bet, by raise level
    int max_pre = 4;
    std::vector<float> hb, hr, mb, mr;       // post-flop bet / raise pot fractions: heads-up, multiway
    int max_post = 3;
    int S = 0;                               // size slots
    int A = 0;                               // action slots: fold, call, S sizes, all-in

    static Menu named(const std::string& name);  // lean, medium, mtt, wide (ninemax_engine.MENUS)
    void finish();                               // sets S and A
};

class Table {
  public:
    Table(const std::vector<int64_t>& stacks, const Menu& menu, const Rules& rules);

    int P() const { return P_; }
    int actor() const { return actor_; }  // -1 when the hand is over
    bool done() const { return done_; }
    bool showdown() const { return showdown_; }
    int street() const { return street_; }

    // legal action slots for the actor [A], and the raise-to totals of the size slots [S] (0 where unused)
    std::vector<uint8_t> legal(std::vector<float>* targets = nullptr) const;
    void step(int action);  // the actor takes an action slot (throws if illegal)
    // chips won minus chips paid per seat, for a finished hand; ranks: lower is better (showdown ranks)
    std::vector<double> payoffs(const std::vector<int>& ranks) const;

    // state, read by the features
    std::vector<int64_t> stack, avail, contrib;
    std::vector<uint8_t> seated, folded, allin, acted;
    int64_t dead = 0, bet = BB, last_raise = BB;
    int raises = 0, pre_raises = 0, aggressor = -1;
    bool hu = false;
    std::vector<uint8_t> needs() const;
    std::vector<float> targets() const;  // [S], NaN where unused
    const Menu& menu() const { return menu_; }
    const Rules& rules() const { return rules_; }

  private:
    void advance();
    int P_;
    Menu menu_;
    Rules rules_;
    int street_ = 0, to_act_ = 1, actor_ = -1;
    bool done_ = false, showdown_ = false;
};

}  // namespace kestrel
