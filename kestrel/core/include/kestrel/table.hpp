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
    std::vector<double> payoffs(const std::vector<int>& ranks) const;

    std::vector<int64_t> stack, avail, contrib;
    std::vector<uint8_t> seated, folded, allin, acted;
    int64_t dead = 0, bet = BB, last_raise = BB;
    int raises = 0, pre_raises = 0, aggressor = -1;
    bool hu = false;
    std::vector<uint8_t> needs() const;
    std::vector<float> targets() const;
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

}
