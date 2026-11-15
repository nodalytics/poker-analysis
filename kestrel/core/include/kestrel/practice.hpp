// A practice tournament: a final table played out by a 9-max tournament network in every seat, to watch it decide.
// Stacks carry over, the button moves, blinds rise every few hands, busted players finish in places with
// final-table payouts, and every hand is played with the ICM features the network was trained on.
#pragma once
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "kestrel/net.hpp"

namespace kestrel {

// position names by Kestrel seat for n players (0 small blind, 1 big blind, the last seat the button)
std::vector<std::string> position_names(int n);
// chips as big blinds, "2.5" or "3"
std::string bb_str(double chips);

struct PracticeConfig {
    int players = 9;                 // 2..9 at the start
    double avg_bb = 20;              // starting average stack, big blinds
    double spread = 0.5;             // stacks vary log-uniformly within avg x e^(+-spread)
    int hands_per_level = 15;        // blinds rise by level_factor every this many hands
    double level_factor = 1.25;
    std::vector<double> payouts{0.30, 0.20, 0.145, 0.105, 0.08, 0.065, 0.05, 0.035, 0.02};
    uint64_t seed = 1;
    bool human = false;  // the first player is the user ("You"); the model plays the others, and hides its cards
    double bounty = 0;   // progressive knockouts: this share of the prize money is bounties (a PKO model only)
};

class PracticeTournament {
  public:
    PracticeTournament(const Net& net, const PracticeConfig& c, std::string label);
    bool over() const;
    // plays one hand; emit sees {"type": "practice_decision" | "practice_hand", ...}. With a human seat, ask gets
    // {"type": "practice_your_turn", ...} and returns the chosen action slot (-1: the session is stopping)
    void play_hand(const std::function<void(const nlohmann::json&)>& emit,
                   const std::function<int(const nlohmann::json&)>& ask = nullptr);
    nlohmann::json standings() const;
    int hands() const { return hands_; }

  private:
    struct Player {
        std::string name;
        int64_t chips = 0;
        int place = 0;      // 0 while playing
        double bounty = 0;  // on this player's head, as a share of the prize money
        double won = 0;     // bounty cash collected
    };
    const Net& net_;
    PracticeConfig cfg_;
    std::string label_;
    std::vector<Player> players_;
    std::mt19937_64 rng_;
    int button_ = 0, hands_ = 0, level_ = 1;
};

}  // namespace kestrel
