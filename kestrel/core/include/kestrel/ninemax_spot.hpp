// A spot at any 2-9 handed table, described the way a platform sees it, replayed into Kestrel's table so a 9-max
// network can decide: seats placed by position from the button, chips rescaled to Kestrel's big blind, every bet
// replayed at its own size (the network reads the table state, so any size works), and the answer turned back into
// the platform's chips.
#pragma once
#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "kestrel/net.hpp"

namespace kestrel {

struct NMPlayer {
    std::string id;
    double stack = 0;  // chips at the start of the hand, before blinds and antes
};

struct NMAction {
    std::string player;
    int street = -1;   // 0 pre-flop .. 3 river; checked against the replay when given
    std::string kind;  // fold, check, call, bet, raise, allin
    double to = 0;     // bet, raise, allin: the player's total on this street after the action (platform chips)
};

struct NMSpot {
    std::vector<NMPlayer> players;  // seated players, in seat order (clockwise)
    std::string button, me;
    double sb = 0, bb = 0, ante = 0;  // ante: a big-blind ante (posted by the big blind)
    std::array<std::string, 2> hole;
    std::vector<std::string> board;
    std::vector<NMAction> actions;    // the hand so far, in order
    std::vector<double> payouts;      // prize shares for places 1..; empty: chips (winner takes all)

    static NMSpot from_json(const nlohmann::json& j);
};

struct NMOption {
    std::string action;  // fold, check, call, raise, allin
    double to = 0;       // the player's street total after the move (platform chips)
    double add = 0;      // the chips it puts in now
    double prob = 0;
    int slot = 0;
    std::string label;  // "raise to 6 bb", in big blinds
};

struct NMAdvice {
    std::vector<NMOption> options;
    double value_bb = 0;
    int translated = 0;  // bets and raises replayed at a size the network's menu does not have
    std::string error;   // set when the spot could not be replayed
    nlohmann::json view;  // the table as replayed: seats, positions, stacks and the pot in big blinds, the hand's log
    nlohmann::json to_json() const;
};

NMAdvice ninemax_decide(const Net& net, const NMSpot& spot);

}  // namespace kestrel
