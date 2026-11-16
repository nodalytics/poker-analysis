// A spot at any 2-9 handed table, described the way a platform sees it, replayed into Kestrel's table so a 9-max
// network can decide: seats placed by position from the button, chips rescaled to Kestrel's big blind, every bet
// replayed at its own size (the network reads the table state, so any size works), and the answer turned back into
// the platform's chips.
#pragma once
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "kestrel/net.hpp"
#include "kestrel/ninemax_search.hpp"

namespace kestrel {

struct NMPlayer {
    std::string id;
    double stack = 0;  // chips at the start of the hand, before blinds and antes
    double bounty = 0;  // knockout tournaments: the bounty on this player's head, in the same money as payouts
};

struct NMAction {
    std::string player;
    int street = -1;   // 0 pre-flop .. 3 river; checked against the replay when given
    std::string kind;  // fold, check, call, bet, raise, allin
    double to = 0;     // bet, raise, allin: the player's total on this street after the action (platform chips)
    double add = -1;   // or the chips the action put in (platforms that report that instead); -1: not given
    double left = -1;  // or the player's stack after the action; -1: not given
};

struct NMSpot {
    std::vector<NMPlayer> players;  // seated players, in seat order (clockwise)
    std::string button, me;
    double sb = 0, bb = 0, ante = 0;  // ante: a big-blind ante (posted by the big blind)
    bool ante_each = false;           // the ante is posted by every player instead
    std::array<std::string, 2> hole;
    std::vector<std::string> board;
    std::vector<NMAction> actions;    // the hand so far, in order
    std::vector<double> payouts;      // prize shares for places 1..; empty: chips (winner takes all). With bounties,
                                      // payouts and bounties are in the same money and are scaled together

    static NMSpot from_json(const nlohmann::json& j);
};

struct NMOption {
    std::string action;  // fold, check, call, raise, allin
    double to = 0;       // the player's street total after the move (platform chips)
    double add = 0;      // the chips it puts in now
    double prob = 0;
    int slot = 0;
    std::string label;  // "raise to 6 bb", in big blinds
    double q = std::nan("");  // searched: the move's value in big blinds (ICM equity)
};

struct NMAdvice {
    std::vector<NMOption> options;
    double value_bb = 0;
    int translated = 0;  // bets and raises replayed at a size the network's menu does not have
    bool searched = false;  // the advice came from search (a SearchConfig was given and the spot qualifies)
    int samples = 0;
    double ess = 0;      // search: the effective number of deals after weighting by the opponents' actions
    std::string error;   // set when the spot could not be replayed
    nlohmann::json view;  // the table as replayed: seats, positions, stacks and the pot in big blinds, the hand's log
    nlohmann::json to_json() const;
};

// search: when given, search at decision time (ninemax_search.hpp) before the flop and in pots of at most three
// players after it, as tested; elsewhere, and without it, the network's own policy
NMAdvice ninemax_decide(const Net& net, const NMSpot& spot, const SearchConfig* search = nullptr);

}  // namespace kestrel
