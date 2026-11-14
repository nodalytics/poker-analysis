// The "acpc" codec: heads-up action strings in the ACPC style that Slumbot uses (slumbot.com/api), e.g.
// "b200c/kk/kb400c/", one street per "/", tokens k (check), c (call), f (fold), b<N> (bet or raise to N chips, the
// bettor's total on this street). Positions as Slumbot numbers them: 0 the big blind, 1 the small blind (the button).
// Pre-flop the small blind acts first, after the flop the big blind. A port of scripts/slumbot_client.py
// (tokens, legal_range) so the same hands give the same spots.
#pragma once
#include <string>
#include <vector>

#include "kestrel/headsup.hpp"

namespace kestrel::acpc {

struct Game {
    int64_t big = 100;      // the big blind, chips (the small blind is half)
    int64_t stack = 20000;  // every hand starts with this (Slumbot resets each hand)
};

struct Token {
    char kind;       // k c f b
    int64_t to = 0;  // for b
};

std::vector<std::vector<Token>> tokens(const std::string& action);  // throws on a malformed string

// The spot for position `pos` from the action so far, our cards and the board shown. Fills the legal raise range,
// whether a bet is faced, and the actions by blueprint player (0 the small blind / button, 1 the big blind).
// Returns false when it is not `pos`'s turn (the hand is over, or the other player is to act).
bool spot(const std::string& action, int pos, const std::vector<std::string>& hole,
          const std::vector<std::string>& board, const Game& g, HUSpot& out);

// The increment to send for a chosen option: "b<N>", "f", "c" or "k" (a fold when not facing a bet checks, and a
// raise that is not possible becomes a check or call).
std::string encode(const HUOption& o, const HUSpot& s);

}  // namespace kestrel::acpc
