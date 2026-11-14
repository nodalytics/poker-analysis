#pragma once
#include <string>
#include <vector>

#include "kestrel/headsup.hpp"

namespace kestrel::acpc {

struct Game {
    int64_t big = 100;
    int64_t stack = 20000;
};

struct Token {
    char kind;
    int64_t to = 0;
};

std::vector<std::vector<Token>> tokens(const std::string& action);

bool spot(const std::string& action, int pos, const std::vector<std::string>& hole,
          const std::vector<std::string>& board, const Game& g, HUSpot& out);

std::string encode(const HUOption& o, const HUSpot& s);

}
