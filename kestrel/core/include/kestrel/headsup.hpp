#pragma once
#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace kestrel {

int card_from(const std::string& s);
std::string card_str(int c);

struct HUAction {
    int player = 0;
    char kind = 'k';
    int64_t to = 0;
};

struct HUSpot {
    std::vector<std::vector<HUAction>> streets;
    int me = 0;
    std::array<int, 2> hole{};
    std::vector<int> board;
    int64_t min_to = 0, max_to = 0;
    bool can_raise = true, facing = false;
    int64_t pot = 0, my_bet = 0, their_bet = 0, my_stack = 0, their_stack = 0;
};

struct HUOption {
    char kind;
    int64_t to = 0;
    double prob;
    std::string label;
};

struct HUAnswer {
    bool on_tree = false;
    int node = -1, bucket = -1;
    bool floored = false;
    std::vector<HUOption> options;
};

class HUBlueprint {
  public:
    explicit HUBlueprint(const std::string& dir, double purify = 0.05, double floor_from = 0.9);
    HUAnswer decide(const HUSpot& spot, bool deterministic = false, int64_t round_to = 0);
    int bucket(const std::array<int, 2>& hole, const std::vector<int>& board) const;

  private:
    struct Node {
        int kind = 0, player = 0, street = 0;
        int64_t c[2] = {0, 0};
        uint64_t offset = 0;
        std::vector<std::string> labels;
        std::vector<int> children;
    };
    static double frac(double to, double their, double mine, double pot);
    int map_raise(int node, double x, bool deterministic);
    int child(int node, std::initializer_list<const char*> labels) const;
    struct Step {
        int node, child, player, board_n;
    };
    std::vector<double> reach(const std::vector<Step>& history, int player, const std::vector<int>& board) const;
    bool defence_floor(const HUSpot& s, const std::vector<Step>& history, const Node& n, std::vector<double>& st) const;
    double floor_from_;
    std::vector<Node> nodes_;
    std::vector<uint8_t> avg_;
    double purify_;
    std::mt19937_64 rng_{97};
    std::mutex mu_;
};

}
