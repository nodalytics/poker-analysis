// The heads-up blueprint (200bb, blinds 50/100): a port of scripts/blueprint_bot.py over the advisor's data pack
// (blueprint/pack: tree.tsv, avg.u8, flop/turn buckets, river cut points). Against Slumbot, Python's version read
// +12.4 bb/100 [-16, +41] and +30.9 [-10, +72] with the defence floor (5,000 hands each, luck-adjusted).
//
// A spot is the hand so far as streets of actions by blueprint player (0 = button / small blind, 1 = big blind),
// each bet as the bettor's street total in chips. The hand is walked down the abstract betting tree (an off-tree
// size maps to a neighbouring abstract size by the pseudo-harmonic rule); our move is the 8-bit average strategy at
// that node for our card bucket, actions under 5% dropped ("purification").
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace kestrel {

int card_from(const std::string& s);       // "As" -> rank*4 + suit (ranks 2..A = 0..12, suits c d h s)
std::string card_str(int c);

struct HUAction {
    int player = 0;   // blueprint player: 0 button / small blind, 1 big blind
    char kind = 'k';  // f fold, k check, c call, b bet or raise
    int64_t to = 0;   // for b: the bettor's street total, chips
};

struct HUSpot {
    std::vector<std::vector<HUAction>> streets;  // pre-flop first; the last street is the current one
    int me = 0;                                  // our blueprint player
    std::array<int, 2> hole{};
    std::vector<int> board;                      // 0, 3, 4 or 5 cards
    int64_t min_to = 0, max_to = 0;              // the platform's legal raise-to range this street (chips)
    bool can_raise = true, facing = false;
};

struct HUOption {
    char kind;       // f, k (check or call), b
    int64_t to = 0;  // for b: raise-to street total, chips
    double prob;
    std::string label;  // the abstract action it came from
};

struct HUAnswer {
    bool on_tree = false;
    int node = -1, bucket = -1;
    std::vector<HUOption> options;  // empty when off the tree
};

class HUBlueprint {
  public:
    // dir: a data pack (tree.tsv, avg.u8, flop_buckets.u8, turn_buckets.u8, river_cuts.f32, meta.txt)
    explicit HUBlueprint(const std::string& dir, double purify = 0.05);
    // deterministic: map an off-tree size to the nearer abstract size (weight >= 0.5) instead of at random, as the
    // self-test does; round_to: round real bets to this many chips before mapping (the Python replay's units)
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
    std::vector<Node> nodes_;
    std::vector<uint8_t> avg_;
    double purify_;
    std::mt19937_64 rng_{97};
};

}  // namespace kestrel
