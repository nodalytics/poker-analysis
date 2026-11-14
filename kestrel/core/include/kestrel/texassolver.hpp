// TexasSolver (github.com/bupticybee/TexasSolver, AGPL-3.0) as an external program: Kestrel writes its input script,
// runs console_solver, reads the solved tree back. Run as a separate process, never linked, so Kestrel's licence
// stays its own. The same script as llm-research/scripts/poker_agents.py (_solve).
#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kestrel {

struct TexasSolverConfig {
    std::string bin;        // console_solver
    std::string resources;  // its resources/ (the hand-ranking tables; without them every hand ranks equal)
    int threads = 8, river_iters = 200, turn_iters = 60, timeout_s = 90;
    bool usable(std::string* why = nullptr) const;
};

struct TexasSolverSpot {
    std::vector<std::string> board;  // 4 or 5 cards, "As"
    std::string range_oop, range_ip; // "AsKc:1.000,2c2d:0.250,..."
    double pot = 0, eff = 0;         // at the start of the street, in the solver's units
};

// the solved tree of the street, or null on failure (missing program, timeout, unreadable output)
nlohmann::json texassolver_solve(const TexasSolverConfig& c, const TexasSolverSpot& s);

// follow this street's actions down the tree: (kind "check"|"call"|"fold"|"bet"|"raise", amount added in units);
// sized actions go to the nearest size. nullptr off the tree.
const nlohmann::json* texassolver_walk(const nlohmann::json& tree, const std::vector<std::pair<std::string, double>>& acts);

}  // namespace kestrel
