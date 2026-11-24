#include "kestrel/texassolver.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>

namespace kestrel {

namespace fs = std::filesystem;

namespace {

std::string quote(const std::string& s) {
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    std::string q = "'";
    for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
#endif
}

}  // namespace

bool TexasSolverConfig::usable(std::string* why) const {
    auto fail = [&](const std::string& w) {
        if (why) *why = w;
        return false;
    };
    if (bin.empty() || !fs::exists(bin)) return fail("console_solver not found: " + bin);
    if (!fs::exists(fs::path(resources) / "compairer" / "card5_dic_sorted.txt"))
        return fail("TexasSolver's hand-ranking tables missing under " + resources);
    return true;
}

nlohmann::json texassolver_solve(const TexasSolverConfig& c, const TexasSolverSpot& s) {
    if (!c.usable() || (s.board.size() != 4 && s.board.size() != 5)) return nullptr;
    const bool river = s.board.size() == 5;
    const char* street = river ? "river" : "turn";
    std::mt19937_64 rng(std::random_device{}());
    fs::path dir = fs::temp_directory_path() / ("kestrel-ts-" + std::to_string(rng()));
    fs::create_directories(dir);
    std::string board;
    for (size_t k = 0; k < s.board.size(); ++k) board += (k ? "," : "") + s.board[k];
    std::ofstream in(dir / "in.txt");
    in << "set_pot " << s.pot << "\nset_effective_stack " << s.eff << "\nset_board " << board
       << "\nset_range_oop " << s.range_oop << "\nset_range_ip " << s.range_ip << "\n";
    for (const char* p : {"oop", "ip"}) {
        in << "set_bet_sizes " << p << "," << street << ",bet,66\n";
        in << "set_bet_sizes " << p << "," << street << ",raise,100\n";
        if (s.eff <= 4 * s.pot) in << "set_bet_sizes " << p << "," << street << ",allin\n";
    }
    in << "set_allin_threshold 0.67\nbuild_tree\nset_thread_num " << c.threads << "\nset_accuracy 1.0\n"
       << "set_max_iteration " << (river ? c.river_iters : c.turn_iters) << "\nset_print_interval 20\n"
       << "set_use_isomorphism 1\nstart_solve\nset_dump_rounds 1\ndump_result " << (dir / "out.json").string() << "\n";
    in.close();
    // run inside the temporary folder: TexasSolver writes its progress log (tmp_log.txt) to the working directory
    std::string cmd;
#ifndef _WIN32
    // a time limit: coreutils' timeout (gtimeout from Homebrew), else perl's alarm, which macOS has and timeout not
    // (before 0.4.1 the solver never ran on macOS: "timeout" was not found, and every call fell back to the blueprint)
    const std::string secs = std::to_string(c.timeout_s);
    cmd += "T=$(command -v timeout || command -v gtimeout); if [ -n \"$T\" ]; then set -- \"$T\" " + secs +
           "; else set -- perl -e 'alarm shift; exec @ARGV or exit 127' " + secs + "; fi; ";
#endif
    cmd += "cd " + quote(dir.string()) + " && ";
#ifndef _WIN32
    cmd += "\"$@\" nice -n 10 ";
#endif
    cmd += quote(c.bin) + " --input_file " + quote((dir / "in.txt").string()) + " --resource_dir " +
           quote(c.resources);
#ifdef _WIN32
    cmd += " > NUL 2>&1";
#else
    cmd += " > /dev/null 2>&1";
#endif
    nlohmann::json tree;
    if (std::system(cmd.c_str()) == 0) {
        std::ifstream out(dir / "out.json");
        tree = nlohmann::json::parse(out, nullptr, false);
        if (tree.is_discarded()) tree = nullptr;
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    return tree;
}

const nlohmann::json* texassolver_walk(const nlohmann::json& tree,
                                       const std::vector<std::pair<std::string, double>>& acts) {
    const nlohmann::json* node = &tree;
    for (auto& [kind, amount] : acts) {
        if (!node->is_object() || node->value("node_type", "") != "action_node") return nullptr;
        const auto& names = (*node)["actions"];
        std::string want;
        double best = 1e300;
        for (const auto& a : names) {
            std::string n = a.get<std::string>();
            std::string head = n.substr(0, n.find(' '));
            if (kind == "check" || kind == "call" || kind == "fold") {
                std::string up = kind;
                for (auto& ch : up) ch = char(std::toupper(ch));
                if (head == up) {
                    want = n;
                    break;
                }
            } else if (head == "BET" || head == "RAISE") {
                double d = std::fabs(std::stod(n.substr(n.find(' ') + 1)) - amount);
                if (d < best) best = d, want = n;
            }
        }
        if (want.empty() || !node->contains("childrens") || !(*node)["childrens"].contains(want)) return nullptr;
        node = &(*node)["childrens"][want];
    }
    return node;
}

}  // namespace kestrel
