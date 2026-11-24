// Heads-up engine checks that the parity vectors cannot see (they run deterministic mappings and equal stacks):
//  1. an off-tree bet keeps one mapping for the whole hand with deterministic=false (production): replaying the same
//     hand gives the same tree node, strategy and options every time, while different hands still draw differently;
//  2. the TexasSolver spot's effective stack is the smaller of the two players' (stack behind + street bet), as the
//     Python bot's resolve() (a stand-in console_solver records the input script Kestrel writes; POSIX only);
//  3. the defence floor on re-solved river spots can be switched off (set_resolve_floor), and is on by default.
// Usage: kestrel_hu_check --pack PACK_DIR [--resolve-iters N]   (100 by default; ctest runs 30, about 17 s)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "kestrel/codec_acpc.hpp"
#include "kestrel/headsup.hpp"

using namespace kestrel;
namespace fs = std::filesystem;

namespace {

int failures = 0;
const auto t0 = std::chrono::steady_clock::now();
double secs() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

void check(bool ok, const std::string& what) {
    std::printf("  %-4s %s  [%.1fs]\n", ok ? "ok" : "FAIL", what.c_str(), secs());
    std::fflush(stdout);
    failures += !ok;
}

std::string describe(const HUAnswer& a) {
    std::ostringstream o;
    o << "node " << a.node << " bucket " << a.bucket << " strategy";
    for (double v : a.strategy) o << " " << v;
    o << " options";
    for (const auto& x : a.options) o << " " << x.kind << ":" << x.to << ":" << x.prob;
    return o.str();
}

const std::vector<std::string> kDeck = [] {
    std::vector<std::string> d;
    for (char r : std::string("23456789TJQKA"))
        for (char s : std::string("cdhs")) d.push_back(std::string(1, r) + s);
    return d;
}();

}  // namespace

int main(int argc, char** argv) {
    std::string pack;
    int resolve_iters = 100;
    for (int k = 1; k + 1 < argc; k += 2) {
        std::string a = argv[k];
        if (a == "--pack") pack = argv[k + 1];
        else if (a == "--resolve-iters") resolve_iters = std::atoi(argv[k + 1]);
    }
    if (pack.empty()) {
        std::fprintf(stderr, "usage: kestrel_hu_check --pack PACK_DIR [--resolve-iters N]\n");
        return 2;
    }
    acpc::Game g;
    const std::vector<std::string> board{"7h", "4s", "4h", "Td"};

    // 1. off-tree bet mappings, replayed with deterministic=false
    std::printf("off-tree mappings, deterministic=false:\n");
    {
        HUBlueprint bp(pack, 0.05, 0.9);
        int unstable = 0, spots = 0, varied = 0;
        std::string first_bad;
        for (int x = 230; x <= 1900; x += 30) {  // the opponent's flop bet, off the tree's sizes for most x
            const std::string action = "b200c/kb" + std::to_string(x) + "c/";
            HUSpot sp;
            if (!acpc::spot(action, 0, {"Kd", "8h"}, board, g, sp)) continue;
            ++spots;
            const std::string want = describe(bp.decide(sp, false, 50));
            bool same = true;
            for (int r = 0; r < 25 && same; ++r) {
                const std::string got = describe(bp.decide(sp, false, 50));
                if (got != want) {
                    same = false;
                    if (first_bad.empty()) first_bad = action + ": " + want + "  vs  " + got;
                }
            }
            unstable += !same;
            // other hands with the same betting: the mapping is still drawn, not fixed (some x land on two nodes)
            std::set<int> nodes;
            for (int h = 0; h < 40; ++h) {
                HUSpot o;
                const std::string a = kDeck[size_t(h)], b = kDeck[size_t(51 - h)];
                if (std::find(board.begin(), board.end(), a) != board.end() ||
                    std::find(board.begin(), board.end(), b) != board.end())
                    continue;
                if (acpc::spot(action, 0, {a, b}, board, g, o)) nodes.insert(bp.decide(o, false, 50).node);
            }
            varied += nodes.size() > 1;
        }
        check(unstable == 0, "the same hand replayed 26 times decides from the same node with the same options (" +
                                 std::to_string(spots - unstable) + " of " + std::to_string(spots) + " bet sizes stable)" +
                                 (first_bad.empty() ? "" : "\n         first: " + first_bad));
        check(varied > 0, "across hands the mapping is still randomised (" + std::to_string(varied) +
                              " bet sizes reach two or more nodes)");
    }

#ifndef _WIN32
    // 2. the TexasSolver effective stack: a stand-in solver copies Kestrel's input script and fails
    std::printf("TexasSolver effective stack:\n");
    {
        fs::path dir = fs::temp_directory_path() / ("kestrel-hu-check-" + std::to_string(::getpid()));
        fs::create_directories(dir / "res" / "compairer");
        std::ofstream(dir / "res" / "compairer" / "card5_dic_sorted.txt") << "\n";
        const fs::path bin = dir / "console_solver", capture = dir / "captured.txt";
        {
            std::ofstream s(bin);
            s << "#!/bin/sh\ncp \"$2\" '" << capture.string() << "'\nexit 1\n";
        }
        fs::permissions(bin, fs::perms::owner_all);
        HUBlueprint bp(pack, 0.05, 0.9);
        TexasSolverConfig c;
        c.bin = bin.string(), c.resources = (dir / "res").string(), c.timeout_s = 10;
        bp.set_texassolver(c);
        HUSpot sp;
        acpc::spot("b200c/kk/", 0, {"Kd", "8h"}, board, g, sp);
        // unequal stacks (a cash table after earlier hands): we have 19,800 behind, the opponent 4,800
        sp.their_stack = 4800;
        bp.decide(sp, false, 50);
        std::ifstream in(capture);
        std::string line, eff;
        while (std::getline(in, line))
            if (line.rfind("set_effective_stack ", 0) == 0) eff = line.substr(20);
        check(eff == "96", "set_effective_stack is min(19800, 4800) / 50 = 96 (got '" + eff + "')");
        // equal stacks: unchanged
        sp.their_stack = sp.my_stack;
        fs::remove(capture);
        bp.decide(sp, true, 50);
        std::ifstream in2(capture);
        eff.clear();
        while (std::getline(in2, line))
            if (line.rfind("set_effective_stack ", 0) == 0) eff = line.substr(20);
        check(eff == "396", "equal stacks: set_effective_stack 19800 / 50 = 396 (got '" + eff + "')");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
#endif

    // 3. the defence floor on re-solved river spots: on by default, off with set_resolve_floor(false)
    std::printf("defence floor on re-solved spots (%d iterations):\n", resolve_iters);
    {
        std::ifstream vf(std::string(KESTREL_TESTDATA) + "/vectors_headsup_floor.json");
        nlohmann::json V = nlohmann::json::parse(vf);
        // one engine: a spot is re-solved once (cached by its river history) and decided with the floor on, then off
        HUBlueprint bp(pack, 0.05, 0.9);
        bp.set_resolve(resolve_iters);
        check(bp.resolve_floor(), "the floor is on in a new engine");
        int resolved = 0, floored_on = 0, floored_off = 0, tried = 0;
        for (auto& C : V["cases"]) {
            if (C["board"].size() != 5 || !C["legal"]["facing"].get<bool>()) continue;
            HUSpot sp;
            if (!acpc::spot(C["action"], C["pos"], C["hole"].get<std::vector<std::string>>(),
                            C["board"].get<std::vector<std::string>>(), g, sp))
                continue;
            bp.set_resolve_floor(true);
            auto a = bp.decide(sp, true, 50);
            bp.set_resolve_floor(false);
            auto b = bp.decide(sp, true, 50);
            if (!a.resolved) continue;
            ++resolved;
            floored_on += a.floored, floored_off += b.resolved && b.floored;
            if (++tried >= 40) break;
        }
        check(floored_on > 0, "floor on (default): " + std::to_string(floored_on) + " of " + std::to_string(resolved) +
                                  " re-solved spots floored");
        check(floored_off == 0, "floor off: " + std::to_string(floored_off) + " re-solved spots floored");
    }
    std::printf(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
