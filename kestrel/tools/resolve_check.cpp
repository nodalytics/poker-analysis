// kestrel_resolve_check --vectors FILE --run DIR [--abs DIR] [--iters 500]: Kestrel's safe river re-solve against
// the Python bot's (libresolve) on the same logged river decisions (llm-research/scripts/kestrel_hu_vectors.py
// --resolve). Compares the re-solved strategy at our hand, action by action. Tolerance 5e-3: Kestrel normalises the
// blueprint in double precision and libresolve in single, and 500 CFR+ iterations carry that into the third decimal.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "kestrel/codec_acpc.hpp"
#include "kestrel/headsup.hpp"

using namespace kestrel;

int main(int argc, char** argv) {
    std::string vectors, run, abs;
    int iters = 500;
    for (int k = 1; k + 1 < argc; k += 2) {
        std::string a = argv[k], v = argv[k + 1];
        if (a == "--vectors") vectors = v;
        else if (a == "--run") run = v;
        else if (a == "--abs") abs = v;
        else if (a == "--iters") iters = std::stoi(v);
    }
    if (vectors.empty() || run.empty()) {
        std::fprintf(stderr, "usage: kestrel_resolve_check --vectors FILE --run DIR [--abs DIR] [--iters N]\n");
        return 2;
    }
    std::ifstream f(vectors);
    auto V = nlohmann::json::parse(f);
    HUBlueprint bp(run, 0.05, 0.0, abs);
    bp.set_resolve(iters, 0.02);
    acpc::Game g;
    int cases = 0, resolved = 0, bad = 0, skipped = 0;
    double worst = 0;
    for (auto& C : V["cases"]) {
        if (C["resolve"].is_null()) continue;
        HUSpot sp;
        if (!acpc::spot(C["action"], C["pos"], C["hole"].get<std::vector<std::string>>(),
                        C["board"].get<std::vector<std::string>>(), g, sp)) {
            ++skipped;
            continue;
        }
        ++cases;
        auto ans = bp.decide(sp, true, 50);
        if (!ans.resolved) {
            ++bad;
            std::printf("not re-solved: %s\n", C["action"].get<std::string>().c_str());
            continue;
        }
        ++resolved;
        auto want = C["resolve"].get<std::vector<double>>();
        double e = 0;
        for (size_t a = 0; a < want.size() && a < ans.strategy.size(); ++a)
            e = std::max(e, std::fabs(ans.strategy[a] - want[a]));
        worst = std::max(worst, e);
        if (e > 5e-3) {
            ++bad;
            std::printf("differs by %.4f: %s\n", e, C["action"].get<std::string>().c_str());
        }
        if (cases % 20 == 0) std::printf("%d cases, worst %.2e\n", cases, worst), std::fflush(stdout);
    }
    std::printf("%d river decisions, %d re-solved, %d differ by more than 5e-3, worst difference %.2e%s\n", cases,
                resolved, bad, worst, skipped ? " (some skipped)" : "");
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
