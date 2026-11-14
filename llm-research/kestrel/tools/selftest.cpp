// kestrel_selftest VECTORS MODEL_DIR: replays the Python reference hands (scripts/advisor_vectors.py) through the C++
// table, features and network, and checks every decision (actor, street, legal actions, raise targets, features,
// strength, logits, value), every hand's payoffs and ICM equities, and the ICM cases. Exit 0 only if all match.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "kestrel/icm.hpp"
#include "kestrel/net.hpp"
#include "kestrel/table.hpp"

using namespace kestrel;
using nlohmann::json;

namespace {

struct Check {
    std::string name;
    double tol, worst = 0;
    long n = 0, bad = 0;
    std::string first;
    void cmp(double got, double want, const std::string& where) {
        double e = std::fabs(got - want);
        worst = std::max(worst, e);
        ++n;
        if (!(e <= tol)) {
            if (!bad) first = where + ": got " + std::to_string(got) + ", want " + std::to_string(want);
            ++bad;
        }
    }
    void eq(long got, long want, const std::string& where) { cmp(double(got), double(want), where); }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: kestrel_selftest VECTORS.json MODEL_DIR\n");
        return 2;
    }
    std::ifstream vf(argv[1]);
    json V = json::parse(vf);
    Net net(argv[2]);
    const NetInfo& info = net.info();
    Menu menu = Menu::named(info.menu);

    Check actor{"actor", 0}, street{"street", 0}, legal{"legal", 0}, targets{"targets", 0}, feat{"features", 2e-5},
        str{"strength", 1e-5}, logit{"logits", 2e-3}, value{"value", 2e-3}, pay{"payoffs", 1e-2},
        eq0{"start ICM", 1e-5}, done{"hand over", 0}, icmc{"ICM cases", 1e-5};
    int h = 0;
    for (auto& H : V["hands"]) {
        std::string w0 = "hand " + std::to_string(h++);
        std::vector<int64_t> stacks = H["stacks"].get<std::vector<int64_t>>();
        Table T(stacks, menu, info.rules);
        TourneyCtx ctx;
        if (!H["pay"].is_null()) {
            ctx.pay = H["pay"].get<std::vector<double>>();
            ctx.eq0 = icm(std::vector<double>(stacks.begin(), stacks.end()), ctx.pay);
            auto want = H["eq0"].get<std::vector<double>>();
            for (size_t j = 0; j < want.size(); ++j) eq0.cmp(ctx.eq0[j], want[j], w0 + " seat " + std::to_string(j));
        }
        auto bks = H["buckets"].get<std::vector<std::vector<int>>>();
        int s = 0;
        for (auto& St : H["steps"]) {
            std::string w = w0 + " step " + std::to_string(s++);
            actor.eq(T.actor(), St["actor"].get<int>(), w);
            street.eq(T.street(), St["street"].get<int>(), w);
            if (T.actor() != St["actor"].get<int>()) break;
            std::vector<float> t;
            auto L = T.legal(&t);
            auto wl = St["legal"].get<std::vector<int>>();
            auto wt = St["targets"].get<std::vector<long>>();
            for (size_t k = 0; k < wl.size(); ++k) legal.eq(L[k], wl[k], w + " slot " + std::to_string(k));
            for (size_t k = 0; k < wt.size(); ++k) targets.eq(long(t[k]), wt[k], w + " size " + std::to_string(k));
            auto f = features(T, info, ctx.pay.empty() ? nullptr : &ctx);
            auto wf = St["feat"].get<std::vector<double>>();
            for (size_t k = 0; k < wf.size(); ++k) feat.cmp(f[k], wf[k], w + " feature " + std::to_string(k));
            int i = T.actor();
            std::array<int, 4> bk{bks[i][0], bks[i][1], bks[i][2], bks[i][3]};
            float sv = strength(info, T.street(), bk[T.street()]);
            str.cmp(sv, St["strength"].get<double>(), w);
            std::vector<float> lg;
            float v;
            // the network is checked on the reference features, so a feature error is not counted twice
            std::vector<float> rf(wf.begin(), wf.end());
            net.forward(rf, bk, T.street(), float(St["strength"].get<double>()), lg, v);
            auto wlg = St["logits"].get<std::vector<double>>();
            for (size_t k = 0; k < wlg.size(); ++k) logit.cmp(lg[k], wlg[k], w + " logit " + std::to_string(k));
            value.cmp(v, St["value"].get<double>(), w);
            T.step(St["action"].get<int>());
        }
        done.eq(T.done(), 1, w0);
        done.eq(T.showdown(), H["showdown"].get<bool>(), w0 + " showdown");
        auto u = T.payoffs(H["ranks"].get<std::vector<int>>());
        auto wu = H["payoffs"].get<std::vector<double>>();
        for (size_t j = 0; j < wu.size(); ++j) pay.cmp(u[j], wu[j], w0 + " seat " + std::to_string(j));
    }
    int c = 0;
    for (auto& C : V["icm"]) {
        auto e = icm(C["stacks"].get<std::vector<double>>(), C["pay"].get<std::vector<double>>());
        auto want = C["eq"].get<std::vector<double>>();
        for (size_t j = 0; j < want.size(); ++j) icmc.cmp(e[j], want[j], "case " + std::to_string(c));
        ++c;
    }
    bool ok = true;
    std::printf("kestrel self-test: %zu hands, %d ICM cases, model %s\n", V["hands"].size(), c, argv[2]);
    for (Check* k : {&actor, &street, &legal, &targets, &feat, &str, &logit, &value, &done, &pay, &eq0, &icmc}) {
        std::printf("  %-6s %-10s %6ld checks, worst error %.2e (tolerance %.0e)%s\n", k->bad ? "FAIL" : "ok",
                    k->name.c_str(), k->n, k->worst, k->tol, k->bad ? "" : "");
        if (k->bad) {
            std::printf("         %ld bad; first: %s\n", k->bad, k->first.c_str());
            ok = false;
        }
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
