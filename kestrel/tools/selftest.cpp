// kestrel_selftest --vectors DIR --model MODEL_DIR --pack PACK_DIR: checks Kestrel against the Python reference.
//  9-max (vectors_nm17.json, llm-research/scripts/advisor_vectors.py): every decision of the reference hands through
//    the C++ table, features and network (actor, street, legal actions, raise targets, features, strength, logits,
//    value), every hand's payoffs and ICM equities, and the ICM cases.
//  heads-up (vectors_headsup.json, llm-research/scripts/kestrel_hu_vectors.py): logged Slumbot decisions through the
//    acpc codec and the blueprint (node, bucket, strategy, raise sizes).
// Exit 0 only if all match.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "kestrel/codec_acpc.hpp"
#include "kestrel/headsup.hpp"
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
    std::string vdir, model, pack;
    for (int k = 1; k + 1 < argc; k += 2) {
        std::string a = argv[k];
        (a == "--vectors" ? vdir : a == "--model" ? model : a == "--pack" ? pack : vdir) = argv[k + 1];
    }
    if (vdir.empty() || model.empty() || pack.empty()) {
        std::fprintf(stderr, "usage: kestrel_selftest --vectors DIR --model MODEL_DIR --pack PACK_DIR\n");
        return 2;
    }
    std::ifstream vf(vdir + "/vectors_nm17.json");
    json V = json::parse(vf);
    Net net(model);
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
    // ---- heads-up: the blueprint through the Slumbot codec ----
    Check hturn{"hu: our turn", 0}, hnode{"hu: node", 0}, hbk{"hu: bucket", 0}, hn{"hu: options", 0},
        hkind{"hu: kinds", 0}, hprob{"hu: probs", 1e-5}, hto{"hu: raise to", 50};
    std::ifstream hf(vdir + "/vectors_headsup.json");
    json HV = json::parse(hf);
    HUBlueprint bp(pack);
    acpc::Game g;
    int hc = 0;
    for (auto& C : HV["cases"]) {
        std::string w = "case " + std::to_string(hc++) + " '" + C["action"].get<std::string>() + "'";
        HUSpot sp;
        bool mine = acpc::spot(C["action"], C["pos"], C["hole"].get<std::vector<std::string>>(),
                               C["board"].get<std::vector<std::string>>(), g, sp);
        hturn.eq(mine, 1, w);
        if (!mine) continue;
        hto.eq(sp.min_to, C["legal"]["min_to"].get<long>(), w + " min_to");
        hto.eq(sp.max_to, C["legal"]["max_to"].get<long>(), w + " max_to");
        auto ans = bp.decide(sp, true, 50);
        if (C["node"].is_null()) {
            hnode.eq(ans.on_tree, 0, w);
            continue;
        }
        hnode.eq(ans.node, C["node"].get<long>(), w);
        hbk.eq(ans.bucket, C["bucket"].get<long>(), w);
        auto& d = C["dist"];
        hn.eq(long(ans.options.size()), long(d.size()), w);
        for (size_t k = 0; k < std::min(ans.options.size(), d.size()); ++k) {
            const auto& o = ans.options[k];
            std::string kind = d[k][0];
            char want = kind == "fold" ? 'f' : kind == "call" ? 'k' : 'b';
            hkind.eq(o.kind, want, w + " option " + std::to_string(k));
            hprob.cmp(o.prob, d[k][2].get<double>(), w + " option " + std::to_string(k));
            if (o.kind == 'b' && !d[k][1].is_null()) hto.eq(o.to, d[k][1].get<long>(), w + " option " + std::to_string(k));
        }
    }

    bool ok = true;
    std::printf("kestrel self-test: %zu 9-max hands, %d ICM cases, %d heads-up decisions\n", V["hands"].size(), c, hc);
    for (Check* k : {&actor, &street, &legal, &targets, &feat, &str, &logit, &value, &done, &pay, &eq0, &icmc, &hturn,
                     &hnode, &hbk, &hn, &hkind, &hprob, &hto}) {
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
