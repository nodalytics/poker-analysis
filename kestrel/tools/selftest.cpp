#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "kestrel/codec_acpc.hpp"
#include "kestrel/headsup.hpp"
#include "kestrel/icm.hpp"
#include "kestrel/net.hpp"
#include "kestrel/ninemax_spot.hpp"
#include "kestrel/buckets.hpp"
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

}

int main(int argc, char** argv) {
    std::string vdir, model, pack, nine;
    for (int k = 1; k + 1 < argc; k += 2) {
        std::string a = argv[k];
        (a == "--vectors" ? vdir : a == "--model" ? model : a == "--pack" ? pack : a == "--nine-max" ? nine : vdir) =
            argv[k + 1];
    }
    if (vdir.empty() || model.empty() || pack.empty()) {
        std::fprintf(stderr, "usage: kestrel_selftest --vectors DIR --model MODEL_DIR --pack PACK_DIR\n");
        return 2;
    }
    std::ifstream vf(nine.empty() ? vdir + "/vectors_nm17.json" : nine);
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
    Check hturn{"hu: our turn", 0}, hnode{"hu: node", 0}, hbk{"hu: bucket", 0}, hn{"hu: options", 0},
        hkind{"hu: kinds", 0}, hprob{"hu: probs", 1e-5}, hto{"hu: raise to", 50}, fagree{"hu floor: agree", 0.005};
    acpc::Game g;
    int hc = 0, fc = 0, floored = 0;
    // one heads-up file through the codec and an engine; strict: every check counts, else only whole-case agreement
    auto run_hu = [&](const std::string& file, HUBlueprint& bp, bool strict, int& n) {
        std::ifstream hf(file);
        if (!hf) return;
        json HV = json::parse(hf);
        long cases = 0, bad = 0;
        for (auto& C : HV["cases"]) {
            std::string w = "case " + std::to_string(n++) + " '" + C["action"].get<std::string>() + "'";
            HUSpot sp;
            bool mine = acpc::spot(C["action"], C["pos"], C["hole"].get<std::vector<std::string>>(),
                                   C["board"].get<std::vector<std::string>>(), g, sp);
            if (strict) hturn.eq(mine, 1, w);
            if (!mine) continue;
            if (strict) {
                hto.eq(sp.min_to, C["legal"]["min_to"].get<long>(), w + " min_to");
                hto.eq(sp.max_to, C["legal"]["max_to"].get<long>(), w + " max_to");
            }
            auto ans = bp.decide(sp, true, 50);
            floored += !strict && ans.floored;
            if (C["node"].is_null()) {
                if (strict) hnode.eq(ans.on_tree, 0, w);
                continue;
            }
            auto& d = C["dist"];
            bool same = ans.node == C["node"].get<long>() && ans.options.size() == d.size();
            for (size_t k = 0; same && k < d.size(); ++k)
                same = std::fabs(ans.options[k].prob - d[k][2].get<double>()) <= 1e-5;
            ++cases, bad += !same;
            if (!strict) continue;
            hnode.eq(ans.node, C["node"].get<long>(), w);
            hbk.eq(ans.bucket, C["bucket"].get<long>(), w);
            hn.eq(long(ans.options.size()), long(d.size()), w);
            for (size_t k = 0; k < std::min(ans.options.size(), d.size()); ++k) {
                const auto& o = ans.options[k];
                std::string kind = d[k][0];
                char want = kind == "fold" ? 'f' : kind == "call" ? 'k' : 'b';
                hkind.eq(o.kind, want, w + " option " + std::to_string(k));
                hprob.cmp(o.prob, d[k][2].get<double>(), w + " option " + std::to_string(k));
                if (o.kind == 'b' && !d[k][1].is_null())
                    hto.eq(o.to, d[k][1].get<long>(), w + " option " + std::to_string(k));
            }
        }
        // the floor ranks tied hands by deck order where Python's sort is arbitrary, so a few cases may differ
        if (!strict && cases) fagree.cmp(double(bad) / double(cases), 0.0, "share of cases that differ");
    };
    HUBlueprint plain(pack, 0.05, 0.0);
    run_hu(vdir + "/vectors_headsup.json", plain, true, hc);
    HUBlueprint with_floor(pack, 0.05, 0.9);
    run_hu(vdir + "/vectors_headsup_floor.json", with_floor, false, fc);

    // the platform adapter: self-played hands at 2-9 players, re-described as a platform would (its own seat order,
    // button, chips at 40 a big blind, the prizes left) must give the network's own decision at every step
    buckets_init(pack);
    Check rturn{"9max replay: ok", 0}, rprob{"9max replay: probs", 1e-5}, rto{"9max replay: to", 1e-6};
    std::mt19937_64 rng(7);
    int rdec = 0;
    const double cb = 40;  // platform chips per big blind
    for (int hand = 0; hand < 300; ++hand) {
        const int n = 2 + hand % 8;
        std::vector<int> who(9, -1);  // Kestrel seat -> platform seat; the platform's button is seat b
        const int b = int(rng() % n);
        if (n == 2) {
            who[0] = b, who[1] = (b + 1) % n;
        } else {
            who[0] = (b + 1) % n, who[1] = (b + 2) % n;
            for (int k = 0; k < n - 2; ++k) who[9 - (n - 2) + k] = (b + 3 + k) % n;
        }
        std::vector<int64_t> stacks(9, 0);
        NMSpot sp;
        sp.players.resize(n);
        for (int s = 0; s < 9; ++s)
            if (who[s] >= 0) {
                stacks[s] = BB * (2 + int64_t(rng() % 60));
                sp.players[who[s]] = {"p" + std::to_string(who[s]), double(stacks[s]) / BB * cb};
            }
        sp.button = "p" + std::to_string(b), sp.sb = cb / 2, sp.bb = cb, sp.ante = info.rules.ante_bb * cb;
        static const double prizes[9] = {0.30, 0.20, 0.145, 0.105, 0.08, 0.065, 0.05, 0.035, 0.02};
        TourneyCtx ctx;
        ctx.pay.assign(9, 0.0);
        for (int k = 0; k < n; ++k) ctx.pay[k] = prizes[k], sp.payouts.push_back(prizes[k]);
        ctx.eq0 = icm(std::vector<double>(stacks.begin(), stacks.end()), ctx.pay);
        std::vector<int> deck(52);
        std::iota(deck.begin(), deck.end(), 0);
        std::shuffle(deck.begin(), deck.end(), rng);
        std::vector<std::array<int, 4>> bk(9);
        std::vector<std::array<int, 2>> hole(9);
        int d = 10 * 2;
        for (int s = 0; s < 9; ++s) {
            if (who[s] < 0) continue;
            hole[s] = {deck[2 * s], deck[2 * s + 1]};
            int c[7] = {hole[s][0], hole[s][1], deck[d], deck[d + 1], deck[d + 2], deck[d + 3], deck[d + 4]};
            bk[s] = {bucket_of(c, 2), bucket_of(c, 5), bucket_of(c, 6), bucket_of(c, 7)};
        }
        Table T(stacks, menu, info.rules);
        const int nb[4] = {0, 3, 4, 5};
        std::vector<int64_t> base(9, 0);
        int cur = 0;
        while (!T.done()) {
            const int i = T.actor(), st = T.street();
            if (st != cur) base = T.contrib, cur = st;
            std::vector<float> t;
            auto L = T.legal(&t);
            auto f = features(T, info, &ctx);
            std::vector<float> lg;
            float v;
            net.forward(f, bk[i], st, strength(info, st, bk[i][st]), lg, v);
            auto p = net.policy(lg, L);
            sp.me = "p" + std::to_string(who[i]);
            sp.hole = {card_str(hole[i][0]), card_str(hole[i][1])};
            sp.board.clear();
            for (int k = 0; k < nb[st]; ++k) sp.board.push_back(card_str(deck[d + k]));
            std::string w = "hand " + std::to_string(hand) + " decision " + std::to_string(rdec++);
            auto adv = ninemax_decide(net, sp);
            rturn.eq(adv.error.empty() && adv.translated == 0, 1, w + " " + adv.error);
            std::vector<double> got(menu.A, 0.0);
            for (auto& o : adv.options) {
                got[o.slot] = o.prob;
                if (o.slot >= 2)
                    rto.cmp(o.to, double((o.slot == menu.A - 1 ? T.avail[i] : int64_t(t[o.slot - 2])) - base[i]) / BB * cb, w);
            }
            for (int a = 0; a < menu.A; ++a) rprob.cmp(got[a], L[a] ? p[a] : 0.0, w + " slot " + std::to_string(a));
            const int act = std::discrete_distribution<int>(p.begin(), p.end())(rng);
            const int64_t to_call = T.bet - T.contrib[i];
            NMAction na{sp.me, st, "", 0};
            if (act == 0) na.kind = "fold";
            else if (act == 1) na.kind = to_call ? "call" : "check";
            else if (act == menu.A - 1) na.kind = "allin", na.to = double(T.avail[i] - base[i]) / BB * cb;
            else na.kind = "raise", na.to = double(int64_t(t[act - 2]) - base[i]) / BB * cb;
            sp.actions.push_back(na);
            T.step(act);
        }
    }

    bool ok = true;
    std::printf("kestrel self-test: %zu 9-max hands, %d ICM cases, %d heads-up decisions, %d with the defence floor "
                "(%d floored), %d replayed platform decisions\n", V["hands"].size(), c, hc, fc, floored, rdec);
    for (Check* k : {&actor, &street, &legal, &targets, &feat, &str, &logit, &value, &done, &pay, &eq0, &icmc, &hturn,
                     &hnode, &hbk, &hn, &hkind, &hprob, &hto, &fagree, &rturn, &rprob, &rto}) {
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
