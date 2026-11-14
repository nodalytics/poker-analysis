#include "kestrel/net.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace kestrel {

namespace {

float half_to_float(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: normalise
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) man <<= 1, --exp;
            bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void linear(const std::vector<float>& w, const std::vector<float>& b, int out, int in, const float* x, float* y) {
    for (int o = 0; o < out; ++o) {
        const float* r = w.data() + size_t(o) * in;
        float s = 0;
        for (int k = 0; k < in; ++k) s += r[k] * x[k];
        y[o] = s + b[o];
    }
}

void layernorm(const std::vector<float>& w, const std::vector<float>& b, const float* x, float* y, int n) {
    double m = 0, v = 0;
    for (int k = 0; k < n; ++k) m += x[k];
    m /= n;
    for (int k = 0; k < n; ++k) v += (x[k] - m) * (x[k] - m);
    v /= n;
    float inv = float(1.0 / std::sqrt(v + 1e-5));
    for (int k = 0; k < n; ++k) y[k] = (x[k] - float(m)) * inv * w[k] + b[k];
}

}  // namespace

float strength(const NetInfo& info, int street, int bucket) {
    if (street == 0) return info.pre_strength[std::min(bucket, 168)];
    return float(bucket) / float(info.buckets - 1);
}

std::vector<float> features(const Table& T, const NetInfo& info, const TourneyCtx* ctx) {
    const int P = T.P(), i = T.actor();
    if (i < 0) throw std::logic_error("features: the hand is over");
    std::vector<float> f;
    f.reserve(info.nfeat);
    auto push = [&](double x) { f.push_back(float(x)); };
    std::vector<float> c(P);
    for (int j = 0; j < P; ++j) c[j] = float(T.contrib[j]);
    float pot = std::accumulate(c.begin(), c.end(), 0.0f) + float(T.dead);
    float mine = c[i], bet = float(T.bet), to_call = bet - mine;
    float left = float(T.avail[i]) - mine;
    for (int j = 0; j < P; ++j) push(j == i);
    for (int s = 0; s < 4; ++s) push(s == T.street());
    int n_live = 0, n_allin = 0, n_need = 0;
    auto need = T.needs();
    for (int j = 0; j < P; ++j) n_live += !T.folded[j], n_allin += T.allin[j], n_need += need[j];
    push(pot / float(P * STACK));
    push(to_call / std::max(pot + to_call, 1.0f));
    push(mine / float(STACK));
    push(left / float(STACK));
    push(std::min(left / std::max(pot, 1.0f), 20.0f) / 20.0f);
    push(float(T.raises) / 4.0f);
    push(float(T.pre_raises) / 4.0f);
    push(float(n_live) / P);
    push(float(n_allin) / P);
    push(float(n_need) / P);
    std::vector<int> rel(P - 1);
    for (int k = 0; k < P - 1; ++k) rel[k] = (i + 1 + k) % P;
    for (int r : rel) push(!T.folded[r]);
    for (int r : rel) push(T.allin[r]);
    for (int r : rel) push(c[r] / float(STACK));
    for (int r : rel) push(T.acted[r]);
    for (int r : rel) push(r == T.aggressor);
    if (info.feat_stacks || info.feat_tourney) {
        std::vector<float> behind(P);
        for (int j = 0; j < P; ++j) behind[j] = float(T.avail[j]) - c[j];
        push(std::log(float(T.avail[i]) / float(BB)) / std::log(200.0f));
        for (int r : rel) push(std::clamp(behind[r] / float(STACK), 0.0f, 3.0f));
        float opp = 0;
        bool any = false;
        for (int j = 0; j < P; ++j) {
            float x = (!T.folded[j] && j != i) ? behind[j] : 0.0f;
            opp = any ? std::max(opp, x) : x, any = true;
        }
        push(std::clamp(std::min(left, opp) / float(STACK), 0.0f, 3.0f));
    }
    if (info.feat_tourney) {
        if (!ctx) throw std::invalid_argument("features: a tournament network needs payouts and ICM equities");
        for (int j = 0; j < P; ++j) push(float(ctx->pay[j]) * P);
        push(float(ctx->eq0[i]) * P);
        for (int r : rel) push(float(ctx->eq0[r]) * P);
        float tot = 0;
        for (int j = 0; j < P; ++j) tot += float(T.stack[j]);
        float share = std::max(float(T.stack[i]) / tot, 1e-6f);
        push(std::clamp(float(ctx->eq0[i]) / share, 0.0f, 5.0f));
    }
    std::vector<float> t;
    auto L = T.legal(&t);
    for (auto x : L) push(x);
    const int S = T.menu().S;
    for (int k = 0; k < S; ++k)
        push(std::clamp((t[k] - bet) / std::max(pot, 1.0f), 0.0f, 10.0f) / 10.0f * float(L[2 + k]));
    if (int(f.size()) != info.nfeat)
        throw std::logic_error("features: " + std::to_string(f.size()) + " features, the network reads " +
                               std::to_string(info.nfeat));
    return f;
}

Net::Net(const std::string& dir) {
    std::ifstream jf(dir + "/net.json");
    if (!jf) throw std::runtime_error("no " + dir + "/net.json");
    nlohmann::json j = nlohmann::json::parse(jf);
    info_.nfeat = j["nfeat"], info_.actions = j["actions"], info_.buckets = j["buckets"];
    info_.players = j.value("players", 9);
    info_.menu = j["menu"];
    info_.rules.ante_bb = j.value("ante_bb", 0.0);
    info_.rules.open_limp = j.value("open_limp", true);
    info_.feat_stacks = j["features"].value("stacks", false);
    info_.feat_tourney = j["features"].value("tourney", false);
    info_.pre_strength = j["pre_strength"].get<std::vector<float>>();

    std::ifstream bf(dir + "/net.bin", std::ios::binary);
    if (!bf) throw std::runtime_error("no " + dir + "/net.bin");
    auto read = [&](size_t n) {
        std::vector<uint16_t> h(n);
        bf.read(reinterpret_cast<char*>(h.data()), std::streamsize(n * 2));
        if (!bf) throw std::runtime_error("net.bin is shorter than net.json says");
        std::vector<float> r(n);
        for (size_t k = 0; k < n; ++k) r[k] = half_to_float(h[k]);
        return r;
    };
    // tensors in state_dict order; place each by name
    int nblocks = 0;
    for (auto& t : j["tensors"]) {
        std::string name = t[0];
        if (name.rfind("blocks.", 0) == 0) nblocks = std::max(nblocks, std::stoi(name.substr(7)) + 1);
    }
    bln_.resize(nblocks), b1_.resize(nblocks), b2_.resize(nblocks);
    for (auto& t : j["tensors"]) {
        std::string name = t[0];
        std::vector<int> shape = t[1].get<std::vector<int>>();
        size_t n = 1;
        for (int d : shape) n *= size_t(d);
        auto data = read(n);
        auto lin_w = [&](Lin& L) { L.out = shape[0], L.in = shape[1], L.w = std::move(data); };
        if (name.rfind("emb.", 0) == 0) {
            int s = name[4] - '0';
            emb_rows_[s] = shape[0], emb_dim_ = shape[1], emb_[s] = std::move(data);
        } else if (name == "inp.weight") {
            lin_w(inp_);
            hidden_ = shape[0];
        } else if (name == "inp.bias") {
            inp_.b = std::move(data);
        } else if (name == "pi.weight") {
            lin_w(pi_);
        } else if (name == "pi.bias") {
            pi_.b = std::move(data);
        } else if (name == "v.weight") {
            lin_w(v_);
        } else if (name == "v.bias") {
            v_.b = std::move(data);
        } else if (name == "norm.weight") {
            norm_.w = std::move(data);
        } else if (name == "norm.bias") {
            norm_.b = std::move(data);
        } else if (name.rfind("blocks.", 0) == 0) {  // blocks.<k>.<0 LayerNorm | 1 Linear | 3 Linear>.<weight|bias>
            int k = std::stoi(name.substr(7));
            size_t dot = name.find('.', 7);
            int part = name[dot + 1] - '0';
            bool w = name.substr(name.rfind('.') + 1) == "weight";
            if (part == 0) (w ? bln_[k].w : bln_[k].b) = std::move(data);
            else if (part == 1) w ? lin_w(b1_[k]) : void(b1_[k].b = std::move(data));
            else if (part == 3) w ? lin_w(b2_[k]) : void(b2_[k].b = std::move(data));
            else throw std::runtime_error("unexpected tensor " + name);
        } else {
            throw std::runtime_error("unexpected tensor " + name);
        }
    }
    if (bf.peek() != EOF) throw std::runtime_error("net.bin is longer than net.json says");
    if (inp_.in != info_.nfeat + 4 * emb_dim_ + 1) throw std::runtime_error("net.json: input size mismatch");
}

void Net::forward(const std::vector<float>& feat, const std::array<int, 4>& bk, int street, float st,
                  std::vector<float>& logits, float& value) const {
    std::vector<float> x(feat);
    for (int s = 0; s < 4; ++s) {
        int row = std::min(bk[s], emb_rows_[s] - 1);
        float on = street >= s ? 1.0f : 0.0f;
        for (int k = 0; k < emb_dim_; ++k) x.push_back(emb_[s][size_t(row) * emb_dim_ + k] * on);
    }
    x.push_back(st);
    const int H = hidden_;
    std::vector<float> h(H), a(H), b(H);
    linear(inp_.w, inp_.b, H, inp_.in, x.data(), h.data());
    for (size_t k = 0; k < b1_.size(); ++k) {
        layernorm(bln_[k].w, bln_[k].b, h.data(), a.data(), H);
        linear(b1_[k].w, b1_[k].b, H, H, a.data(), b.data());
        for (auto& z : b) z = 0.5f * z * (1.0f + std::erf(z * 0.70710678118654752f));  // exact GELU
        linear(b2_[k].w, b2_[k].b, H, H, b.data(), a.data());
        for (int q = 0; q < H; ++q) h[q] += a[q];
    }
    layernorm(norm_.w, norm_.b, h.data(), a.data(), H);
    logits.assign(pi_.out, 0.0f);
    linear(pi_.w, pi_.b, pi_.out, H, a.data(), logits.data());
    linear(v_.w, v_.b, 1, H, a.data(), &value);
}

std::vector<float> Net::policy(const std::vector<float>& logits, const std::vector<uint8_t>& legal) const {
    std::vector<float> p(logits.size(), 0.0f);
    float mx = -INFINITY;
    for (size_t k = 0; k < logits.size(); ++k)
        if (legal[k]) mx = std::max(mx, logits[k]);
    double z = 0;
    for (size_t k = 0; k < logits.size(); ++k)
        if (legal[k]) z += p[k] = std::exp(logits[k] - mx);
    for (auto& q : p) q = float(q / z);
    return p;
}

}  // namespace kestrel
