#include "kestrel/net.hpp"

#include "kestrel/icm.hpp"

#include <algorithm>
#include <map>
#include <mutex>
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
        } else {
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

}

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
        if (info.feat_bounty > 0) {  // every seat's bounty, mine first
            auto b = [&](int j) { return j < int(ctx->bounty.size()) ? float(ctx->bounty[j]) * P : 0.0f; };
            push(b(i));
            for (int r : rel) push(b(r));
        }
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
    info_.feat_bounty = j["features"].value("bounty", 0.0);
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
        } else if (name.rfind("blocks.", 0) == 0) {
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

namespace {
// y[i][o] = b[o] + w[o] . x[i] for B inputs at once: tiles of 4 weight rows by 4 inputs, each weight and input value
// loaded once per tile and the sums vectorised (omp simd), so the batch runs several times faster than B single
// passes. The sums are taken in a different order from linear(), so results differ from it in the last bits.
// on x86-64 Linux the kernel is built twice, for AVX2 with FMA and for the baseline, and the loader picks the one the
// CPU runs (released binaries target the baseline, so they would otherwise never use the wider registers)
#if defined(__x86_64__) && defined(__linux__) && defined(__GNUC__) && !defined(__clang__)
__attribute__((target_clones("arch=haswell", "default")))
#endif
void linear_batch(const std::vector<float>& w, const std::vector<float>& b, int out, int in, const float* X, float* Y,
                  int B) {
    const int O4 = out / 4 * 4, B4 = B / 4 * 4;
#pragma omp parallel for schedule(static) if (size_t(B) * size_t(out) * size_t(in) > 4000000)
    for (int o = 0; o < O4; o += 4) {
        const float *r0 = w.data() + size_t(o) * in, *r1 = r0 + in, *r2 = r1 + in, *r3 = r2 + in;
        for (int i = 0; i < B4; i += 4) {
            const float *x0 = X + size_t(i) * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
            float s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
            float s20 = 0, s21 = 0, s22 = 0, s23 = 0, s30 = 0, s31 = 0, s32 = 0, s33 = 0;
#pragma omp simd reduction(+ : s00, s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23, s30, s31, s32, s33)
            for (int k = 0; k < in; ++k) {
                const float a0 = x0[k], a1 = x1[k], a2 = x2[k], a3 = x3[k];
                s00 += r0[k] * a0, s01 += r0[k] * a1, s02 += r0[k] * a2, s03 += r0[k] * a3;
                s10 += r1[k] * a0, s11 += r1[k] * a1, s12 += r1[k] * a2, s13 += r1[k] * a3;
                s20 += r2[k] * a0, s21 += r2[k] * a1, s22 += r2[k] * a2, s23 += r2[k] * a3;
                s30 += r3[k] * a0, s31 += r3[k] * a1, s32 += r3[k] * a2, s33 += r3[k] * a3;
            }
            float* y0 = Y + size_t(i) * out + o;
            const size_t st = size_t(out);
            y0[0] = s00 + b[o], y0[1] = s10 + b[o + 1], y0[2] = s20 + b[o + 2], y0[3] = s30 + b[o + 3];
            y0[st] = s01 + b[o], y0[st + 1] = s11 + b[o + 1], y0[st + 2] = s21 + b[o + 2], y0[st + 3] = s31 + b[o + 3];
            y0[2 * st] = s02 + b[o], y0[2 * st + 1] = s12 + b[o + 1], y0[2 * st + 2] = s22 + b[o + 2],
                    y0[2 * st + 3] = s32 + b[o + 3];
            y0[3 * st] = s03 + b[o], y0[3 * st + 1] = s13 + b[o + 1], y0[3 * st + 2] = s23 + b[o + 2],
                    y0[3 * st + 3] = s33 + b[o + 3];
        }
    }
    // the edges: inputs past the last full tile, outputs past the last full tile
    auto one = [&](int o, int i) {
        const float *r = w.data() + size_t(o) * in, *x = X + size_t(i) * in;
        float s = 0;
#pragma omp simd reduction(+ : s)
        for (int k = 0; k < in; ++k) s += r[k] * x[k];
        Y[size_t(i) * out + o] = s + b[o];
    };
    for (int o = 0; o < O4; ++o)
        for (int i = B4; i < B; ++i) one(o, i);
    for (int o = O4; o < out; ++o)
        for (int i = 0; i < B; ++i) one(o, i);
}
}  // namespace

void Net::forward_batch(int B, const float* feat, const int* bk, const int* street, const float* st, float* logits,
                        float* value) const {
    const int nf = info_.nfeat, nin = inp_.in, H = hidden_;
    std::vector<float> X(size_t(B) * nin), h(size_t(B) * H), a(size_t(B) * H), b(size_t(B) * H);
    for (int i = 0; i < B; ++i) {
        float* x = X.data() + size_t(i) * nin;
        std::copy(feat + size_t(i) * nf, feat + size_t(i + 1) * nf, x);
        int at = nf;
        for (int s = 0; s < 4; ++s) {
            int row = std::min(bk[i * 4 + s], emb_rows_[s] - 1);
            float on = street[i] >= s ? 1.0f : 0.0f;
            for (int k = 0; k < emb_dim_; ++k) x[at++] = emb_[s][size_t(row) * emb_dim_ + k] * on;
        }
        x[at] = st[i];
    }
    linear_batch(inp_.w, inp_.b, H, nin, X.data(), h.data(), B);
    for (size_t k = 0; k < b1_.size(); ++k) {
        for (int i = 0; i < B; ++i) layernorm(bln_[k].w, bln_[k].b, h.data() + size_t(i) * H, a.data() + size_t(i) * H, H);
        linear_batch(b1_[k].w, b1_[k].b, H, H, a.data(), b.data(), B);
        for (auto& z : b) z = 0.5f * z * (1.0f + std::erf(z * 0.70710678118654752f));
        linear_batch(b2_[k].w, b2_[k].b, H, H, b.data(), a.data(), B);
        for (size_t q = 0; q < h.size(); ++q) h[q] += a[q];
    }
    for (int i = 0; i < B; ++i) layernorm(norm_.w, norm_.b, h.data() + size_t(i) * H, a.data() + size_t(i) * H, H);
    linear_batch(pi_.w, pi_.b, pi_.out, H, a.data(), logits, B);
    if (value) linear_batch(v_.w, v_.b, 1, H, a.data(), value, B);
}

void Net::forward(const std::vector<float>& feat, const std::array<int, 4>& bk, int street, float st,
                  std::vector<float>& logits, float& value, std::vector<float>* hidden) const {
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
        for (auto& z : b) z = 0.5f * z * (1.0f + std::erf(z * 0.70710678118654752f));
        linear(b2_[k].w, b2_[k].b, H, H, b.data(), a.data());
        for (int q = 0; q < H; ++q) h[q] += a[q];
    }
    layernorm(norm_.w, norm_.b, h.data(), a.data(), H);
    logits.assign(pi_.out, 0.0f);
    linear(pi_.w, pi_.b, pi_.out, H, a.data(), logits.data());
    linear(v_.w, v_.b, 1, H, a.data(), &value);
    if (hidden) hidden->assign(a.begin(), a.begin() + H);
}

std::vector<float> embed_hidden(const std::vector<float>& h) {
    const int n = int(h.size());
    // the projection matrix, generated once per input size from a fixed seed (splitmix64 + Box-Muller): portable,
    // unlike std::normal_distribution, whose numbers differ between standard libraries
    static std::mutex mu;
    static std::map<int, std::vector<float>> mats;
    const std::vector<float>* M;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto& m = mats[n];
        if (m.empty()) {
            uint64_t x = 0x6b657374726c3132ULL;
            auto next = [&x] {
                uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                return double((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
            };
            m.resize(size_t(kEmbedDim) * n);
            for (size_t k = 0; k < m.size(); k += 2) {
                double u1 = std::max(next(), 1e-300), u2 = next(), r = std::sqrt(-2.0 * std::log(u1));
                m[k] = float(r * std::cos(6.283185307179586 * u2));
                if (k + 1 < m.size()) m[k + 1] = float(r * std::sin(6.283185307179586 * u2));
            }
        }
        M = &m;
    }
    std::vector<float> e(kEmbedDim, 0.0f);
    double norm = 0;
    for (int o = 0; o < kEmbedDim; ++o) {
        double s = 0;
        const float* r = M->data() + size_t(o) * n;
        for (int k = 0; k < n; ++k) s += double(r[k]) * h[k];
        e[o] = float(s), norm += s * s;
    }
    norm = std::sqrt(norm);
    if (norm > 0)
        for (auto& v : e) v = float(v / norm);
    return e;
}

float cosine(const float* a, const float* b, int n) {
    double ab = 0, aa = 0, bb = 0;
    for (int k = 0; k < n; ++k) ab += double(a[k]) * b[k], aa += double(a[k]) * a[k], bb += double(b[k]) * b[k];
    return aa > 0 && bb > 0 ? float(ab / std::sqrt(aa * bb)) : 0.0f;
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

std::vector<double> start_equity(const std::vector<double>& stacks, const TourneyCtx& ctx) {
    auto eq = icm(stacks, ctx.pay);
    double total = 0;
    for (double x : stacks) total += x;
    double pool = 0;  // every bounty, shared by chips (ninemax_selfplay.start_equity): the equities keep the money whole
    for (double b : ctx.bounty) pool += b;
    for (size_t j = 0; j < eq.size() && !ctx.bounty.empty(); ++j) eq[j] += pool * stacks[j] / std::max(total, 1.0);
    return eq;
}

}
