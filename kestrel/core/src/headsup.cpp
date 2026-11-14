#include "kestrel/headsup.hpp"

#include "kestrel/buckets.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>


namespace kestrel {

int card_from(const std::string& s) {
    static const std::string R = "23456789TJQKA", S = "cdhs";
    if (s.size() != 2) throw std::invalid_argument("bad card: " + s);
    auto r = R.find(s[0]), u = S.find(char(std::tolower(s[1])));
    if (r == std::string::npos || u == std::string::npos) throw std::invalid_argument("bad card: " + s);
    return int(r) * 4 + int(u);
}

std::string card_str(int c) {
    static const char* R = "23456789TJQKA";
    static const char* S = "cdhs";
    return std::string(1, R[c >> 2]) + S[c & 3];
}

namespace {
std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    if (s.empty()) return out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, sep)) out.push_back(x);
    return out;
}
}  // namespace

HUBlueprint::HUBlueprint(const std::string& dir, double purify) : purify_(purify) {
    buckets_init(dir);
    std::ifstream t(dir + "/tree.tsv");
    if (!t) throw std::runtime_error("no " + dir + "/tree.tsv");
    std::string line;
    std::getline(t, line);  // header
    while (std::getline(t, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> f;
        size_t a = 0;
        for (size_t b; (b = line.find('\t', a)) != std::string::npos; a = b + 1) f.push_back(line.substr(a, b - a));
        f.push_back(line.substr(a));
        while (f.size() < 9) f.emplace_back();
        int id = std::stoi(f[0]);
        if (id >= int(nodes_.size())) nodes_.resize(id + 1);
        Node& n = nodes_[id];
        n.kind = std::stoi(f[1]), n.player = std::stoi(f[2]), n.street = std::stoi(f[3]);
        n.c[0] = std::stoll(f[4]), n.c[1] = std::stoll(f[5]), n.offset = std::stoull(f[6]);
        n.labels = split(f[7], ',');
        for (auto& c : split(f[8], ',')) n.children.push_back(std::stoi(c));
    }
    std::ifstream a(dir + "/avg.u8", std::ios::binary | std::ios::ate);
    if (!a) throw std::runtime_error("no " + dir + "/avg.u8");
    avg_.resize(size_t(a.tellg()));
    a.seekg(0);
    a.read(reinterpret_cast<char*>(avg_.data()), std::streamsize(avg_.size()));
}

int HUBlueprint::bucket(const std::array<int, 2>& hole, const std::vector<int>& board) const {
    std::vector<int> c{hole[0], hole[1]};
    c.insert(c.end(), board.begin(), board.end());
    return bucket_of(c.data(), int(c.size()));
}

double HUBlueprint::frac(double to, double their, double mine, double pot) {
    return (to - their) / std::max(pot + (their - mine), 1.0);  // the raise as a fraction of the pot after calling
}

int HUBlueprint::child(int node, std::initializer_list<const char*> labels) const {
    const Node& n = nodes_[node];
    for (const char* l : labels)
        for (size_t k = 0; k < n.labels.size(); ++k)
            if (n.labels[k] == l) return n.children[k];
    return -1;
}

int HUBlueprint::map_raise(int node, double x, bool det) {
    const Node& n = nodes_[node];
    int p = n.player;
    std::vector<std::pair<double, int>> c;
    for (size_t k = 0; k < n.labels.size(); ++k) {
        const auto& l = n.labels[k];
        if (l == "fold" || l == "check" || l == "call") continue;
        const Node& ch = nodes_[n.children[k]];
        c.emplace_back(frac(double(ch.c[p]), double(n.c[1 - p]), double(n.c[p]), double(n.c[0] + n.c[1])),
                       n.children[k]);
    }
    if (c.empty()) return -1;
    std::sort(c.begin(), c.end());
    if (x <= c.front().first) return c.front().second;
    if (x >= c.back().first) return c.back().second;
    for (size_t k = 0; k + 1 < c.size(); ++k) {
        double A = c[k].first, B = c[k + 1].first;
        if (A <= x && x <= B) {  // pseudo-harmonic mapping (Ganzfried & Sandholm)
            double pa = ((B - x) * (1 + A)) / std::max((B - A) * (1 + x), 1e-9);
            if (det) return pa >= 0.5 ? c[k].second : c[k + 1].second;
            return std::uniform_real_distribution<double>(0, 1)(rng_) < pa ? c[k].second : c[k + 1].second;
        }
    }
    return c.back().second;
}

HUAnswer HUBlueprint::decide(const HUSpot& s, bool det, int64_t round_to) {
    HUAnswer ans;
    int node = 0;
    double bets[2] = {50, 100}, pot = 0;  // blueprint player 0 posts the small blind
    auto rnd = [&](int64_t v) {
        return round_to > 0 ? double(std::llround(double(v) / double(round_to)) * round_to) : double(v);
    };
    for (size_t si = 0; si < s.streets.size(); ++si) {
        if (si > 0) {
            pot += bets[0] + bets[1];
            bets[0] = bets[1] = 0;
        }
        for (const HUAction& a : s.streets[si]) {
            const Node& n = nodes_[node];
            if (n.kind != 0 || n.player != a.player) return ans;
            int p = a.player;
            if (a.kind == 'f') {
                node = child(node, {"fold"});
            } else if (a.kind == 'k' || a.kind == 'c') {
                bets[p] = bets[1 - p];
                node = child(node, {"check", "call"});
            } else {
                double to = rnd(a.to);
                double x = frac(to, bets[1 - p], bets[p], pot + bets[0] + bets[1]);
                bets[p] = to;
                node = map_raise(node, x, det);
            }
            if (node < 0) return ans;
        }
    }
    // the walked node must be our decision on the current street
    const Node& n = nodes_[node];
    int p = s.me;
    if (n.kind != 0 || n.player != p || n.street != int(s.streets.size()) - 1) return ans;
    int b = bucket(s.hole, s.board);
    if (b < 0) return ans;
    ans.node = node, ans.bucket = b;
    size_t k = n.labels.size();
    std::vector<double> st(k);
    double tot = 0;
    for (size_t j = 0; j < k; ++j) tot += st[j] = avg_[n.offset + uint64_t(b) * k + j];
    for (auto& v : st) v = tot > 0 ? v / tot : 1.0 / double(k);
    // purification, then renormalise (as blueprint_bot.distribution)
    tot = 0;
    for (auto& v : st) tot += v = v < purify_ ? 0.0 : v;
    for (auto& v : st) v = tot > 0 ? v / tot : 1.0 / double(k);
    for (size_t j = 0; j < k; ++j) {
        if (st[j] <= 0) continue;
        const std::string& l = n.labels[j];
        if (l == "fold") {
            ans.options.push_back({'f', 0, st[j], l});
        } else if (l == "check" || l == "call" || !s.can_raise) {
            ans.options.push_back({'k', 0, st[j], l});
        } else if (l == "allin") {
            ans.options.push_back({'b', s.max_to, st[j], l});
        } else {
            const Node& ch = nodes_[n.children[j]];
            double f = frac(double(ch.c[p]), double(n.c[1 - p]), double(n.c[p]), double(n.c[0] + n.c[1]));
            double real_to = bets[1 - p] + f * (pot + bets[0] + bets[1] + bets[1 - p] - bets[p]);
            int64_t to = std::llround(real_to);
            if (round_to > 0) to = std::llround(real_to / double(round_to)) * round_to;
            ans.options.push_back({'b', std::clamp(to, s.min_to, s.max_to), st[j], l});
        }
    }
    ans.on_tree = true;
    return ans;
}

}  // namespace kestrel
