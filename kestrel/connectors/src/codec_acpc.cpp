#include "kestrel/codec_acpc.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace kestrel::acpc {

std::vector<std::vector<Token>> tokens(const std::string& a) {
    std::vector<std::vector<Token>> st(1);
    for (size_t k = 0; k < a.size();) {
        char ch = a[k];
        if (ch == '/') {
            st.emplace_back();
            ++k;
        } else if (ch == 'b') {
            size_t j = k + 1;
            while (j < a.size() && std::isdigit(static_cast<unsigned char>(a[j]))) ++j;
            if (j == k + 1) throw std::invalid_argument("acpc: bet without an amount in " + a);
            st.back().push_back({'b', std::stoll(a.substr(k + 1, j - k - 1))});
            k = j;
        } else if (ch == 'k' || ch == 'c' || ch == 'f') {
            st.back().push_back({ch, 0});
            ++k;
        } else {
            throw std::invalid_argument(std::string("acpc: unexpected '") + ch + "' in " + a);
        }
    }
    return st;
}

bool spot(const std::string& action, int pos, const std::vector<std::string>& hole,
          const std::vector<std::string>& board, const Game& g, HUSpot& s) {
    auto st = tokens(action);
    s = HUSpot{};
    int64_t prev[2] = {0, 0};
    int64_t street[2] = {0, 0}, last = 0, raise_size = g.big;
    int actor = 1;
    bool over = false;
    for (size_t si = 0; si < st.size(); ++si) {
        if (si == 0) {
            street[0] = g.big, street[1] = g.big / 2;
            actor = 1, last = g.big, raise_size = g.big;
        } else {
            street[0] = street[1] = 0;
            actor = 0, last = 0, raise_size = g.big;
        }
        s.streets.emplace_back();
        for (const Token& t : st[si]) {
            s.streets.back().push_back({1 - actor, t.kind, t.to});
            if (t.kind == 'b') {
                raise_size = std::max(raise_size, t.to - last);
                last = t.to;
                street[actor] = t.to;
            } else if (t.kind == 'c') {
                street[actor] = last;
            } else if (t.kind == 'f') {
                over = true;
            }
            actor = 1 - actor;
        }
        if (si + 1 < st.size())
            for (int p = 0; p < 2; ++p) prev[p] += street[p];
    }
    if (over || actor != pos) return false;
    int64_t max_to = g.stack - prev[pos];
    s.pot = prev[0] + prev[1] + street[0] + street[1];
    s.my_bet = street[pos], s.their_bet = street[1 - pos];
    s.my_stack = g.stack - prev[pos] - street[pos], s.their_stack = g.stack - prev[1 - pos] - street[1 - pos];
    s.me = 1 - pos;
    s.min_to = std::min(last + raise_size, max_to);
    s.max_to = max_to;
    s.facing = street[pos] < last;
    s.can_raise = s.min_to <= s.max_to && s.max_to > last;
    if (hole.size() != 2) throw std::invalid_argument("acpc: two hole cards expected");
    s.hole = {card_from(hole[0]), card_from(hole[1])};
    for (const auto& c : board) s.board.push_back(card_from(c));
    size_t want[] = {0, 3, 4, 5};
    if (s.streets.size() > 4 || s.board.size() != want[s.streets.size() - 1])
        throw std::invalid_argument("acpc: the board does not match the street");
    return true;
}

std::string encode(const HUOption& o, const HUSpot& s) {
    if (o.kind == 'b' && s.can_raise) return "b" + std::to_string(std::clamp(o.to, s.min_to, s.max_to));
    if (o.kind == 'f' && s.facing) return "f";
    return s.facing ? "c" : "k";
}

}
