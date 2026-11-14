#pragma once
#include <array>
#include <string>
#include <vector>

#include "kestrel/table.hpp"

namespace kestrel {

struct NetInfo {
    int nfeat = 0, actions = 0, buckets = 0, players = 9;
    std::string menu;
    Rules rules;
    bool feat_stacks = false, feat_tourney = false;
    std::vector<float> pre_strength;
};

struct TourneyCtx {
    std::vector<double> pay;
    std::vector<double> eq0;
};

std::vector<float> features(const Table& T, const NetInfo& info, const TourneyCtx* ctx);

float strength(const NetInfo& info, int street, int bucket);

class Net {
  public:
    explicit Net(const std::string& dir);
    const NetInfo& info() const { return info_; }
    void forward(const std::vector<float>& feat, const std::array<int, 4>& buckets, int street, float strength,
                 std::vector<float>& logits, float& value) const;
    std::vector<float> policy(const std::vector<float>& logits, const std::vector<uint8_t>& legal) const;

  private:
    struct Lin {
        int out = 0, in = 0;
        std::vector<float> w, b;
    };
    struct LN {
        std::vector<float> w, b;
    };
    NetInfo info_;
    std::array<std::vector<float>, 4> emb_;
    std::array<int, 4> emb_rows_{};
    int emb_dim_ = 0, hidden_ = 0;
    Lin inp_, pi_, v_;
    std::vector<LN> bln_;
    std::vector<Lin> b1_, b2_;
    LN norm_;
};

}
