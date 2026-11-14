// The 9-max network (scripts/ninemax_selfplay.py NineNet) and its input features (features()), on the CPU.
// Weights come from models/ninemax/<name>/net.bin + net.json (scripts/advisor_vectors.py).
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
    std::vector<float> pre_strength;  // [169]: pre-flop class strength, 0..1
};

// what the features need beyond the table: tournament payouts and the start-of-hand ICM equities (empty for cash)
struct TourneyCtx {
    std::vector<double> pay;  // prize share for places 1..P
    std::vector<double> eq0;  // each seat's ICM equity at the start of the hand
};

// the public features of the table for its current actor [nfeat]
std::vector<float> features(const Table& T, const NetInfo& info, const TourneyCtx* ctx);

// 0..1 strength of a bucket on a street: pre-flop the class strength, after it bucket / (B - 1)
float strength(const NetInfo& info, int street, int bucket);

class Net {
  public:
    explicit Net(const std::string& dir);  // reads dir/net.json and dir/net.bin
    const NetInfo& info() const { return info_; }
    // logits [A] and value for one decision. buckets: the actor's bucket per street (pre-flop class 0..168 first)
    void forward(const std::vector<float>& feat, const std::array<int, 4>& buckets, int street, float strength,
                 std::vector<float>& logits, float& value) const;
    // policy over the legal slots (softmax of the logits, zero where illegal)
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

}  // namespace kestrel
