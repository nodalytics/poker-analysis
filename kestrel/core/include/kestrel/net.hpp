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
    double feat_bounty = 0;
    bool size_head = false;  // plan B: fold/call/raise/all-in, then a size scored by g(h, size) (research SizeHeadNet)
    int nbase = 0;           // a size-head network's inputs before the 2 x S size columns  // progressive knockouts: the share of the money in bounties the network trained with (0: none)
    std::vector<float> pre_strength;
};

struct TourneyCtx {
    std::vector<double> pay;     // the regular prizes for places 1..P, as shares of the money at stake
    std::vector<double> bounty;  // each seat's bounty, as a share of the same money (empty: none, read as zeros)
    std::vector<double> eq0;     // start_equity
};

// each seat's equity at the start of a hand: ICM on the regular prizes, plus its chip share of all the bounties (each
// is eventually collected by someone, in proportion to chips), as the trainer values it (ninemax_selfplay.start_equity)
std::vector<double> start_equity(const std::vector<double>& stacks, const TourneyCtx& ctx);

// each seat's equity at the end of a hand that started with stacks and paid payoff (chips): ICM on the regular prizes of
// the stacks after it and, with bounties, the knockout term (each bust's bounty to the winner(s) of the pot that took
// its last chips: half in cash, half onto the collector's own head; the bounties still in play by chip share), as the trainer rewards
// it (ninemax_selfplay.end_equity). Without bounties: icm(end_stacks(stacks, payoff), ctx.pay).
// contrib, folded, ranks: the finished hand's (Table::contrib, Table::folded, the showdown ranks, lower is better), which
// decide who collects each bust's bounty: the winner(s) of the pot that took the player's last chips (knockouts).
std::vector<double> end_equity(const std::vector<double>& stacks, const std::vector<double>& payoff, const TourneyCtx& ctx,
                               const std::vector<int64_t>& contrib, const std::vector<uint8_t>& folded,
                               const std::vector<int>& ranks);

std::vector<float> features(const Table& T, const NetInfo& info, const TourneyCtx* ctx);

float strength(const NetInfo& info, int street, int bucket);

// a spot's embedding: the last hidden layer projected to kEmbedDim numbers by a fixed random Gaussian matrix (the same
// on every machine, so stored embeddings stay comparable) and scaled to length 1, so the dot product of two
// embeddings is their cosine similarity. Random projection keeps cosines to about +/-0.1 at 128 numbers.
constexpr int kEmbedDim = 128;
std::vector<float> embed_hidden(const std::vector<float>& hidden);
float cosine(const float* a, const float* b, int n);  // of two vectors (0 when either is zero)

class Net {
  public:
    explicit Net(const std::string& dir);
    const NetInfo& info() const { return info_; }
    // hidden: when given, the last hidden layer (after the final norm), the network's own picture of the spot
    void forward(const std::vector<float>& feat, const std::array<int, 4>& buckets, int street, float strength,
                 std::vector<float>& logits, float& value, std::vector<float>* hidden = nullptr) const;
    std::vector<float> policy(const std::vector<float>& logits, const std::vector<uint8_t>& legal) const;
    // B positions at once (row-major inputs and outputs; value may be null): the same arithmetic as forward, but each
    // weight is read once for the whole batch, which is what makes search affordable on a CPU
    void forward_batch(int B, const float* feat, const int* buckets, const int* street, const float* strength,
                       float* logits, float* value) const;

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
    Lin sz_h_, sz_z_, sz1_, sz2_;  // the size head: g = sz2(gelu(sz1(gelu(sz_h(h) + sz_z(z)))))
    // the joint logits of a size-head network from its last hidden layer a and its inputs
    void size_logits(const float* a, const float* feat, int nfeat, int street, float* logits) const;
    std::vector<LN> bln_;
    std::vector<Lin> b1_, b2_;
    LN norm_;
};

}
