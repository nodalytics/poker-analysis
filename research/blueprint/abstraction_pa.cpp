// Distribution-aware card abstraction (after Johanson, Burch, Valenzano & Bowling 2013, "Evaluating state-space
// abstractions in extensive-form games"): hands are grouped by the whole distribution of their future strength, not
// its average, so a draw and a made hand with the same expected strength land in different buckets.
//   flop   50-bin CDF of the turn's expected strength (mean river HS) over the 47 turn cards
//   turn   50-bin CDF of the river HS over the 46 river cards
//   river  OCHS: hand strength against each of 8 groups of opponent hands (pre-flop classes by equity, equal combos)
// k-means (k-means++ start, Lloyd iterations) on a sample of each street, then every state is assigned to its
// nearest centre. Squared L2 between CDFs stands in for the 1-D earth mover's distance (L1 between CDFs).
// Needs ABS_IN/river_hs.f32 and ABS_IN/turn_ehs.f32 from `abstraction` (the average-strength abstraction).
// Usage: abstraction_pa ABS_IN OUTDIR [K=500] [SAMPLE=1500000] [ITERS=15]
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

#include "common.hpp"

static const int BINS = 50, OCHS = 8;

static std::vector<float> load_f32(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "missing %s\n", p.c_str());
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  size_t n = ftell(f) / 4;
  fseek(f, 0, SEEK_SET);
  std::vector<float> v(n);
  if (fread(v.data(), 4, n, f) != n) exit(1);
  fclose(f);
  return v;
}

static void write(const std::string& path, const void* data, size_t bytes) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(data, 1, bytes, f) != bytes) {
    fprintf(stderr, "write failed: %s\n", path.c_str());
    exit(1);
  }
  fclose(f);
}

// CDF of values in [0,1] over BINS bins
static void cdf(const float* vals, int n, float* out) {
  float h[BINS] = {0};
  for (int i = 0; i < n; ++i) h[std::min(BINS - 1, (int)(vals[i] * BINS))] += 1.0f / n;
  float c = 0;
  for (int b = 0; b < BINS; ++b) out[b] = (c += h[b]);
}

struct KMeans {
  int k, d;
  std::vector<float> c;  // k x d
  KMeans(int k_, int d_) : k(k_), d(d_), c((size_t)k_ * d_) {}
  int nearest(const float* x) const {
    int best = 0;
    float bd = 1e30f;
    for (int j = 0; j < k; ++j) {
      const float* cj = &c[(size_t)j * d];
      float s = 0;
      for (int t = 0; t < d; ++t) {
        float u = x[t] - cj[t];
        s += u * u;
      }
      if (s < bd) bd = s, best = j;
    }
    return best;
  }
  void fit(const std::vector<float>& X, int iters, uint64_t seed) {
    size_t n = X.size() / d;
    std::mt19937_64 rng(seed);
    // k-means++ start
    std::vector<float> dist(n, 1e30f);
    size_t first = rng() % n;
    std::copy(&X[first * d], &X[first * d] + d, &c[0]);
    for (int j = 1; j < k; ++j) {
      const float* cj = &c[(size_t)(j - 1) * d];
      double tot = 0;
#pragma omp parallel for reduction(+ : tot)
      for (size_t i = 0; i < n; ++i) {
        float s = 0;
        for (int t = 0; t < d; ++t) {
          float u = X[i * d + t] - cj[t];
          s += u * u;
        }
        if (s < dist[i]) dist[i] = s;
        tot += dist[i];
      }
      double r = std::uniform_real_distribution<double>(0, tot)(rng), acc = 0;
      size_t pick = n - 1;
      for (size_t i = 0; i < n; ++i) {
        acc += dist[i];
        if (acc >= r) {
          pick = i;
          break;
        }
      }
      std::copy(&X[pick * d], &X[pick * d] + d, &c[(size_t)j * d]);
    }
    std::vector<int> a(n);
    for (int it = 0; it < iters; ++it) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; ++i) a[i] = nearest(&X[i * d]);
      std::vector<double> sum((size_t)k * d, 0.0);
      std::vector<size_t> cnt(k, 0);
      for (size_t i = 0; i < n; ++i) {
        ++cnt[a[i]];
        for (int t = 0; t < d; ++t) sum[(size_t)a[i] * d + t] += X[i * d + t];
      }
      for (int j = 0; j < k; ++j)
        if (cnt[j])
          for (int t = 0; t < d; ++t) c[(size_t)j * d + t] = (float)(sum[(size_t)j * d + t] / cnt[j]);
      fprintf(stderr, "  k-means iteration %d\n", it + 1);
    }
  }
};

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: abstraction_pa ABS_IN OUTDIR [K] [SAMPLE] [ITERS]\n");
    return 1;
  }
  std::string in = argv[1], out = argv[2];
  int K = argc > 3 ? atoi(argv[3]) : 500;
  size_t SAMPLE = argc > 4 ? strtoull(argv[4], nullptr, 10) : 1500000;
  int ITERS = argc > 5 ? atoi(argv[5]) : 15;
  Indexers ix;
  const uint64_t NF = ix.size(1), NT = ix.size(2), NR = ix.size(3);
  double t0 = omp_get_wtime();
  std::vector<float> hs = load_f32(in + "/river_hs.f32"), tehs = load_f32(in + "/turn_ehs.f32");
  if (hs.size() != NR || tehs.size() != NT) {
    fprintf(stderr, "table sizes do not match\n");
    return 1;
  }
  std::mt19937_64 rng(11);

  // ---- opponent clusters for OCHS: pre-flop classes ranked by mean river strength, 8 groups of equal combos ----
  std::vector<double> cls_sum(169, 0), cls_n(169, 0);
  for (int s = 0; s < 400000; ++s) {
    uint64_t i = rng() % NR;
    uint8_t c[7];
    hand_unindex(&ix.river, 1, i, c);
    int k = (int)hand_index_last(&ix.pre, c);
    cls_sum[k] += hs[i];
    cls_n[k] += 1;
  }
  std::vector<int> order(169);
  for (int k = 0; k < 169; ++k) order[k] = k;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return cls_sum[a] / cls_n[a] < cls_sum[b] / cls_n[b]; });
  std::vector<int> combos(169), group(169);
  for (int k = 0; k < 169; ++k) {
    uint8_t c[2];
    hand_unindex(&ix.pre, 0, k, c);
    combos[k] = (c[0] >> 2) == (c[1] >> 2) ? 6 : ((c[0] & 3) == (c[1] & 3) ? 4 : 12);
  }
  int acc = 0;
  for (int r = 0; r < 169; ++r) {
    group[order[r]] = std::min(OCHS - 1, acc * OCHS / 1326);
    acc += combos[order[r]];
  }
  std::vector<int8_t> combo_group(52 * 52, -1);
  for (int a = 0; a < 52; ++a)
    for (int b = 0; b < 52; ++b)
      if (a != b) {
        uint8_t c[2] = {(uint8_t)a, (uint8_t)b};
        combo_group[a * 52 + b] = (int8_t)group[hand_index_last(&ix.pre, c)];
      }
  fprintf(stderr, "opponent groups ready %.0fs\n", omp_get_wtime() - t0);

  auto river_feat = [&](uint64_t i, float* f) {
    uint8_t c[7];
    hand_unindex(&ix.river, 1, i, c);
    bool used[52] = {false};
    for (int k = 0; k < 7; ++k) used[c[k]] = true;
    int me = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    float w[OCHS] = {0}, n[OCHS] = {0};
    for (int a = 0; a < 52; ++a) {
      if (used[a]) continue;
      for (int b = a + 1; b < 52; ++b) {
        if (used[b]) continue;
        int op = evaluate_7cards(a, b, c[2], c[3], c[4], c[5], c[6]);
        int g = combo_group[a * 52 + b];
        w[g] += me < op ? 1.0f : me == op ? 0.5f : 0.0f;
        n[g] += 1;
      }
    }
    for (int g = 0; g < OCHS; ++g) f[g] = n[g] > 0 ? w[g] / n[g] : 0.5f;
  };
  auto turn_feat = [&](uint64_t i, float* f) {
    uint8_t c[7];
    hand_unindex(&ix.turn, 1, i, c);
    bool used[52] = {false};
    for (int k = 0; k < 6; ++k) used[c[k]] = true;
    float v[46];
    int n = 0;
    for (int r = 0; r < 52; ++r) {
      if (used[r]) continue;
      c[6] = (uint8_t)r;
      v[n++] = hs[hand_index_last(&ix.river, c)];
    }
    cdf(v, n, f);
  };
  auto flop_feat = [&](uint64_t i, float* f) {
    uint8_t c[7];
    hand_unindex(&ix.flop, 1, i, c);
    bool used[52] = {false};
    for (int k = 0; k < 5; ++k) used[c[k]] = true;
    float v[47];
    int n = 0;
    for (int t = 0; t < 52; ++t) {
      if (used[t]) continue;
      c[5] = (uint8_t)t;
      v[n++] = tehs[hand_index_last(&ix.turn, c)];
    }
    cdf(v, n, f);
  };

  auto run = [&](const char* name, uint64_t N, int d, auto feat) {
    size_t S = std::min<uint64_t>(SAMPLE, N);
    std::vector<uint64_t> idx(S);
    for (size_t s = 0; s < S; ++s) idx[s] = S == N ? s : rng() % N;
    std::vector<float> X(S * d);
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t s = 0; s < S; ++s) feat(idx[s], &X[s * d]);
    fprintf(stderr, "%s: %zu sample features %.0fs\n", name, S, omp_get_wtime() - t0);
    KMeans km(K, d);
    km.fit(X, ITERS, 7);
    std::vector<uint16_t> b(N);
#pragma omp parallel for schedule(dynamic, 4096)
    for (uint64_t i = 0; i < N; ++i) {
      float f[BINS];
      feat(i, f);
      b[i] = (uint16_t)km.nearest(f);
    }
    write(out + "/" + name + "_buckets.u16", b.data(), b.size() * 2);
    write(out + "/" + name + "_centres.f32", km.c.data(), km.c.size() * 4);
    std::vector<size_t> cnt(K, 0);
    for (auto x : b) ++cnt[x];
    fprintf(stderr, "%s done %.0fs, bucket sizes %zu..%zu\n", name, omp_get_wtime() - t0,
            *std::min_element(cnt.begin(), cnt.end()), *std::max_element(cnt.begin(), cnt.end()));
  };
  run("flop", NF, BINS, flop_feat);
  run("turn", NT, BINS, turn_feat);
  run("river", NR, OCHS, river_feat);
  std::ofstream m(out + "/meta.txt");
  m << "buckets " << K << "\nflop_states " << NF << "\nturn_states " << NT << "\nriver_states " << NR
    << "\nmetric flop,turn=CDF of next-street strength (k-means, L2 on CDF ~ EMD) river=OCHS(8) k-means"
    << "\nseconds " << omp_get_wtime() - t0 << "\n";
  return 0;
}
