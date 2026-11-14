// Card abstraction for the blueprint. For every suit-isomorphic hand+board state:
//   river  HS   = share of the 990 opponent hands beaten (ties count half)
//   turn   EHS  = mean HS over the 46 rivers, EHS2 = mean HS^2 (squared strength rewards draws that may hit big)
//   flop   EHS, EHS2 = means over the 47 turns of the turn values
// Each street is cut into B equal-population buckets on EHS2 (flop, turn) or HS (river); pre-flop keeps its 169
// classes. Writes OUT/{flop,turn,river}_buckets.u8, OUT/{turn,flop}_ehs.f32 and OUT/meta.txt.
// Usage: abstraction OUTDIR [BUCKETS=200]                       the same count on every street
//        abstraction OUTDIR FLOP TURN RIVER [GAMMA=1]           a count per street; GAMMA > 1 puts the cuts closer
//                                                                together at the strong end (non-uniform buckets)
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <fstream>

#include "common.hpp"

static void write(const std::string& path, const void* data, size_t bytes) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(data, 1, bytes, f) != bytes) {
    fprintf(stderr, "write failed: %s\n", path.c_str());
    exit(1);
  }
  fclose(f);
}

// cut k at quantile g(k/B), g(u) = 1 - (1 - u)^gamma: gamma 1 gives equal-population buckets; above 1 the buckets
// get narrower towards the strong end, where the most chips are decided
static std::vector<uint16_t> quantile_buckets(const std::vector<float>& v, int B, double gamma = 1.0) {
  std::vector<float> s(v);
  std::sort(s.begin(), s.end());
  std::vector<float> cut(B - 1);
  for (int k = 1; k < B; ++k) {
    double q = 1.0 - std::pow(1.0 - (double)k / B, gamma);
    cut[k - 1] = s[std::min(s.size() - 1, (size_t)(q * s.size()))];
    // a tie cannot be split: when a cut lands on the previous cut's value, move it to the next distinct value, so no
    // bucket is left empty while distinct strengths remain (only the leftovers past the strongest value stay empty)
    if (k > 1 && cut[k - 1] <= cut[k - 2]) {
      auto nx = std::upper_bound(s.begin(), s.end(), cut[k - 2]);
      cut[k - 1] = nx == s.end() ? std::numeric_limits<float>::infinity() : *nx;
    }
  }
  std::vector<uint16_t> out(v.size());
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = (uint16_t)(std::upper_bound(cut.begin(), cut.end(), v[i]) - cut.begin());
  return out;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: abstraction OUTDIR [BUCKETS] | OUTDIR FLOP TURN RIVER [GAMMA]\n");
    return 1;
  }
  std::string out = argv[1];
  int B = argc > 2 ? atoi(argv[2]) : 200;
  int BF = B, BT = B, BR = B;
  double gamma = 1.0;
  if (argc >= 5) BF = atoi(argv[2]), BT = atoi(argv[3]), BR = atoi(argv[4]), B = std::max({BF, BT, BR});
  if (argc >= 6) gamma = atof(argv[5]);
  Indexers ix;
  const uint64_t NR = ix.size(3), NT = ix.size(2), NF = ix.size(1);
  fprintf(stderr, "states: flop %lu turn %lu river %lu\n", NF, NT, NR);

  // river hand strength
  std::vector<float> hs(NR);
  double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 4096)
  for (uint64_t i = 0; i < NR; ++i) {
    uint8_t c[7];
    hand_unindex(&ix.river, 1, i, c);
    bool used[52] = {false};
    for (int k = 0; k < 7; ++k) used[c[k]] = true;
    int me = evaluate_7cards(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    int win = 0, tie = 0, n = 0;
    for (int a = 0; a < 52; ++a) {
      if (used[a]) continue;
      for (int b = a + 1; b < 52; ++b) {
        if (used[b]) continue;
        int op = evaluate_7cards(a, b, c[2], c[3], c[4], c[5], c[6]);
        win += me < op;
        tie += me == op;
        ++n;
      }
    }
    hs[i] = (win + 0.5f * tie) / n;
    if (i % 20000000 == 0) fprintf(stderr, "river %lu/%lu %.0fs\n", i, NR, omp_get_wtime() - t0);
  }
  // turn: mean HS and HS^2 over rivers
  std::vector<float> tehs(NT), tehs2(NT);
#pragma omp parallel for schedule(dynamic, 4096)
  for (uint64_t i = 0; i < NT; ++i) {
    uint8_t c[7];
    hand_unindex(&ix.turn, 1, i, c);
    bool used[52] = {false};
    for (int k = 0; k < 6; ++k) used[c[k]] = true;
    double s = 0, s2 = 0;
    int n = 0;
    for (int r = 0; r < 52; ++r) {
      if (used[r]) continue;
      c[6] = (uint8_t)r;
      float h = hs[hand_index_last(&ix.river, c)];
      s += h;
      s2 += h * h;
      ++n;
    }
    tehs[i] = s / n;
    tehs2[i] = s2 / n;
  }
  // flop: means over turns
  std::vector<float> fehs(NF), fehs2(NF);
#pragma omp parallel for schedule(dynamic, 1024)
  for (uint64_t i = 0; i < NF; ++i) {
    uint8_t c[7];
    hand_unindex(&ix.flop, 1, i, c);
    bool used[52] = {false};
    for (int k = 0; k < 5; ++k) used[c[k]] = true;
    double s = 0, s2 = 0;
    int n = 0;
    for (int t = 0; t < 52; ++t) {
      if (used[t]) continue;
      c[5] = (uint8_t)t;
      uint64_t j = hand_index_last(&ix.turn, c);
      s += tehs[j];
      s2 += tehs2[j];
      ++n;
    }
    fehs[i] = s / n;
    fehs2[i] = s2 / n;
  }
  fprintf(stderr, "metrics done %.0fs\n", omp_get_wtime() - t0);
  auto fb = quantile_buckets(fehs2, BF, gamma), tb = quantile_buckets(tehs2, BT, gamma),
       rb = quantile_buckets(hs, BR, gamma);
  auto save = [&](const std::string& name, const std::vector<uint16_t>& v) {
    if (*std::max_element(v.begin(), v.end()) > 255) {
      write(out + "/" + name + "_buckets.u16", v.data(), v.size() * 2);
    } else {
      std::vector<uint8_t> n(v.begin(), v.end());
      write(out + "/" + name + "_buckets.u8", n.data(), n.size());
    }
  };
  save("flop", fb);
  save("turn", tb);
  save("river", rb);
  write(out + "/flop_ehs.f32", fehs.data(), fehs.size() * 4);
  write(out + "/turn_ehs.f32", tehs.data(), tehs.size() * 4);
  write(out + "/river_hs.f32", hs.data(), hs.size() * 4);
  std::ofstream m(out + "/meta.txt");
  m << "buckets " << B << "\nbuckets_flop " << BF << "\nbuckets_turn " << BT << "\nbuckets_river " << BR
    << "\ngamma " << gamma << "\nflop_states " << NF << "\nturn_states " << NT << "\nriver_states " << NR
    << "\nmetric flop,turn=EHS2 river=HS\nseconds " << omp_get_wtime() - t0 << "\n";
  fprintf(stderr, "done %.0fs\n", omp_get_wtime() - t0);
  return 0;
}
