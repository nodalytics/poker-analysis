// The blueprint's betting tree (heads-up, 200bb, blinds 50/100), shared by mccfr.cpp and mccfr_vec.cpp so both
// train the identical game. See mccfr.cpp's header for the action abstraction.
#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

static const int STACK = 20000, SB = 50, BB = 100;
static int BUCKETS[4] = {169, 200, 200, 200};  // flop..river set from the abstraction's meta.txt

struct Node {
  int8_t kind = 0;  // 0 decision, 1 fold (player `folder` folded), 2 showdown
  int8_t player = 0, street = 0, folder = -1;
  int c0 = 0, c1 = 0;  // chips put in by player 0 and 1
  std::vector<std::string> labels;
  std::vector<int> children;
  uint64_t offset = 0;  // into the regret / average arrays
};

static std::vector<Node> T;

struct Bet {
  int street, player, c0, c1, raises, acted;  // acted: decisions taken this street
  int last_raise;                             // size of the last raise increment
};

static int add_node(const Node& n) {
  T.push_back(n);
  return (int)T.size() - 1;
}

static int build(Bet b);

static int next_street(int c0, int c1, int street) {
  if (street == 3 || c0 >= STACK || c1 >= STACK) {
    Node n;
    n.kind = 2;
    n.street = street;
    n.c0 = c0, n.c1 = c1;
    return add_node(n);
  }
  return build({street + 1, 1, c0, c1, 0, 0, BB});
}

static int build(Bet b) {
  Node n;
  n.player = b.player;
  n.street = b.street;
  n.c0 = b.c0, n.c1 = b.c1;
  int id = add_node(n);
  int mine = b.player ? b.c1 : b.c0, theirs = b.player ? b.c0 : b.c1;
  int to_call = theirs - mine, pot = b.c0 + b.c1;
  std::vector<std::pair<std::string, int>> acts;  // label, raise-to (or -1 fold, -2 check/call)
  if (to_call > 0) acts.push_back({"fold", -1});
  acts.push_back({to_call > 0 ? "call" : "check", -2});
  int cap = b.street == 0 ? 4 : 3;
  if (b.raises < cap && theirs < STACK) {
    std::vector<std::pair<std::string, int>> sizes;
    if (b.street == 0) {
      if (b.raises == 0 && b.player == 0) {
        sizes.push_back({"open2", 200});
        sizes.push_back({"open2.5", 250});
        sizes.push_back({"open3", 300});
      } else if (b.raises == 0) {
        sizes.push_back({"raise3", 300});
        sizes.push_back({"raise4", 400});
      } else if (b.raises == 1) {
        sizes.push_back({"3bet3x", 3 * theirs});
        sizes.push_back({"3bet4x", 4 * theirs});
      } else if (b.raises == 2) {
        sizes.push_back({"4bet2.3x", (int)(2.3 * theirs)});
      }
    } else if (to_call == 0) {
      sizes.push_back({"bet0.33", mine + pot / 3});
      sizes.push_back({"bet0.66", mine + 2 * pot / 3});
      sizes.push_back({"bet1", mine + pot});
      if (b.raises == 0) sizes.push_back({"bet2", mine + 2 * pot});
    } else {
      sizes.push_back({"raise1", theirs + (pot + to_call)});
      if (b.raises == 1) sizes.push_back({"raise2.5x", theirs + (int)(1.5 * to_call) + to_call});
    }
    int min_to = theirs + std::max(b.last_raise, BB);
    for (auto& s : sizes) {
      int to = std::max(s.second, min_to);
      if (to >= STACK - STACK / 3 || to >= STACK) continue;  // close to all-in: use all-in instead
      acts.push_back({s.first, to});
    }
    acts.push_back({"allin", STACK});
  }
  std::vector<int> kids;
  std::vector<std::string> labels;
  for (auto& a : acts) {
    int child;
    if (a.second == -1) {
      Node f;
      f.kind = 1;
      f.folder = b.player;
      f.street = b.street;
      f.c0 = b.c0, f.c1 = b.c1;
      child = add_node(f);
    } else if (a.second == -2) {
      int c0 = b.c0, c1 = b.c1;
      if (b.player) c1 = theirs; else c0 = theirs;
      // the street closes on a call, or on a check by the second player to act (pre-flop: the big blind)
      bool closes = (to_call > 0 && !(b.street == 0 && b.raises == 0 && b.player == 0)) || (to_call == 0 && b.acted >= 1);
      if (b.street == 0 && b.raises == 0 && b.player == 0 && to_call > 0)
        child = build({0, 1, c0, c1, 0, b.acted + 1, b.last_raise});  // the small blind limps: the big blind acts
      else if (closes)
        child = next_street(c0, c1, b.street);
      else
        child = build({b.street, 1 - b.player, c0, c1, b.raises, b.acted + 1, b.last_raise});
    } else {
      int to = std::min(a.second, STACK);
      int c0 = b.c0, c1 = b.c1;
      if (b.player) c1 = to; else c0 = to;
      child = build({b.street, 1 - b.player, c0, c1, b.raises + 1, b.acted + 1, std::max(to - theirs, b.last_raise)});
    }
    kids.push_back(child);
    labels.push_back(a.first);
  }
  T[id].children = kids;
  T[id].labels = labels;
  return id;
}

