// Card buckets of the heads-up blueprint's abstraction (the data pack's tables). Cards are rank*4 + suit.
#pragma once
#include <string>

namespace kestrel {

void buckets_init(const std::string& pack_dir);  // once; throws if the tables are missing
// hole[2] then the board (0, 3, 4 or 5 cards): the pre-flop class 0..168 with no board, else the street's bucket;
// -1 on bad input
int bucket_of(const int* cards, int n);

}  // namespace kestrel
