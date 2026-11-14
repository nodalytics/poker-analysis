#pragma once
#include <string>

namespace kestrel {

void buckets_init(const std::string& pack_dir);
int bucket_of(const int* cards, int n);

}
