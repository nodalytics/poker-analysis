cmake_minimum_required(VERSION 3.24)
project(kestrel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
                     URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(json)

# core: the game, ICM and the networks (no I/O beyond reading model files)
add_library(kestrel_core core/src/table.cpp core/src/icm.cpp core/src/net.cpp)
target_include_directories(kestrel_core PUBLIC core/include)
target_link_libraries(kestrel_core PUBLIC nlohmann_json::nlohmann_json)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kestrel_core PRIVATE -Wall -Wextra)
endif()

# the self-test: reproduces the Python reference (scripts/advisor_vectors.py)
add_executable(kestrel_selftest tools/selftest.cpp)
target_link_libraries(kestrel_selftest PRIVATE kestrel_core)

enable_testing()
add_test(NAME selftest
         COMMAND kestrel_selftest ${CMAKE_CURRENT_SOURCE_DIR}/testdata/vectors_nm17.json
                 ${CMAKE_CURRENT_SOURCE_DIR}/../models/ninemax/nm17)
