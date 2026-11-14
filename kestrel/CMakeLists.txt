cmake_minimum_required(VERSION 3.24)
project(kestrel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# where the models and the heads-up data pack live (models/ninemax/<name>/net.bin, blueprint/pack/); only the
# self-test needs them at build time
set(KESTREL_DATA "${CMAKE_CURRENT_SOURCE_DIR}/../llm-research" CACHE PATH "directory holding models/ and blueprint/pack/")

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
                     URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
# the same pinned commits as llm-research/blueprint/Makefile, so the buckets match the trainer's
FetchContent_Declare(phe GIT_REPOSITORY https://github.com/HenryRLee/PokerHandEvaluator.git
                     GIT_TAG 10be452e4c1ee40a6a56f06457f46bff27ca495a)
FetchContent_Declare(iso GIT_REPOSITORY https://github.com/kdub0/hand-isomorphism.git
                     GIT_TAG dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f)
FetchContent_MakeAvailable(json)
FetchContent_GetProperties(phe)
if(NOT phe_POPULATED)
  FetchContent_Populate(phe)
endif()
FetchContent_GetProperties(iso)
if(NOT iso_POPULATED)
  FetchContent_Populate(iso)
endif()

# third-party C: the 7-card evaluator and suit-isomorphic hand indexing (only the files the buckets need)
add_library(kestrel_cards STATIC
  ${phe_SOURCE_DIR}/cpp/src/evaluator7.c ${phe_SOURCE_DIR}/cpp/src/hash.c ${phe_SOURCE_DIR}/cpp/src/hashtable.c
  ${phe_SOURCE_DIR}/cpp/src/hashtable7.c ${phe_SOURCE_DIR}/cpp/src/dptables.c
  ${phe_SOURCE_DIR}/cpp/src/tables_bitwise.c
  ${iso_SOURCE_DIR}/src/hand_index.c ${iso_SOURCE_DIR}/src/deck.c)
set_target_properties(kestrel_cards PROPERTIES C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)
target_include_directories(kestrel_cards PUBLIC ${phe_SOURCE_DIR}/cpp/include ${iso_SOURCE_DIR}/src)

# core: the game, ICM, the 9-max networks and the heads-up blueprint
add_library(kestrel_core
  core/src/table.cpp core/src/icm.cpp core/src/net.cpp core/src/buckets.cpp core/src/headsup.cpp)
target_include_directories(kestrel_core PUBLIC core/include)
target_link_libraries(kestrel_core PUBLIC nlohmann_json::nlohmann_json PRIVATE kestrel_cards)

# connectors: platform notations (codecs) and, next, the transports and the session runner
add_library(kestrel_connectors connectors/src/codec_acpc.cpp)
target_include_directories(kestrel_connectors PUBLIC connectors/include)
target_link_libraries(kestrel_connectors PUBLIC kestrel_core)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kestrel_core PRIVATE -Wall -Wextra)
  target_compile_options(kestrel_connectors PRIVATE -Wall -Wextra)
endif()

# the self-test: reproduces the Python reference (llm-research/scripts/advisor_vectors.py, kestrel_hu_vectors.py)
add_executable(kestrel_selftest tools/selftest.cpp)
target_link_libraries(kestrel_selftest PRIVATE kestrel_core kestrel_connectors)

enable_testing()
add_test(NAME selftest
         COMMAND kestrel_selftest --vectors ${CMAKE_CURRENT_SOURCE_DIR}/testdata
                 --model ${KESTREL_DATA}/models/ninemax/nm17 --pack ${KESTREL_DATA}/blueprint/pack)
