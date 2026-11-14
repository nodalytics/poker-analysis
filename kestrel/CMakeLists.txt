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
                     GIT_TAG 10be452e4c1ee40a6a56f06457f46bff27ca495a SOURCE_SUBDIR no-cmake)
FetchContent_Declare(iso GIT_REPOSITORY https://github.com/kdub0/hand-isomorphism.git
                     GIT_TAG dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f SOURCE_SUBDIR no-cmake)
# (SOURCE_SUBDIR no-cmake: fetch the sources only; the files we need are compiled below)
FetchContent_MakeAvailable(json phe iso)

# connectors: HTTP(S) client and server (cpp-httplib, with OpenSSL) and YAML connector files (yaml-cpp)
find_package(OpenSSL REQUIRED)
set(HTTPLIB_REQUIRE_OPENSSL ON CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git GIT_TAG v0.18.3)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)  # yaml-cpp 0.8.0 asks for an older CMake than CMake 4 accepts
FetchContent_Declare(yaml-cpp GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git GIT_TAG 0.8.0)
FetchContent_MakeAvailable(httplib yaml-cpp)

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
add_library(kestrel_connectors connectors/src/codec_acpc.cpp connectors/src/connector.cpp connectors/src/session.cpp)
target_include_directories(kestrel_connectors PUBLIC connectors/include)
target_link_libraries(kestrel_connectors PUBLIC kestrel_core PRIVATE httplib::httplib yaml-cpp::yaml-cpp)

# kestreld: the daemon (sessions now; the local API and the web page next)
add_executable(kestreld tools/kestreld.cpp)
target_link_libraries(kestreld PRIVATE kestrel_core kestrel_connectors)

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

# ---- install / package (the release archives: cmake --install build --prefix <dir>) ----
include(GNUInstallDirs)
install(TARGETS kestrel_selftest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
foreach(t kestreld)  # the daemon, once it exists
  if(TARGET ${t})
    install(TARGETS ${t} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endforeach()
install(FILES README.md docs/design.md DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel)
install(FILES connectors/slumbot.yaml DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/connectors)
install(DIRECTORY testdata DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel)
# the tournament network the router uses (4.5 MB); the heads-up pack (80 MB) ships as its own release asset
install(FILES ${KESTREL_DATA}/models/ninemax/nm17/net.bin ${KESTREL_DATA}/models/ninemax/nm17/net.json
        DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/models/ninemax/nm17 OPTIONAL)
# third-party licences
install(FILES ${phe_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses RENAME PokerHandEvaluator.txt OPTIONAL)
install(FILES ${iso_SOURCE_DIR}/LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses RENAME hand-isomorphism.txt OPTIONAL)
install(FILES ${json_SOURCE_DIR}/LICENSE.MIT DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses RENAME nlohmann-json.txt OPTIONAL)

option(KESTREL_STATIC "link the C and C++ runtimes statically (portable release binaries)" OFF)
if(KESTREL_STATIC AND (MINGW OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  foreach(t kestrel_selftest kestreld)
    target_link_options(${t} PRIVATE -static-libgcc -static-libstdc++ $<$<BOOL:${MINGW}>:-static>)
  endforeach()
endif()
