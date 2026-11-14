cmake_minimum_required(VERSION 3.24)
project(kestrel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(KESTREL_DATA "${CMAKE_CURRENT_SOURCE_DIR}/../llm-research" CACHE PATH "directory holding models/ and blueprint/pack/")
option(KESTREL_CORE_ONLY "the core and its checks only: no connectors, server or daemon (no OpenSSL needed)" OFF)
option(KESTREL_STATIC "link the C and C++ runtimes statically (portable release binaries)" OFF)
option(KESTREL_NATIVE "compile for this machine's CPU (faster re-solves; the binary may not run elsewhere)" OFF)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
                     URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_Declare(phe GIT_REPOSITORY https://github.com/HenryRLee/PokerHandEvaluator.git
                     GIT_TAG 10be452e4c1ee40a6a56f06457f46bff27ca495a SOURCE_SUBDIR no-cmake)
FetchContent_Declare(iso GIT_REPOSITORY https://github.com/kdub0/hand-isomorphism.git
                     GIT_TAG dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f SOURCE_SUBDIR no-cmake)
FetchContent_MakeAvailable(json phe iso)

add_library(kestrel_cards STATIC
  ${phe_SOURCE_DIR}/cpp/src/evaluator5.c ${phe_SOURCE_DIR}/cpp/src/evaluator6.c ${phe_SOURCE_DIR}/cpp/src/evaluator7.c
  ${phe_SOURCE_DIR}/cpp/src/hash.c ${phe_SOURCE_DIR}/cpp/src/hashtable.c ${phe_SOURCE_DIR}/cpp/src/hashtable5.c
  ${phe_SOURCE_DIR}/cpp/src/hashtable6.c ${phe_SOURCE_DIR}/cpp/src/hashtable7.c ${phe_SOURCE_DIR}/cpp/src/dptables.c
  ${phe_SOURCE_DIR}/cpp/src/tables_bitwise.c
  ${iso_SOURCE_DIR}/src/hand_index.c ${iso_SOURCE_DIR}/src/deck.c)
set_target_properties(kestrel_cards PROPERTIES C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)
target_include_directories(kestrel_cards PUBLIC ${phe_SOURCE_DIR}/cpp/include ${iso_SOURCE_DIR}/src)

add_library(kestrel_core
  core/src/table.cpp core/src/icm.cpp core/src/net.cpp core/src/buckets.cpp core/src/headsup.cpp core/src/resolve.cpp
  core/src/texassolver.cpp
  connectors/src/codec_acpc.cpp)
target_include_directories(kestrel_core PUBLIC core/include connectors/include)
target_link_libraries(kestrel_core PUBLIC nlohmann_json::nlohmann_json PRIVATE kestrel_cards)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(kestrel_core PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(kestrel_selftest tools/selftest.cpp)
target_link_libraries(kestrel_selftest PRIVATE kestrel_core)
add_executable(kestrel_resolve_check tools/resolve_check.cpp)
target_link_libraries(kestrel_resolve_check PRIVATE kestrel_core)
set(KESTREL_BINARIES kestrel_selftest)

if(NOT KESTREL_CORE_ONLY)
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
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  FetchContent_Declare(yaml-cpp GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git GIT_TAG 0.8.0)
  FetchContent_MakeAvailable(httplib yaml-cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yaml-cpp PRIVATE -include cstdint)
  endif()

  add_library(kestrel_connectors connectors/src/connector.cpp connectors/src/session.cpp connectors/src/server.cpp)
  target_link_libraries(kestrel_connectors PUBLIC kestrel_core PRIVATE httplib::httplib yaml-cpp::yaml-cpp)
  add_executable(kestreld tools/kestreld.cpp)
  target_link_libraries(kestreld PRIVATE kestrel_core kestrel_connectors)
  list(APPEND KESTREL_BINARIES kestreld)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kestrel_connectors PRIVATE -Wall -Wextra)
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kestrel_core PRIVATE -Wall -Wextra)
endif()
if(KESTREL_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kestrel_core PRIVATE -march=native)
endif()
if(KESTREL_STATIC AND (MINGW OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  foreach(t ${KESTREL_BINARIES})
    target_link_options(${t} PRIVATE -static-libgcc -static-libstdc++ $<$<BOOL:${MINGW}>:-static>)
  endforeach()
endif()

enable_testing()
add_test(NAME selftest
         COMMAND kestrel_selftest --vectors ${CMAKE_CURRENT_SOURCE_DIR}/testdata
                 --model ${KESTREL_DATA}/models/ninemax/nm17 --pack ${KESTREL_DATA}/blueprint/pack)

include(GNUInstallDirs)
install(TARGETS ${KESTREL_BINARIES} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES README.md docs/design.md DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel)
install(FILES connectors/slumbot.yaml DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/connectors)
install(DIRECTORY web/ DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/web)
install(DIRECTORY assets/ DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/assets)
install(DIRECTORY testdata DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel)
install(FILES ${KESTREL_DATA}/models/ninemax/nm17/net.bin ${KESTREL_DATA}/models/ninemax/nm17/net.json
        DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/models/ninemax/nm17 OPTIONAL)
install(FILES ${phe_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses
        RENAME PokerHandEvaluator.txt OPTIONAL)
install(FILES ${iso_SOURCE_DIR}/LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses
        RENAME hand-isomorphism.txt OPTIONAL)
install(FILES ${json_SOURCE_DIR}/LICENSE.MIT DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses
        RENAME nlohmann-json.txt OPTIONAL)
