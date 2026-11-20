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
  core/src/texassolver.cpp core/src/luck.cpp core/src/models.cpp core/src/practice.cpp core/src/ninemax_spot.cpp core/src/ninemax_search.cpp
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
if(OpenMP_CXX_FOUND)  # search against the network alone, paired (a research check, not shipped)
  add_executable(kestrel_search_match tools/search_match.cpp)
  target_link_libraries(kestrel_search_match PRIVATE kestrel_core OpenMP::OpenMP_CXX)
endif()
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
  set(USE_TLS ON CACHE BOOL "" FORCE)        # wss:// connectors (Open Poker), through the same OpenSSL as https
  set(USE_OPEN_SSL ON CACHE BOOL "" FORCE)
  set(USE_ZLIB OFF CACHE BOOL "" FORCE)
  set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(ixwebsocket GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
                       GIT_TAG 2efe037c9cc96fd536774f17bdb5215161ee5087)
  FetchContent_MakeAvailable(httplib yaml-cpp ixwebsocket)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yaml-cpp PRIVATE -include cstdint)
  endif()

  # SQLite, built in (the amalgamation, 3.54.1: SHA3-256 checked against sqlite.org's download page), for kestrel.db
  FetchContent_Declare(sqlite URL https://www.sqlite.org/2026/sqlite-amalgamation-3540100.zip
                       URL_HASH SHA256=0a9c8e34352dad46006aa82836d5383bff9dc6bc9ce5a60553c05aa30e6ad1f9)
  FetchContent_MakeAvailable(sqlite)
  add_library(kestrel_sqlite STATIC ${sqlite_SOURCE_DIR}/sqlite3.c)
  target_include_directories(kestrel_sqlite PUBLIC ${sqlite_SOURCE_DIR})
  target_compile_definitions(kestrel_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION SQLITE_DQS=0
                             SQLITE_DEFAULT_WAL_SYNCHRONOUS=1)
  find_package(Threads REQUIRED)
  target_link_libraries(kestrel_sqlite PRIVATE Threads::Threads)
  add_library(kestrel_connectors connectors/src/connector.cpp connectors/src/store.cpp connectors/src/profile.cpp connectors/src/session.cpp connectors/src/opponents.cpp connectors/src/server.cpp
    connectors/src/gateway.cpp)
  target_link_libraries(kestrel_connectors PUBLIC kestrel_core PRIVATE httplib::httplib yaml-cpp::yaml-cpp ixwebsocket kestrel_sqlite)
  add_executable(kestreld tools/kestreld.cpp)
  target_link_libraries(kestreld PRIVATE kestrel_core kestrel_connectors httplib::httplib)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(kestreld PRIVATE OpenMP::OpenMP_CXX)
  endif()
  list(APPEND KESTREL_BINARIES kestreld)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kestrel_connectors PRIVATE -Wall -Wextra)
  endif()
  # the desktop app (Qt 6 Widgets): a client of kestreld's local API, built when Qt is found
  option(KESTREL_QT "build the desktop app when Qt 6 is found" ON)
  if(KESTREL_QT)
    find_package(Qt6 6.5 QUIET COMPONENTS Widgets Network)
    if(Qt6_FOUND)
      qt_add_executable(kestrel-app WIN32 MACOSX_BUNDLE app/main.cpp app/client.cpp app/window.cpp app/tableview.cpp app/study.cpp app/app.qrc)
      set_target_properties(kestrel-app PROPERTIES AUTOMOC ON AUTORCC ON OUTPUT_NAME Kestrel)
      if(UNIX AND NOT APPLE)  # the installed app loads the Qt bundled beside it (lib/), never the system's
        set_target_properties(kestrel-app PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
      endif()
      target_link_libraries(kestrel-app PRIVATE Qt6::Widgets Qt6::Network)
      message(STATUS "Kestrel desktop app: Qt ${Qt6_VERSION}")
    else()
      message(STATUS "Kestrel desktop app: skipped (no Qt 6 Widgets and Network)")
    endif()
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
add_test(NAME selftest_pko  # progressive knockouts: bounty features and equity, against nm30 (the PKO model) and its vectors
         COMMAND kestrel_selftest --vectors ${CMAKE_CURRENT_SOURCE_DIR}/testdata
                 --nine-max ${CMAKE_CURRENT_SOURCE_DIR}/testdata/vectors_nm30.json
                 --model ${KESTREL_DATA}/models/ninemax/nm30 --pack ${KESTREL_DATA}/blueprint/pack)
add_test(NAME selftest_cash6  # 6-max cash (no ICM, deep stacks, three empty seats), against nm37 and its vectors
         COMMAND kestrel_selftest --vectors ${CMAKE_CURRENT_SOURCE_DIR}/testdata
                 --nine-max ${CMAKE_CURRENT_SOURCE_DIR}/testdata/vectors_nm37.json
                 --model ${KESTREL_DATA}/models/ninemax/nm37 --pack ${KESTREL_DATA}/blueprint/pack)

include(GNUInstallDirs)
install(TARGETS ${KESTREL_BINARIES} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(TARGET kestrel-app)  # Qt's shared libraries: not linked statically; the release bundles them beside the app
  install(TARGETS kestrel-app RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} BUNDLE DESTINATION .)
  option(KESTREL_DEPLOY_QT "copy Qt's libraries and plugins into the install (release packages)" OFF)
  if(KESTREL_DEPLOY_QT)
    qt_generate_deploy_app_script(TARGET kestrel-app OUTPUT_SCRIPT kestrel_deploy NO_UNSUPPORTED_PLATFORM_ERROR)
    install(SCRIPT ${kestrel_deploy})
  endif()
endif()
install(FILES README.md docs/design.md DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel)
install(PROGRAMS packaging/install-desktop.sh DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel)
install(FILES connectors/slumbot.yaml DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/connectors)
install(DIRECTORY web/ DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/web)
install(DIRECTORY assets/ DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/assets)
install(DIRECTORY testdata DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel)
# nm25 is the default (ServeOptions::nine_max_default); nm17 the previous champion; nm30 bounty (PKO) tournaments; nm37
# 6-max cash (experimental: its exploiter still finds +5.6 bb/100)
foreach(m nm25 nm17 nm30 nm37)
  install(FILES ${KESTREL_DATA}/models/ninemax/${m}/net.bin ${KESTREL_DATA}/models/ninemax/${m}/net.json
          DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/models/ninemax/${m} OPTIONAL)
endforeach()
option(KESTREL_SHIP_PACK "include the heads-up data pack (about 80 MB) in the install, so heads-up tables work out of the box" OFF)
if(KESTREL_SHIP_PACK)
  foreach(f tree.tsv avg.u8 flop_buckets.u8 turn_buckets.u8 river_cuts.f32 meta.txt)
    install(FILES ${KESTREL_DATA}/blueprint/pack/${f} DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/pack)
  endforeach()
endif()
install(FILES ${phe_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses
        RENAME PokerHandEvaluator.txt OPTIONAL)
install(FILES ${iso_SOURCE_DIR}/LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses
        RENAME hand-isomorphism.txt OPTIONAL)
install(FILES ${json_SOURCE_DIR}/LICENSE.MIT DESTINATION ${CMAKE_INSTALL_DATADIR}/kestrel/licenses
        RENAME nlohmann-json.txt OPTIONAL)
