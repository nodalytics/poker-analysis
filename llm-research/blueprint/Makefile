# Heads-up no-limit blueprint (Libratus/Pluribus recipe): card abstraction, sampled CFR, strategy export.
# Third-party code is fetched at pinned commits into deps/ (not committed):
#   PokerHandEvaluator (Henry Lee, Apache-2.0)         7-card hand evaluation
#   hand-isomorphism    (Kevin Waugh, BSD-style licence) suit-isomorphic hand indexing
# Build on the lab: make -j
PHE_COMMIT = 10be452e4c1ee40a6a56f06457f46bff27ca495a
ISO_COMMIT = dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f
CXX = g++
CC = gcc
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17 -Wall -Ideps/phe/cpp/include -Ideps/iso/src
CFLAGS = -O3 -march=native -std=c99 -Ideps/phe/cpp/include -Ideps/iso/src
# only the 7-card evaluator and its tables (the 8- and 9-card variants do not build on their own)
PHE_SRC = $(addprefix deps/phe/cpp/src/,evaluator7.c hash.c hashtable.c hashtable7.c dptables.c tables_bitwise.c)
ISO_SRC = deps/iso/src/hand_index.c deps/iso/src/deck.c

all: abstraction check mccfr preclass libbp.so

deps/phe deps/iso:
	mkdir -p deps
	git clone -q https://github.com/HenryRLee/PokerHandEvaluator.git deps/phe && git -C deps/phe checkout -q $(PHE_COMMIT)
	git clone -q https://github.com/kdub0/hand-isomorphism.git deps/iso && git -C deps/iso checkout -q $(ISO_COMMIT)

deps/libdeps.a: deps/phe deps/iso
	cd deps && for f in $(patsubst deps/%,%,$(PHE_SRC) $(ISO_SRC)); do $(CC) $(patsubst -Ideps/%,-I%,$(CFLAGS)) -c $$f -o $$(echo $$f | tr / _).o || exit 1; done && ar rcs libdeps.a *.o

abstraction: abstraction.cpp common.hpp deps/libdeps.a
	$(CXX) $(CXXFLAGS) abstraction.cpp deps/libdeps.a -o $@

mccfr: mccfr.cpp common.hpp deps/libdeps.a
	$(CXX) $(CXXFLAGS) mccfr.cpp deps/libdeps.a -o $@

preclass: preclass.cpp common.hpp deps/libdeps.a
	$(CXX) $(CXXFLAGS) preclass.cpp deps/libdeps.a -o $@

# position-independent copy of the dependencies for the Python library (separate, so a running mccfr is untouched)
deps/libdeps_pic.a: deps/phe deps/iso
	mkdir -p deps/pic && cd deps && for f in $(patsubst deps/%,%,$(PHE_SRC) $(ISO_SRC)); do $(CC) -fPIC $(patsubst -Ideps/%,-I%,$(CFLAGS)) -c $$f -o pic/$$(echo $$f | tr / _).o || exit 1; done && ar rcs libdeps_pic.a pic/*.o

libbp.so: bp.cpp common.hpp deps/libdeps_pic.a
	$(CXX) $(CXXFLAGS) -fPIC -shared bp.cpp deps/libdeps_pic.a -o $@

check: check.cpp common.hpp deps/libdeps.a
	$(CXX) $(CXXFLAGS) check.cpp deps/libdeps.a -o $@

clean:
	rm -f abstraction mccfr deps/*.o deps/libdeps.a
