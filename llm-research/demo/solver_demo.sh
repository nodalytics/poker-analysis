#!/bin/bash
# Short live proof that TexasSolver runs and gives correct answers (~30 s). Run on the lab; record it with
#   script -q -T demo.timing -c "bash solver_demo.sh" demo.typescript
# and turn the recording into a video with scripts/render_terminal_video.py.
S=${TEXASSOLVER:-$HOME/poker-analysis/tools/TexasSolver}
say() { printf '\n\033[1;33m%s\033[0m\n' "$*"; sleep 1; }
say "TexasSolver (github.com/bupticybee/TexasSolver), built on the lab:"
(cd "$S" && git log -1 --format='  commit %h  %s' && ls -l build/console_solver | awk '{print "  " $NF, $5, "bytes"}')
say "Textbook river game: AA (nuts) or 76 (air) vs KQ (bluff-catcher), pot 100, all-in 100"
echo "  Theory: always bet AA; bluff 9.4% of 76; KQ calls 50%."
T=$(mktemp -d)
printf '%s\n' "set_pot 100" "set_effective_stack 100" "set_board As,Kd,8h,5c,2s" "set_range_oop AA,76" "set_range_ip KQ" \
  "set_bet_sizes oop,river,allin" "set_bet_sizes ip,river,allin" "set_allin_threshold 0.67" "build_tree" \
  "set_thread_num 4" "set_accuracy 0.05" "set_max_iteration 2000" "set_print_interval 400" "set_use_isomorphism 0" \
  "start_solve" "set_dump_rounds 2" "dump_result $T/out.json" > "$T/in.txt"
say "Solving..."
(cd "$T" && "$S/build/console_solver" --input_file in.txt --resource_dir "$S/resources" 2>&1 | grep -E 'exploitab|time used' | sed 's/^/  /')
python3 - "$T/out.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); acts = t["actions"]; st = t["strategy"]["strategy"]
mean = lambda c, j: sum(v[j] for k, v in st.items() if {k[0], k[2]} == set(c)) / sum(1 for k in st if {k[0], k[2]} == set(c))
b = [i for i, a in enumerate(acts) if a != "CHECK"][0]
node = t["childrens"][acts[b]]; cs = node["strategy"]["strategy"]
call = sum(v[node["actions"].index("CALL")] for v in cs.values()) / len(cs)
print("\n\033[1;33mSolver vs theory:\033[0m")
print(f"  bets AA         {mean('AA', b):6.1%}   theory 100.0%")
print(f"  bluffs 76       {mean('76', b):6.1%}   theory   9.4%")
print(f"  KQ calls        {call:6.1%}   theory  50.0%")
PY
rm -rf "$T"
sleep 3
