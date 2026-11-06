#!/bin/sh
# Time the original and optimised TexasSolver builds on identical spots, 1 and 16 threads.
# Usage (on the lab): sh scripts/solver_bench.sh > solver_bench.log
# Each line: build, street, threads, seconds, final exploitability, and the root strategy of the first hand, which
# must match between builds (same answer, only faster).
R="AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AK,AQ,AJ,AT,A9,A8,A7,A6,A5,A4,A3,A2,KQ,KJ,KT,K9,K8,QJ,QT,Q9,JT,J9,T9,T8,98,97,87,76,65,54"
RES=~/poker-analysis/tools/TexasSolver/resources
for bin in ~/poker-analysis/tools/TexasSolver/build/console_solver ~/solver-fast/build/console_solver; do
  for board in "Qh,9h,3d,2s" "Qs,Jh,2d"; do
    for th in 1 16; do
      T=$(mktemp -d)
      st=$([ ${#board} -gt 9 ] && echo turn || echo flop)
      {
        echo "set_pot 10"; echo "set_effective_stack 95"; echo "set_board $board"
        echo "set_range_oop $R"; echo "set_range_ip $R"
        for p in oop ip; do for s in flop turn river; do
          echo "set_bet_sizes $p,$s,bet,66"; echo "set_bet_sizes $p,$s,allin"
        done; done
        echo "set_allin_threshold 0.67"; echo "build_tree"; echo "set_thread_num $th"
        echo "set_accuracy 2.0"; echo "set_max_iteration 80"; echo "set_print_interval 1000"
        echo "set_use_isomorphism 1"; echo "start_solve"; echo "set_dump_rounds 1"; echo "dump_result $T/o.json"
      } > "$T/in.txt"
      s=$(date +%s.%N)
      (cd "$T" && timeout 1800 "$bin" --input_file in.txt --resource_dir "$RES" > log 2>&1)
      e=$(date +%s.%N)
      root=$(python3 -c "import json;d=json.load(open('$T/o.json'));s=d['strategy']['strategy'];k=sorted(s)[0];print(k,[round(x,2) for x in s[k]])" 2>/dev/null || echo "no result")
      echo "$(basename "$(dirname "$(dirname "$bin")")")  $st  threads $th: $(echo "$e - $s" | bc | cut -c1-7) s  $(grep -o 'Total exploitability [0-9.]*' "$T/log" | tail -1)  root $root"
      rm -rf "$T"
    done
  done
done
echo ALLDONE
