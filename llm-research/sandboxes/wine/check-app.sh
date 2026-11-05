#!/bin/sh
# check-app.sh NAME MODE URL SHA256|- PATTERN [installer args...]   (on the lab, next to run-wine-app.sh)
# Starts the app's Wine container, installs it through install-app.sh (checksum + ClamAV), finds the installed
# executable by PATTERN, launches it, and writes ~/wine-apps/NAME/compat.json and NAME-wine.png. For real-money
# clients run MODE human: nothing but the viewer is published, and this script stops at the app's first window.
set -eu
name=$1; mode=$2; url=$3; sha=$4; pat=$5; shift 5
here=$(cd "$(dirname "$0")" && pwd); dir=$HOME/wine-apps/$name
docker rm -f "wine-$name" >/dev/null 2>&1 || true
sh "$here/run-wine-app.sh" "$name" "$mode" >/dev/null
sleep 8
t0=$(date +%s)
docker exec "wine-$name" install-app.sh "$url" "$sha" -- "$@" > "$dir/install.log" 2>&1; ic=$?
t1=$(date +%s)
exe=$(docker exec "wine-$name" sh -c "find /home/wine/prefix/drive_c -iname '*.exe' 2>/dev/null | grep -i '$pat' | grep -viE 'unins|setup|update|crash' | head -1")
started=no; title=""; mem=""
if [ -n "$exe" ]; then
  docker exec -d "wine-$name" sh -c "wine \"$exe\" >> /home/wine/app.log 2>&1"
  sleep 45
  title=$(docker exec "wine-$name" sh -c "xdotool search --onlyvisible --name . getwindowname %@ 2>/dev/null | grep -v '^$' | head -5 | tr '\n' '|'")
  [ -n "$title" ] && started=yes
  docker exec "wine-$name" sh -c "import -window root /tmp/shot.png" && docker cp "wine-$name:/tmp/shot.png" "$dir/$name-wine.png"
  mem=$(docker stats --no-stream --format '{{.MemUsage}}' "wine-$name")
fi
python3 - "$dir/compat.json" <<PY
import json, sys
json.dump({"name": "$name", "runtime": "wine 10.0 (Debian trixie)", "mode": "$mode",
           "install_exit": $ic, "install_seconds": $((t1 - t0)), "executable": "$exe", "window_appeared": "$started" == "yes",
           "windows": [x for x in "$title".split("|") if x], "container_memory": "$mem",
           "ui_tree": "none: Windows UI Automation does not exist under Wine; screenshots and VNC input only"},
          open(sys.argv[1], "w"), indent=1)
PY
cat "$dir/compat.json"; tail -3 "$dir/install.log"
