#!/bin/sh
# run-wine-app.sh NAME MODE [APP_CMD]   (on the lab)
#   MODE agent: publishes VNC on 127.0.0.1:5901 (password in ~/wine-apps/NAME/vnc.pass) and noVNC on 127.0.0.1:6080
#   MODE human: publishes only noVNC on 127.0.0.1:6081; VNC stays inside the container, so no agent can drive it
set -eu
name=$1; mode=$2; cmd=${3:-}
dir=$HOME/wine-apps/$name; mkdir -p "$dir/prefix" "$dir/installers"
args="--name wine-$name -d --memory 4g --cpus 2 --shm-size 512m -v $dir/prefix:/home/wine/prefix -v $dir/installers:/installers"
if [ "$mode" = agent ]; then
  umask 077; [ -s "$dir/vnc.pass" ] || head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n' | head -c 8 > "$dir/vnc.pass"; umask 022
  args="$args -v $dir/vnc.pass:/run/vnc.pass:ro -e VNC_PASSFILE=/run/vnc.pass -p 127.0.0.1:5901:5900 -p 127.0.0.1:6080:6080"
else
  args="$args -p 127.0.0.1:6081:6080"
fi
docker run $args -e APP_CMD="$cmd" wine-desktop:trixie-20261005
