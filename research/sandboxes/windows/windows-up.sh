#!/bin/sh
# From your machine: copy the Windows sandbox files to the lab and run provision.sh there.
#   sh windows-up.sh [--reinstall]
# Needs LAB (user@host) and LAB_KEY (the key file) set (`. .secrets/lab.env` from the repo root). The VM's ports stay on the lab's
# loopback; for the desktop, tunnel: ssh -i "$LAB_KEY" -L 8006:127.0.0.1:8006 "$LAB"
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PEM=${LAB_KEY:?set LAB_KEY to the lab key file}
LAB=${LAB:?set LAB=user@host for the lab}
: "$HERE"
lab() { ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$PEM" "$LAB" "$@"; }
lab 'mkdir -p ~/windows11/oem ~/windows11/setup/oem'
scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$PEM" \
  "$HERE/provision.sh" "$HERE/run-win11.sh" "$HERE/rebuild-if-expiring.sh" "$LAB:windows11/setup/"
scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$PEM" \
  "$HERE/oem/install.bat" "$HERE/oem/start-mcp.bat" "$LAB:windows11/setup/oem/"
lab "sh ~/windows11/setup/provision.sh ${1:-}"
