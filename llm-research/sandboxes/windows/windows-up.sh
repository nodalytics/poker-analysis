#!/bin/sh
# From your machine: copy the Windows sandbox files to the lab and run provision.sh there.
#   sh windows-up.sh [--reinstall]
# Needs lab.pem; set STB_PEM if it is not in the parent folder. The VM's ports stay on the lab's loopback;
# for the desktop, tunnel: ssh -i "$STB_PEM" -L 8006:127.0.0.1:8006 labuser@LAB_HOST
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PEM=${STB_PEM:-$HERE/../../../../lab.pem}
LAB=labuser@LAB_HOST
lab() { ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$PEM" "$LAB" "$@"; }
lab 'mkdir -p ~/windows11/oem ~/windows11/setup/oem'
scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$PEM" \
  "$HERE/provision.sh" "$HERE/run-win11.sh" "$HERE/rebuild-if-expiring.sh" "$LAB:windows11/setup/"
scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$PEM" \
  "$HERE/oem/install.bat" "$HERE/oem/start-mcp.bat" "$LAB:windows11/setup/oem/"
lab "sh ~/windows11/setup/provision.sh ${1:-}"
