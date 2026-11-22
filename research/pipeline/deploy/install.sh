#!/bin/sh
# Install the research pipeline's service on the lab. Run once, with sudo, from ~/research:
#   sudo sh pipeline/deploy/install.sh            (sudo sh pipeline/deploy/install.sh --remove to take it out)
# It runs scripts/pipeline.py poll as the user who ran sudo, starts it now and at every boot. Re-running is safe.
set -eu
cd "$(dirname "$0")"
if [ "$(id -u)" != 0 ]; then echo "run with sudo" >&2; exit 1; fi
U=${SUDO_USER:?run with sudo from the lab user}
HOMEDIR=$(getent passwd "$U" | cut -d: -f6)
if [ "${1:-}" = "--remove" ]; then
  systemctl disable --now research-pipeline.service 2>/dev/null || true
  rm -f /etc/systemd/system/research-pipeline.service
  systemctl daemon-reload
  echo "removed"
  exit 0
fi
sed -e "s|LAB_USER|$U|" -e "s|LAB_HOME|$HOMEDIR|g" research-pipeline.service > /etc/systemd/system/research-pipeline.service
chmod 644 /etc/systemd/system/research-pipeline.service
systemctl daemon-reload
systemctl enable --now research-pipeline.service
sleep 3
systemctl --no-pager --lines 5 status research-pipeline.service || true
