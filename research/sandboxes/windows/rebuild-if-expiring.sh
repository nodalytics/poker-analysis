#!/bin/sh
# Daily check, run by cron on the lab: rebuild the Windows 11 evaluation VM before its 90 days run out.
#   Install:  (crontab -l 2>/dev/null; echo '17 4 * * * sh $HOME/windows11/setup/rebuild-if-expiring.sh >> $HOME/windows11/rebuild.log 2>&1') | crontab -
#   Remove:   crontab -l | grep -v rebuild-if-expiring | crontab -
# Rebuilds after MAX_DAYS (default 80). Skips while ~/windows11/in-use exists (agent harnesses create it for
# the length of a run) unless that lock is more than a day old, and never runs twice at once.
set -eu
W=$HOME/windows11
MAX_DAYS=${MAX_DAYS:-80}
exec 9> "$W/.rebuild.lock"
flock -n 9 || { echo "$(date -Is) another rebuild is running"; exit 0; }
[ -s "$W/installed-at" ] || { echo "$(date -Is) no installed-at marker; run provision.sh first"; exit 0; }
age=$(( ( $(date +%s) - $(cat "$W/installed-at") ) / 86400 ))
if [ "$age" -lt "$MAX_DAYS" ]; then
  echo "$(date -Is) install is $age days old; rebuild at $MAX_DAYS"
  exit 0
fi
if [ -e "$W/in-use" ] && [ $(( $(date +%s) - $(stat -c %Y "$W/in-use") )) -lt 86400 ]; then
  echo "$(date -Is) install is $age days old but the VM is in use; trying again tomorrow"
  exit 0
fi
echo "$(date -Is) install is $age days old; rebuilding"
sh "$W/setup/provision.sh" --reinstall
echo "$(date -Is) rebuild finished"
