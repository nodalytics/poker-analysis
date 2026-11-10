#!/bin/sh
# Publish the 9-max watch page at https://poker.stbfarm.org on the lab. Run once, with sudo, from llm-research:
#   sudo sh web/deploy/install.sh
# It (1) installs the systemd service that runs scripts/ninemax_watch.py on 127.0.0.1:8790 as labuser, so the
# page survives reboots, and (2) installs the nginx vhost that proxies poker.stbfarm.org to it. It touches only its
# own files: /etc/systemd/system/poker-watch.service and /etc/nginx/sites-{available,enabled}/poker.stbfarm.org.
# Re-running it is safe. To remove: sudo sh web/deploy/install.sh --remove
set -eu
cd "$(dirname "$0")"
SITE=poker.stbfarm.org
if [ "$(id -u)" != 0 ]; then echo "run with sudo" >&2; exit 1; fi

if [ "${1:-}" = "--remove" ]; then
  systemctl disable --now poker-watch.service 2>/dev/null || true
  rm -f /etc/systemd/system/poker-watch.service /etc/nginx/sites-enabled/$SITE /etc/nginx/sites-available/$SITE
  systemctl daemon-reload
  nginx -t && systemctl reload nginx
  echo "removed"
  exit 0
fi

# 1. the app as a service (replaces a copy started by hand with nohup, if one is running)
pkill -u labuser -f "[n]inemax_watch.py" 2>/dev/null || true
install -m 644 poker-watch.service /etc/systemd/system/poker-watch.service
systemctl daemon-reload
systemctl enable --now poker-watch.service
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  curl -fs -o /dev/null http://127.0.0.1:8790/ && break
  sleep 2
done
curl -fs -o /dev/null http://127.0.0.1:8790/ || { echo "the app did not come up: journalctl -u poker-watch" >&2; exit 1; }

# 2. the vhost; nginx is only reloaded if the whole configuration still tests clean
install -m 644 nginx/$SITE.conf /etc/nginx/sites-available/$SITE
ln -sf /etc/nginx/sites-available/$SITE /etc/nginx/sites-enabled/$SITE
if nginx -t; then
  systemctl reload nginx
else
  rm -f /etc/nginx/sites-enabled/$SITE
  echo "nginx config test failed; the site was not enabled" >&2
  exit 1
fi
curl -fsk -o /dev/null --resolve $SITE:443:127.0.0.1 https://$SITE/ && echo "https://$SITE is up"
