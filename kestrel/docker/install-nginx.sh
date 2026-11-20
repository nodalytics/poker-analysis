#!/bin/sh
# Publish Kestrel at https://kestrel.stbfarm.org through the lab's nginx. Run once, with sudo, from the repo root:
#   sudo sh kestrel/docker/install-nginx.sh
# Kestrel itself runs in Docker (kestrel/docker/deploy.sh) on 127.0.0.1:8890; this installs only the vhost:
# /etc/nginx/sites-{available,enabled}/kestrel.stbfarm.org. Re-running it is safe. To remove: --remove
set -eu
cd "$(dirname "$0")"
SITE=kestrel.stbfarm.org
if [ "$(id -u)" != 0 ]; then echo "run with sudo" >&2; exit 1; fi
if [ "${1:-}" = "--remove" ]; then
  rm -f /etc/nginx/sites-enabled/$SITE /etc/nginx/sites-available/$SITE
  nginx -t && systemctl reload nginx
  echo "removed"
  exit 0
fi
curl -fs -o /dev/null -H "Host: localhost:8890" http://127.0.0.1:8890/v1/health 2>/dev/null \
  || curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8890/v1/health | grep -q 403 \
  || { echo "Kestrel is not answering on 127.0.0.1:8890: sh kestrel/docker/deploy.sh first" >&2; exit 1; }
install -m 644 nginx/$SITE.conf /etc/nginx/sites-available/$SITE
ln -sf /etc/nginx/sites-available/$SITE /etc/nginx/sites-enabled/$SITE
if nginx -t; then
  systemctl reload nginx
else
  rm -f /etc/nginx/sites-enabled/$SITE
  echo "nginx config test failed; the site was not enabled" >&2
  exit 1
fi
# Kestrel answers a request without the token with 403 "needs the access link": that is the site working
# a reload is asynchronous: until the new workers run, the request lands on another vhost (a 502 from it, seen once)
for _ in 1 2 3 4 5 6 7 8 9 10; do
  code=$(curl -sk -o /dev/null -w "%{http_code}" --resolve $SITE:443:127.0.0.1 https://$SITE/v1/health)
  [ "$code" = 403 ] || [ "$code" = 200 ] && break
  sleep 1
done
echo "https://$SITE answers $code (403 without the token is expected; open the access link once per device)"
