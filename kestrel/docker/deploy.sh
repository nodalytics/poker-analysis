#!/bin/sh
# Deploy Kestrel on the lab: the image CI built and tested (ghcr.io, after a one-time `docker login ghcr.io` with a
# read:packages token), or with --local an image built here from the source beside this script.
#   sh kestrel/docker/deploy.sh [--local] [TAG]        TAG: main (default), a version (0.4.0) or sha-abc1234
set -eu
cd "$(dirname "$0")"
export KESTREL_PACK="${KESTREL_PACK:-$HOME/llm-research/blueprint/pack}"
if [ "${1:-}" = "--local" ]; then
  shift
  (cd ../.. && docker build -q -f kestrel/Dockerfile -t kestrel:local . && \
   docker build -q -f kestrel/docker/testplatform.Dockerfile -t kestrel-testplatform:local kestrel)
  export KESTREL_IMAGE=kestrel:local KESTREL_TESTPLATFORM_IMAGE=kestrel-testplatform:local
else
  export KESTREL_IMAGE="ghcr.io/nodalytics/kestrel:${1:-main}"
  export KESTREL_TESTPLATFORM_IMAGE="ghcr.io/nodalytics/kestrel-testplatform:${1:-main}"
  docker compose -f compose.yml pull
fi
docker compose -f compose.yml up -d
for i in $(seq 30); do
  docker compose -f compose.yml exec -T kestrel curl -fs http://127.0.0.1:8890/v1/health && echo && exit 0
  sleep 1
done
echo "Kestrel did not answer within 30 s: docker compose -f compose.yml logs kestrel" >&2
exit 1
