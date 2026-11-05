#!/bin/sh
# install-app.sh URL [SHA256] [-- installer arguments]
# Downloads into /installers, checks the SHA-256 (or records it on first download as trust-on-first-use in
# /installers/<file>.sha256), scans with ClamAV, and only then runs the installer under Wine.
set -eu
url=$1; shift
want=${1:-}; [ "$#" -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift
f=/installers/$(basename "$url")
[ -s "$f" ] || curl -fsSL "$url" -o "$f"
got=$(sha256sum "$f" | cut -d' ' -f1)
if [ -n "$want" ] && [ "$want" != "-" ]; then
  [ "$got" = "$want" ] || { echo "SHA-256 mismatch: $got" >&2; exit 2; }
  echo "sha256 ok (published): $got"
elif [ -s "$f.sha256" ]; then
  [ "$got" = "$(cat "$f.sha256")" ] || { echo "SHA-256 changed since first download: $got" >&2; exit 2; }
  echo "sha256 ok (recorded earlier): $got"
else
  echo "$got" > "$f.sha256"; echo "sha256 recorded (no published checksum): $got"
fi
clamscan --no-summary --infected "$f" && echo "clamav: clean" || { echo "clamav flagged $f; not running it" >&2; exit 3; }
if [ "$#" -eq 0 ]; then  # no arguments given: pick the silent switch from the installer type
  if grep -aq "Nullsoft" "$f"; then set -- /S; kind=nsis
  elif grep -aq "Inno Setup" "$f"; then set -- /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-; kind=inno
  elif grep -aq "Squirrel" "$f"; then set -- --silent; kind=squirrel
  else kind=unknown; fi
  echo "installer type: $kind, arguments: $*"
fi
timeout 900 wine "$f" "$@"
