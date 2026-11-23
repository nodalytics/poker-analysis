#!/usr/bin/env bash
# Probe a local kestreld for the server hardening checks: malformed input is a 400 (not a 500), and the shutdown route
# refuses a request that came through a public host name, even from this computer (a reverse proxy on the same machine).
# Starts its own server on a free port with a scratch config, sessions folder and no database, and stops it by its PID.
#   kestrel/tools/server_check.sh path/to/kestreld          (exits 1 on a failure)
set -u
KD=${1:?usage: server_check.sh KESTRELD}
T=$(mktemp -d)
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$KD" serve --port "$PORT" --lan on --public-host kestrel.test --config "$T/cfg" --out "$T/out" --db off \
    --ws-port 0 --tcp-port 0 --pack "$T/nopack" > "$T/server.log" 2>&1 &
PID=$!
cleanup() {
    kill "$PID" 2> /dev/null && wait "$PID" 2> /dev/null
    rm -rf "$T"
}
trap cleanup EXIT
B="http://127.0.0.1:$PORT"
for _ in $(seq 1 100); do
    curl -s -o /dev/null "$B/v1/health" && break
    sleep 0.1
done
TOKEN=$(cat "$T/cfg/token" 2> /dev/null)
fails=0
expect() {  # expect WANT DESCRIPTION curl-args...
    local want=$1 what=$2
    shift 2
    local got
    got=$(curl -s -o "$T/body" -w '%{http_code}' --max-time 3 "$@")
    if [ "$got" = "$want" ]; then
        printf '  ok   %s: %s\n' "$got" "$what"
    else
        printf '  FAIL %s (want %s): %s  %s\n' "$got" "$want" "$what" "$(head -c 160 "$T/body")"
        fails=$((fails + 1))
    fi
}
J=(-H 'Content-Type: application/json')
echo "kestreld on port $PORT (--lan on --public-host kestrel.test):"
expect 200 "GET /v1/health" "$B/v1/health"
expect 400 "GET /v1/sessions/<20 digits> (int overflow)" "$B/v1/sessions/99999999999999999999"
expect 400 "DELETE /v1/sessions/<20 digits>" -X DELETE "$B/v1/sessions/99999999999999999999"
expect 400 "POST /v1/practice/<20 digits>/act" "${J[@]}" -d '{"slot":1}' "$B/v1/practice/99999999999999999999/act"
expect 400 "GET /v1/events, Last-Event-ID: abc" -H 'Last-Event-ID: abc' "$B/v1/events"
expect 400 "GET /v1/events?from=abc" "$B/v1/events?from=abc"
expect 400 'POST /v1/sessions {"connector":5}' "${J[@]}" -d '{"connector":5}' "$B/v1/sessions"
expect 400 'POST /v1/sessions {"connector":"x","hands":"lots"}' "${J[@]}" -d '{"connector":"x","hands":"lots"}' "$B/v1/sessions"
expect 400 'POST /v1/sessions {"connector":"x","model":7}' "${J[@]}" -d '{"connector":"x","model":7}' "$B/v1/sessions"
expect 400 'POST /v1/practice {"model":[]}' "${J[@]}" -d '{"model":[]}' "$B/v1/practice"
expect 400 'POST /v1/decide {"game":5}' "${J[@]}" -d '{"game":5}' "$B/v1/decide"
expect 403 "public host, wrong token" -H 'Host: kestrel.test' -H 'X-Kestrel-Token: 0000' "$B/v1/health"
expect 200 "public host, the token" -H 'Host: kestrel.test' -H "X-Kestrel-Token: $TOKEN" "$B/v1/health"
expect 403 "POST /v1/admin/shutdown through the public host (a proxy on this computer)" "${J[@]}" \
    -H 'Host: kestrel.test' -H "X-Kestrel-Token: $TOKEN" -X POST -d '{}' "$B/v1/admin/shutdown"
sleep 0.5  # the shutdown route stops the server 200 ms after replying
expect 200 "still serving after that" "$B/v1/health"
expect 200 "POST /v1/admin/shutdown from this computer" "${J[@]}" -X POST -d '{}' "$B/v1/admin/shutdown"
for _ in $(seq 1 50); do
    kill -0 "$PID" 2> /dev/null || break
    sleep 0.1
done
if kill -0 "$PID" 2> /dev/null; then
    echo "  FAIL the server did not stop after the local shutdown"
    fails=$((fails + 1))
else
    echo "  ok   stopped"
fi
[ "$fails" = 0 ] && echo PASS || echo "FAIL ($fails)"
[ "$fails" = 0 ]
