#!/bin/sh
# Start the desktop. APP_CMD, if set, is launched under Wine once the desktop is up.
set -eu
Xvfb :1 -screen 0 1280x800x24 -nolisten tcp &
sleep 1
mkdir -p /home/wine/.fluxbox && echo 'background: none' > /home/wine/.fluxbox/overlay  # no wallpaper warning
fluxbox >/dev/null 2>&1 &
# A VNC password is required in every mode. VNC_PUBLIC=1 (agent mode) listens on the container's network so the lab
# can publish it; otherwise VNC stays on the container's loopback and only the bundled noVNC viewer reaches it.
[ -n "${VNC_PASSFILE:-}" ] && [ -s "$VNC_PASSFILE" ] || { echo "VNC_PASSFILE with a password is required" >&2; exit 1; }
x11vnc -storepasswd "$(cat "$VNC_PASSFILE")" /home/wine/.vncpass >/dev/null 2>&1
if [ "${VNC_PUBLIC:-0}" = 1 ]; then
  x11vnc -display :1 -rfbport 5900 -rfbauth /home/wine/.vncpass -forever -shared -quiet &
else
  x11vnc -display :1 -rfbport 5900 -localhost -rfbauth /home/wine/.vncpass -forever -shared -quiet &
fi
websockify --web /usr/share/novnc 6080 localhost:5900 >/dev/null 2>&1 &
[ -d "$WINEPREFIX/drive_c" ] || wineboot --init >/dev/null 2>&1 || true
if [ -n "${APP_CMD:-}" ]; then
  sleep 2
  sh -c "$APP_CMD" >> /home/wine/app.log 2>&1 &
fi
wait
