#!/bin/sh
# Start the desktop. VNC_PASSFILE (a file with the password) protects VNC; without it VNC listens on the container's
# loopback only, reachable just by the bundled noVNC viewer (the human-only setup). APP_CMD, if set, is launched
# under Wine once the desktop is up.
set -eu
Xvfb :1 -screen 0 1280x800x24 -nolisten tcp &
sleep 1
mkdir -p /home/wine/.fluxbox && echo 'background: none' > /home/wine/.fluxbox/overlay  # no wallpaper warning
fluxbox >/dev/null 2>&1 &
if [ -n "${VNC_PASSFILE:-}" ] && [ -s "$VNC_PASSFILE" ]; then
  x11vnc -storepasswd "$(cat "$VNC_PASSFILE")" /home/wine/.vncpass >/dev/null 2>&1
  x11vnc -display :1 -rfbport 5900 -rfbauth /home/wine/.vncpass -forever -shared -quiet &
else
  x11vnc -display :1 -rfbport 5900 -localhost -nopw -forever -shared -quiet &
fi
websockify --web /usr/share/novnc 6080 localhost:5900 >/dev/null 2>&1 &
[ -d "$WINEPREFIX/drive_c" ] || wineboot --init >/dev/null 2>&1 || true
if [ -n "${APP_CMD:-}" ]; then
  sleep 2
  sh -c "$APP_CMD" >> /home/wine/app.log 2>&1 &
fi
wait
