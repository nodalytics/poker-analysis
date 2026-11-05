# Wine desktop containers

Windows apps under Wine, one Docker container per app, on the lab. Built from `Dockerfile`
(`wine-desktop:trixie-20261005`: Debian trixie, Wine 10.0, Xvfb, fluxbox, x11vnc, noVNC, ClamAV).

| File | Does |
|---|---|
| `run-wine-app.sh NAME agent\|human` | Starts `wine-NAME`. **agent**: VNC on the lab's `127.0.0.1:5901` (password in `~/wine-apps/NAME/vnc.pass`) and noVNC on `127.0.0.1:6080`. **human**: only noVNC on `127.0.0.1:6081`; VNC stays inside the container, so no agent can drive it |
| `install-app.sh URL SHA256\|- [-- args]` | Inside the container: download, check the published SHA-256 (or record it on first download), scan with ClamAV, then run the installer; with no args it detects NSIS / Inno / Squirrel and installs silently |
| `check-app.sh NAME MODE URL SHA PATTERN` | Install, find the executable, launch it, and write `~/wine-apps/NAME/compat.json` and a screenshot |

Rules:

- **Real-money clients run in human mode only** (ACR): installed and launched for a person to use through the viewer.
  No agent, OCR or solver is connected to them, and checks stop at the first window. Their terms forbid bots and
  real-time assistance, and it would cheat the other players.
- Agent targets are apps that allow it: PokerTH (offline against its own bots), our own tables.
- Under Wine there is **no Windows UI Automation**: agents get screenshots and VNC input, not an element tree. The
  Windows 11 VM (`../windows`) is the place for UI-tree control.
