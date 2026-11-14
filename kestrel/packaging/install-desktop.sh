#!/bin/sh
# Add Kestrel to the desktop's application menu (Linux): run once from the unpacked release folder.
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
mkdir -p "$apps" "$icons"
cp "$here/share/kestrel/assets/logo.svg" "$icons/kestrel.svg"
cat > "$apps/kestrel.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Kestrel
Comment=Poker advisor and bot: live tables in your browser
Exec="$here/bin/kestreld"
Icon=kestrel
Terminal=false
Categories=Game;CardGame;
DESK
command -v update-desktop-database >/dev/null && update-desktop-database "$apps" || true
echo "Kestrel is in your applications menu."
