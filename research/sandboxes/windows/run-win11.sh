#!/bin/sh
# Windows 11 Enterprise Evaluation (90 days, free from Microsoft) in Docker via KVM, on the lab.
# Desktop: http://127.0.0.1:8006 (web) or RDP 127.0.0.1:3389, both bound to the lab's loopback only.
# Reach them from another machine with:
#   ssh -i "$LAB_KEY" -L 8006:127.0.0.1:8006 -L 3389:127.0.0.1:3389 "$LAB"
#
# The login lives in ~/windows11/credentials.env on the lab (USERNAME=..., PASSWORD=..., mode 600).
# Never commit that file. Create it once with:
#   umask 077; printf "USERNAME=tester\nPASSWORD=%s\n" "$(head -c 18 /dev/urandom | base64 | tr -d '/+=' | head -c 20)" > ~/windows11/credentials.env
#
# RAM_CHECK=N: the image sizes RAM from MemFree, which page cache keeps low on the lab,
# so without it Windows got 8.7GB instead of 16GB on 2026-11-05 while 107GB was available.
#
# ~/windows11/oem holds oem/install.bat (CRLF line endings). It runs once, at the end of a fresh install,
# and sets up Windows-MCP; to rerun it, delete storage/data.img and start again (the ISO is reused).
# Windows-MCP listens on the guest's port 8000 with no authentication; it is published only on the lab's
# 127.0.0.1:8765, because the lab's own port 8000 is the another service tunnel.
set -eu
mkdir -p ~/windows11/storage ~/windows11/oem
docker run -d --name win11-gui --env-file ~/windows11/credentials.env \
  -e VERSION=11e -e RAM_SIZE=16G -e RAM_CHECK=N -e CPU_CORES=8 -e DISK_SIZE=128G -e USER_PORTS=8000 \
  -p 127.0.0.1:8006:8006 -p 127.0.0.1:3389:3389/tcp -p 127.0.0.1:3389:3389/udp -p 127.0.0.1:8765:8000 \
  --device=/dev/kvm --device=/dev/net/tun --cap-add NET_ADMIN --memory 18g \
  -v ~/windows11/storage:/storage -v ~/windows11/oem:/oem --stop-timeout 120 dockurr/windows:6.06
