# Wine desktop in a container: Xvfb display, fluxbox, x11vnc, noVNC. Windows apps install into a per-app prefix
# on a volume. Installers are never run unless they pass ClamAV (see install-app.sh).
FROM debian:trixie-20261005-slim
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 DISPLAY=:1 WINEDEBUG=-all WINEPREFIX=/home/wine/prefix
RUN dpkg --add-architecture i386 && apt-get update && apt-get install -y --no-install-recommends \
      wine wine32 wine64 winbind xvfb fluxbox x11vnc novnc websockify xdotool x11-utils imagemagick \
      clamav clamav-freshclam ca-certificates curl procps fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/* && freshclam --quiet || true
RUN useradd -m -u 1000 wine && mkdir -p /installers /home/wine/prefix && chown -R wine /installers /home/wine
COPY entry.sh install-app.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entry.sh /usr/local/bin/install-app.sh
USER wine
WORKDIR /home/wine
EXPOSE 5900 6080
ENTRYPOINT ["/usr/local/bin/entry.sh"]
