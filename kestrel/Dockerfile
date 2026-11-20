# syntax=docker/dockerfile:1
# Kestrel's server (kestreld) with the 9-max networks, the web app and the self-test, for the lab and CI.
# Build from the repo root (the networks live in llm-research/models):
#   docker build -f kestrel/Dockerfile -t kestrel .
# The heads-up data pack (80 MB) is not in the image: mount it at /data/pack (heads-up tables need it, and so does
# the self-test).

FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake ninja-build libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY kestrel/ kestrel/
COPY llm-research/models/ninemax/nm25/net.bin llm-research/models/ninemax/nm25/net.json llm-research/models/ninemax/nm25/
COPY llm-research/models/ninemax/nm17/net.bin llm-research/models/ninemax/nm17/net.json llm-research/models/ninemax/nm17/
COPY llm-research/models/ninemax/nm30/net.bin llm-research/models/ninemax/nm30/net.json llm-research/models/ninemax/nm30/
COPY llm-research/models/ninemax/nm37/net.bin llm-research/models/ninemax/nm37/net.json llm-research/models/ninemax/nm37/
# portable x86-64 code (no -march=native): CI builds the image on one machine and the lab runs it on another
RUN cmake -S kestrel -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DKESTREL_QT=OFF -DKESTREL_DATA=/src/llm-research \
          -DCMAKE_INSTALL_PREFIX=/opt/kestrel \
    && cmake --build build \
    && cmake --install build \
    && cp kestrel/connectors/*.yaml /opt/kestrel/share/kestrel/connectors/

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libssl3t64 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 kestrel \
    && mkdir -p /data && chown kestrel /data
COPY --from=build /opt/kestrel /opt/kestrel
USER kestrel
# /data holds everything that must survive a new image: kestrel.db, sessions, the access token and the pack
ENV HOME=/home/kestrel XDG_CONFIG_HOME=/data/config XDG_DATA_HOME=/data PATH=/opt/kestrel/bin:$PATH
VOLUME /data
WORKDIR /data
EXPOSE 8890
HEALTHCHECK --interval=60s --timeout=5s CMD curl -fs http://127.0.0.1:8890/v1/health || exit 1
ENTRYPOINT ["kestreld"]
# the same port inside and out: the server checks the Host header, port included (DNS rebinding)
CMD ["serve", "--port", "8890", "--lan", "on", "--ws-port", "0", "--tcp-port", "0", \
     "--web", "/opt/kestrel/share/kestrel/web", "--assets", "/opt/kestrel/share/kestrel/assets", \
     "--connectors", "/opt/kestrel/share/kestrel/connectors", "--models", "/opt/kestrel/share/kestrel/models/ninemax", \
     "--pack", "/data/pack", "--out", "/data/sessions", "--db", "/data/kestrel.db"]
