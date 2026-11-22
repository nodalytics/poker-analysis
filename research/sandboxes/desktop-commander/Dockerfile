FROM node:24-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 procps && rm -rf /var/lib/apt/lists/* \
 && npm install -g @wonderwhy-er/desktop-commander@0.3.0 && npm cache clean --force
RUN useradd -m agent && mkdir -p /home/agent/work && chown agent /home/agent/work
USER agent
WORKDIR /home/agent/work
ENTRYPOINT ["desktop-commander"]
