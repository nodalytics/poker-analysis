FROM node:24-bookworm-slim
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @playwright/mcp@0.0.83 && npm cache clean --force \
 && cd /usr/local/lib/node_modules/@playwright/mcp && npx playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/*
COPY site.py entry.sh /opt/
RUN chmod +x /opt/entry.sh && useradd -m agent
USER agent
ENTRYPOINT ["/opt/entry.sh"]
