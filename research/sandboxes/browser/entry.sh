#!/bin/sh
python3 /opt/site.py &
exec playwright-mcp --headless --browser chromium --no-sandbox --isolated
