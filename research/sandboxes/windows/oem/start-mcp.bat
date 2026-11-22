@echo off
rem Starts Windows-MCP in the logged-on user's desktop session, with the auth key from C:\OEM\mcp.key.
rem Windows-MCP refuses a network bind without authentication; the key never appears on a command line in a file.
set /p MCPKEY=<C:\OEM\mcp.key
C:\Python314\Scripts\windows-mcp.exe serve --transport streamable-http --host 0.0.0.0 --port 8000 --auth-key %MCPKEY% >> C:\OEM\windows-mcp.log 2>&1
