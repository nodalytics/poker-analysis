// The same methods over JSON-RPC 2.0 on three more interfaces, each a front door to the REST API (so the same checks
// apply): WebSocket (with pushed events), plain TCP (one request per line) and MCP over stdio (for LLMs and agents).
#pragma once
#include <atomic>
#include <string>

#include <nlohmann/json.hpp>

namespace kestrel {

// one JSON-RPC request -> its response (null for a notification), served by the REST API on api_port
nlohmann::json rpc_dispatch(int api_port, const nlohmann::json& request);
// the methods, with a one-line description and a JSON schema for their parameters (MCP's tools/list)
nlohmann::json rpc_methods();

// background servers on 127.0.0.1; they stop when quit becomes true
void start_websocket(int api_port, int ws_port, const std::atomic<bool>& quit);
void start_tcp(int api_port, int tcp_port, const std::atomic<bool>& quit);

// MCP over stdin/stdout until stdin closes (newline-delimited JSON-RPC)
int mcp_stdio(int api_port, const std::string& version);

}  // namespace kestrel
