#!/usr/bin/env node
// Bridges medical-mcp (stdio-only, confirmed against its packed build --
// it only registers a StdioServerTransport, no HTTP option) onto a plain
// HTTP endpoint so it's network-reachable from its own container, matching
// the topology of healthcare-mcp and med-research-mcp-suite. Spawns the
// real medical-mcp process once and reuses one MCP ClientSession for every
// request. Contract matches MedicalMCPClient (src/medagent/tools/mcp/medical.py):
// POST /call-tool {"name": ..., "arguments": ...} -> {"content": [...]}.

import http from "http";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const PORT = parseInt(process.env.PORT || "8080", 10);

let clientPromise = null;

async function getClient() {
  if (clientPromise === null) {
    clientPromise = (async () => {
      // `npx -y medical-mcp` fails here: the installed bin script's shebang
      // isn't honored in this image (executed via sh instead of node,
      // confirmed against a real container -- "import: not found"), so
      // invoke its real entry point with node directly instead.
      const transport = new StdioClientTransport({
        command: "node",
        args: ["/app/node_modules/medical-mcp/build/index.js"],
      });
      const client = new Client({ name: "medical-mcp-bridge", version: "1.0.0" }, { capabilities: {} });
      await client.connect(transport);
      return client;
    })();
  }
  return clientPromise;
}

function writeJson(res, statusCode, body) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "GET" && url.pathname === "/health") {
    return writeJson(res, 200, { status: "ok" });
  }

  if (req.method === "POST" && url.pathname === "/call-tool") {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", async () => {
      try {
        const parsed = JSON.parse(body || "{}");
        const client = await getClient();
        const result = await client.callTool({
          name: parsed.name,
          arguments: parsed.arguments || {},
        });
        return writeJson(res, 200, { content: result.content });
      } catch (err) {
        return writeJson(res, 500, { status: "error", error_message: err.message });
      }
    });
    return;
  }

  writeJson(res, 404, { status: "error", error_message: "Not Found" });
});

server.listen(PORT, () => {
  console.error(`medical-mcp bridge listening on http://0.0.0.0:${PORT}`);
});
