#!/usr/bin/env bash
# Clones and builds the two local-build MCP servers referenced by AGENTS.md's
# MCP Server Config (healthcare-mcp, med-research-mcp-suite). Neither is
# published to npm under those names -- they're real public repos meant to be
# vendored and built locally. `medical-mcp` (the third server) IS a real npx
# package and needs no setup here.
#
# Includes two workarounds for confirmed bugs in the upstream projects:
#   1. healthcare-mcp-public is missing `dicom-parser` from package.json's
#      dependencies despite importing it in server/dicom-tool.js.
#   2. Its real entry point is server/index.js, not build/index.js as
#      AGENTS.md's original config guessed (already fixed in core/config.py).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p vendor

if [ ! -d vendor/healthcare-mcp ]; then
  git clone https://github.com/Cicatriiz/healthcare-mcp-public.git vendor/healthcare-mcp
fi
(cd vendor/healthcare-mcp && npm install && npm install dicom-parser)

if [ ! -d vendor/med-research-mcp-suite ]; then
  git clone https://github.com/ezhou89/medical-research-mcp-suite.git vendor/med-research-mcp-suite
fi
(cd vendor/med-research-mcp-suite && npm install && npm run build)

echo "Done. Verify with:"
echo "  node vendor/healthcare-mcp/server/index.js < /dev/null"
echo "  node vendor/med-research-mcp-suite/dist/index.js < /dev/null"
