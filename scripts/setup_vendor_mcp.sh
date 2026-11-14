#!/usr/bin/env bash
# Fetches the two MCP servers that are built from GitHub source into vendor/
# (healthcare-mcp, med-research-mcp-suite) at PINNED commits, then lays our
# audited dependency overlay over each: docker/<server>/package.json and
# package-lock.json. The Docker images install exactly that lockfile (npm ci).
# medical-mcp (the third server) is an npm package, pinned and locked in
# docker/medical-mcp-bridge/.
#
# Upgrading a server is deliberate: change its commit below, regenerate its
# overlay (see docs/deployment.md, "MCP server supply chain") and run
# `make mcp-audit`.
#
# The overlays differ from upstream where needed:
#   healthcare-mcp: adds dicom-parser (imported by server/dicom-tool.js but missing
#     from upstream's package.json) and drops sqlite3 (never imported; its native
#     build pulled in a vulnerable tar and needed a C++ toolchain in the image).
#   med-research-mcp-suite: tsx moves to dependencies (the image runs
#     src/web-server.ts through it), so dev tooling stays out of the image; the MCP
#     SDK is upgraded to a fixed release (only the stdio src/index.ts imports it).
set -euo pipefail
cd "$(dirname "$0")/.."

HEALTHCARE_MCP_REPO=https://github.com/Cicatriiz/healthcare-mcp-public.git
HEALTHCARE_MCP_COMMIT=1c4c40c3bb63b94ecf71f972d9806c76e7ffee7b
RESEARCH_MCP_REPO=https://github.com/ezhou89/medical-research-mcp-suite.git
RESEARCH_MCP_COMMIT=10ae11baffa0f2d42020ccf8bc17657c8230f6c4

# fetch_pinned <repo> <commit> <dir> <overlay dir>
fetch_pinned() {
  local repo=$1 commit=$2 dir=$3 overlay=$4
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$repo" "$dir"
  fi
  git -C "$dir" fetch --quiet origin
  # Discard anything a previous run (or an older version of this script) changed.
  git -C "$dir" checkout --quiet --force "$commit"
  git -C "$dir" clean --quiet -fdx
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    echo "error: $dir is not at the pinned commit $commit" >&2
    exit 1
  fi
  cp "$overlay/package.json" "$overlay/package-lock.json" "$dir/"
  echo "$dir: $commit + overlay from $overlay"
}

mkdir -p vendor
fetch_pinned "$HEALTHCARE_MCP_REPO" "$HEALTHCARE_MCP_COMMIT" vendor/healthcare-mcp docker/healthcare-mcp
fetch_pinned "$RESEARCH_MCP_REPO" "$RESEARCH_MCP_COMMIT" vendor/med-research-mcp-suite docker/med-research-mcp-suite

# vendor/ is gitignored (third-party clones, not our source), so a tracked
# .dockerignore can't live inside them -- the Docker build context for each is
# its vendor dir (see docker-compose.yml), and `COPY . .` must not pick up a
# host node_modules or secrets.
printf 'node_modules\nlogs\n.env\n.git\n' > vendor/healthcare-mcp/.dockerignore
printf 'node_modules\ndist\nlogs\n.env\n.git\n' > vendor/med-research-mcp-suite/.dockerignore

echo "Done. Audit with: make mcp-audit"
