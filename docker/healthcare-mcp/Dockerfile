FROM node:20-alpine

WORKDIR /app

# package.json / package-lock.json are our audited overlay (docker/healthcare-mcp/,
# copied over the pinned clone by scripts/setup_vendor_mcp.sh). No native
# modules any more (sqlite3 was unused), so no build toolchain in the image.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts

COPY . .

ENV PORT=3000
EXPOSE 3000

HEALTHCHECK --interval=10s --timeout=5s --retries=10 --start-period=15s \
    CMD wget -qO- http://localhost:3000/health || exit 1

CMD ["npm", "run", "server:http"]
