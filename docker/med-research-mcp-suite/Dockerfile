FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
# --ignore-scripts: this package's "prepare" lifecycle hook runs `npm run
# build` (tsc) automatically on `npm ci`, and tsconfig.json's own `exclude`
# list drops src/web-server.ts from that build anyway (confirmed against the
# real project -- dist/web-server.js is never emitted), so the compiled
# build is neither reliable nor needed for what this image actually runs.
RUN npm ci --ignore-scripts

COPY . .

ENV PORT=3000
EXPOSE 3000

HEALTHCHECK --interval=10s --timeout=5s --retries=10 --start-period=15s \
    CMD wget -qO- http://localhost:3000/api/status || exit 1

# web-server.ts isn't part of the tsc build target -- run it directly via
# tsx (already a devDependency, and how the project's own `npm run web`
# script works) instead of a precompiled dist/web-server.js.
CMD ["npx", "tsx", "src/web-server.ts"]
