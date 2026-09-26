FROM node:20-alpine

# sqlite3 is a native npm dependency built via node-gyp at install time,
# which needs a real build toolchain -- alpine doesn't ship one by default
# (confirmed against a real build: "npm error gyp ERR!" without these).
RUN apk add --no-cache python3 make g++

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY . .

ENV PORT=3000
EXPOSE 3000

HEALTHCHECK --interval=10s --timeout=5s --retries=10 --start-period=15s \
    CMD wget -qO- http://localhost:3000/health || exit 1

CMD ["npm", "run", "server:http"]
