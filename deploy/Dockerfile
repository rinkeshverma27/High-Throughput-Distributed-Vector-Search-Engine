# Multi-stage C++20 build
FROM gcc:13 AS builder

WORKDIR /app
COPY . .
RUN make all

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/bin/vectordb /app/vectordb

EXPOSE 9000 9001 9090
VOLUME ["/app/data"]

ENTRYPOINT ["/app/vectordb"]
