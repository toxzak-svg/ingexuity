# syntax=docker/dockerfile:1

FROM rust:1.97-bookworm AS builder
WORKDIR /app

COPY Cargo.toml ./
COPY Cargo.lock ./
COPY crates ./crates
COPY fixtures ./fixtures
COPY migrations ./migrations

RUN cargo build --locked --release -p ingexuity-server

FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ingexuity \
    && mkdir -p /app/data \
    && chown -R ingexuity:ingexuity /app/data

COPY --from=builder /app/target/release/ingexuity-server /usr/local/bin/ingexuity-server

USER ingexuity
ENV INGEXUITY_BIND=0.0.0.0:8000
ENV INGEXUITY_DB_PATH=/app/data/ingexuity.sqlite3
VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health/ready || exit 1

ENTRYPOINT ["/usr/local/bin/ingexuity-server"]
