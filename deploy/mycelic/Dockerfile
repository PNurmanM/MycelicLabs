# syntax=docker/dockerfile:1.7
# Mycelic service image: API + outbox publisher + JetStream consumer/aggregator in one binary.
# Build from the repository root:  docker build -f deploy/mycelic/Dockerfile -t mycelic:local .

# ---- build: the release binary, nothing else -------------------------------------------------------------------------
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
# Behind a TLS-inspecting corporate proxy, pass its CA bundle as a build secret so cargo can reach crates.io:
#   docker build --secret id=ca_bundle,src=/path/to/corporate-ca.pem ...   (compose: CA_BUNDLE_FILE in .env)
# Without the secret the mount is absent or empty and cargo uses its normal trust store.
RUN --mount=type=secret,id=ca_bundle,required=false \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    sh -c 'if [ -s /run/secrets/ca_bundle ]; then export CARGO_HTTP_CAINFO=/run/secrets/ca_bundle; fi; \
           cargo build --release --locked -p mycelic --bin mycelic && cp target/release/mycelic /usr/local/bin/mycelic'

# ---- runtime: Debian slim with the binary; no compiler, no package manager cache --------------------------------------
FROM debian:bookworm-slim
# ca-certificates: the system roots, for verified TLS to a broker (tls://) when MYCELIC_NATS_CA_FILE is not set
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/mycelic /usr/local/bin/mycelic

ENV MYCELIC_HOST=0.0.0.0 MYCELIC_PORT=8080 MYCELIC_DB_PATH=/data/mycelic.db

# Run as a dedicated non-root user; /data is prepared so a fresh named volume is writable on first start.
RUN useradd -r -u 10001 -s /usr/sbin/nologin mycelic && mkdir -p /data && chown mycelic:mycelic /data
VOLUME ["/data"]
USER mycelic
EXPOSE 8080

# The image has no curl/wget: the binary probes itself.  /health answers 200 while the database is open; the probe
# speaks HTTPS (without verifying the certificate) when the service is configured to serve TLS itself.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=4 CMD ["/usr/local/bin/mycelic", "healthcheck"]

ENTRYPOINT ["/usr/local/bin/mycelic"]
CMD ["serve"]
