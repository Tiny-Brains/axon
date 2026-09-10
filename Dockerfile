# syntax=docker/dockerfile:1

# One binary, two roles (docs/design.md §8). `AXON_MODE=replica` stands beside a Kalam replica and
# serves /play; `AXON_MODE=admission` stands beside Soma, accepts URLs, mirrors what it verified,
# and serves /inspect and /validate. The difference is configuration because the trust boundary is:
# a replica's fetch allowlist is empty whatever the environment says.

# ---- build -------------------------------------------------------------------
#
# Trixie, not bookworm: `ort`'s prebuilt aarch64-linux onnxruntime is compiled against a newer
# libstdc++ than Debian 12 ships, and linking on bookworm fails with thousands of undefined GCC 13+
# symbols. Both stages move together -- the runtime image needs the matching libstdc++ at load time.
FROM rust:1-trixie AS build

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# The tests are not run here: they need the ONNX fixtures, and the gate for this repository is
# `cargo test` on a developer machine and in CI.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin axon \
 && cp /src/target/release/axon /usr/local/bin/axon

# ---- runtime -----------------------------------------------------------------
FROM debian:trixie-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10002 --create-home --shell /usr/sbin/nologin axon

COPY --from=build /usr/local/bin/axon /usr/local/bin/axon

# A directory store; deployment replaces it with S3/R2 through the same trait, at which point this
# volume is a cache rather than the record.
RUN mkdir -p /var/lib/axon && chown axon:axon /var/lib/axon
VOLUME /var/lib/axon

USER axon
ENV AXON_BIND=0.0.0.0:9090 \
    AXON_STORE_DIR=/var/lib/axon
EXPOSE 9090

# Kalam's entrypoint waits on this exact endpoint before it will load its package: a replica with no
# loader claims matches it cannot play, and each one costs a lease and two lapses.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD curl -fsS http://127.0.0.1:9090/healthz > /dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/axon"]
