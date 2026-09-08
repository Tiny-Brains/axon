# syntax=docker/dockerfile:1

# Axon — the Model Loader. The one process in TinyBrains that is not Orion, so unlike soma/ and
# kalam/ this image has application code in it and is built rather than assembled.
#
# ONE BINARY, TWO ROLES (docs/design.md §8). `AXON_MODE=replica` stands beside a Kalam replica, fetches
# by hash and serves /play; `AXON_MODE=admission` stands beside Soma, accepts URLs, mirrors what it
# verified, and serves /inspect and /validate while refusing /play. The difference is configuration
# because the trust boundary is: a replica's fetch allowlist is empty *whatever the environment
# says*, so a compromised workflow cannot tell a replica where to pull bytes from.

# ---- build -------------------------------------------------------------------
#
# TRIXIE, NOT BOOKWORM, and the reason is a link error that is worth writing down. `ort`'s
# `download-binaries` feature fetches a prebuilt static onnxruntime, and the aarch64-linux build of
# it is compiled against a newer libstdc++ than Debian 12 ships: linking on bookworm fails with a
# few thousand `undefined reference to __cxa_call_terminate` and
# `basic_string::_M_replace_cold`, which are GCC 13+ symbols. Both stages move together -- the
# runtime image needs the matching libstdc++ at load time for the same reason.
FROM rust:1-trixie AS build

# ort pulls a prebuilt onnxruntime at build time (the `download-binaries` feature), so the builder
# needs to reach the network and the runtime needs the C++ standard library. Nothing else.
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# The tests are not run here. They need the ONNX fixtures, and the gate for this repository is
# `cargo test` on a developer machine and in CI, not a container build that would silently become
# the only place they ran.
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

# The store. A directory here; devops/docs/deployment.md replaces it with S3/R2 through the same trait, at which
# point this volume is a cache rather than the record.
RUN mkdir -p /var/lib/axon && chown axon:axon /var/lib/axon
VOLUME /var/lib/axon

USER axon
ENV AXON_BIND=0.0.0.0:9090 \
    AXON_STORE_DIR=/var/lib/axon
EXPOSE 9090

# /healthz reports the mode, the dialect version and the evaluator digest — docs/design.md §3.7. Kalam's
# entrypoint waits on this exact endpoint before it will load its package, because a replica with
# no loader claims matches it cannot play and each one costs a lease and two lapses.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD curl -fsS http://127.0.0.1:9090/healthz > /dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/axon"]
