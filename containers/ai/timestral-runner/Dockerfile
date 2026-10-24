# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
# The timestral inference package (Apache-2.0) is not published on PyPI: the source tarball of a
# pinned commit is installed without dependencies, which are the locked packages of pyproject.toml.
ARG TIMESTRAL_COMMIT=a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
ARG TIMESTRAL_URL=https://github.com/mistralai/timestral/archive/${TIMESTRAL_COMMIT}.tar.gz
ARG TIMESTRAL_DIGEST=sha256:be734f502f88e95b138558d765f3ad3b3ac8d2662270f9cf8b03138359828150

ADD --checksum=${TIMESTRAL_DIGEST} ${TIMESTRAL_URL} /tmp/timestral.tar.gz

COPY --chown=arduino:arduino --chmod=755 ./src /app
COPY ./pyproject.toml ./uv.lock /tmp/deps/

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    apt-get update -y; \
    # OpenMP runtime of the CPU torch wheel, binutils only to strip the native libraries below
    apt-get install -y --no-install-recommends --no-install-suggests libgomp1 binutils; \
    # The index URLs put torch on the PyTorch CPU index; every artifact is hash checked, so the best match across indexes is safe
    uv export --frozen --emit-index-url --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes --index-strategy unsafe-best-match -r -; \
    tar -xzf /tmp/timestral.tar.gz -C /tmp; \
    uv pip install --system --no-cache-dir --no-deps "/tmp/timestral-${TIMESTRAL_COMMIT}"; \
    rm -rf /tmp/deps /tmp/timestral.tar.gz "/tmp/timestral-${TIMESTRAL_COMMIT}"; \
    # Strip the debug symbols of the native libraries and drop the test suites to reduce the image size
    find /usr/local/lib/python3.13/site-packages -type f \( -name "*.so" -o -name "*.so.*" \) -exec strip -s {} + ; \
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -prune -exec rm -r {} +; \
    apt-get purge -y --auto-remove binutils; \
    apt-get dist-clean; \
    rm -rf /var/lib/apt/lists/*; \
    # Model cache, the compose file mounts a volume here so the weights are downloaded once
    mkdir -p /models; \
    chown -R arduino:arduino /models /app; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall -q /usr/local/lib /app

EXPOSE 1345

USER arduino

WORKDIR /app

ENV HOME=/app
ENV USER=arduino
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# The Hugging Face Hub cache the checkpoints are downloaded to, on the models volume
ENV HF_HOME=/models
ENV HF_HUB_DISABLE_TELEMETRY=1

ENTRYPOINT ["/app/start-forecast-server.sh"]
