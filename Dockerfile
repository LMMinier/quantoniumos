# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos

# QuantoniumOS quick validation container
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential nasm git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /qos
COPY . /qos

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .[dev] && \
    make -C algorithms/rft/kernels all || true

CMD ["pytest", "-m", "not slow"]
