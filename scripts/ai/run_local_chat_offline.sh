#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
set -euo pipefail

# Runs the QuantoniumOS chatbox using a locally cached HF model, with no network.
#
# Usage (first time, while online):
#   python src/apps/cache_local_llm.py --model distilgpt2 --cache-dir ai/hf_cache
#
# Then (offline):
#   scripts/run_local_chat_offline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export QUANTONIUM_LOCAL_LLM=1
export QUANTONIUM_LOCAL_ONLY=1
export HF_HOME="${HF_HOME:-ai/hf_cache}"
export QUANTONIUM_MODEL_ID="${QUANTONIUM_MODEL_ID:-distilgpt2}"

python src/apps/qshll_chatbox.py
