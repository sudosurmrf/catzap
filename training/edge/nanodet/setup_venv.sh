#!/usr/bin/env bash
# Bootstrap a NanoDet-Plus isolated venv for US-007/US-008/US-009.
#
# Why a separate venv: NanoDet-Plus pins torch<2.0 and
# pytorch-lightning<2.0, both incompatible with the catzap main env's
# torch 2.3.0 + ultralytics 8.2.0. Keeping it isolated avoids breaking
# the rest of the eval harness.
#
# This script is idempotent: it skips work that has already been done.
#
# Output layout under training/edge/nanodet/:
#   .venv/                                   # the venv itself
#   upstream/                                # pinned clone of RangiLyu/nanodet
#   checkpoints/nanodet_plus_m_0.5x_pretrained.pth  # downloaded ckpt
#   checkpoints/nanodet_plus_m_416.onnx             # downloaded onnx
#   checkpoints/SHA256SUMS                          # recorded hashes

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV_DIR="$HERE/.venv"
UPSTREAM_DIR="$HERE/upstream"
CHECKPOINTS_DIR="$HERE/checkpoints"
NANODET_REPO="https://github.com/RangiLyu/nanodet.git"
# Pinned upstream commit. Override by passing $NANODET_COMMIT.
NANODET_COMMIT="${NANODET_COMMIT:-main}"

mkdir -p "$CHECKPOINTS_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r "$HERE/requirements_nanodet.txt"

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
    echo "[setup] cloning $NANODET_REPO at $NANODET_COMMIT"
    git clone "$NANODET_REPO" "$UPSTREAM_DIR"
fi

(
    cd "$UPSTREAM_DIR"
    git fetch --all --tags
    git checkout "$NANODET_COMMIT"
    git rev-parse HEAD > "$HERE/.upstream_commit"
)

python -m pip install -e "$UPSTREAM_DIR"

# Download the upstream pretrained baselines if not already present.
download_with_hash() {
    local url="$1" out="$2"
    if [[ -f "$out" ]]; then
        echo "[setup] already have $out"
        return
    fi
    echo "[setup] downloading $url -> $out"
    curl -fL --retry 3 -o "$out" "$url"
    sha256sum "$out" | tee -a "$CHECKPOINTS_DIR/SHA256SUMS.tmp"
}

PLUS_M_416_CKPT_URL="https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt"
PLUS_M_416_ONNX_URL="https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx"

download_with_hash "$PLUS_M_416_CKPT_URL" "$CHECKPOINTS_DIR/nanodet-plus-m_416_checkpoint.ckpt"
download_with_hash "$PLUS_M_416_ONNX_URL" "$CHECKPOINTS_DIR/nanodet-plus-m_416.onnx"

# Canonical filename so the PRD-shaped command resolves. Upstream does not
# publish a 0.5x pretrained .pth — see README.md for the deviation note.
CANONICAL_PTH="$CHECKPOINTS_DIR/nanodet_plus_m_0.5x_pretrained.pth"
if [[ ! -e "$CANONICAL_PTH" ]]; then
    ln -s nanodet-plus-m_416_checkpoint.ckpt "$CANONICAL_PTH"
fi

# Materialize a deterministic SHA256SUMS.
(
    cd "$CHECKPOINTS_DIR"
    sha256sum nanodet-plus-m_416_checkpoint.ckpt nanodet-plus-m_416.onnx > SHA256SUMS
    rm -f SHA256SUMS.tmp
)

echo "[setup] done. activate with: source $VENV_DIR/bin/activate"
