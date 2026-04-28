#!/usr/bin/env bash
# Clones tflite-micro at a pinned commit, builds the linux microlite static
# library, and links firmware/edge-bench/build/edgebench.
#
# Pinned commit (recorded in firmware/edge-bench/README.md):
#   tensorflow/tflite-micro @ 51bee03bed4776f1de88dd87226ff8c260f88e3c
#
# This script is idempotent: re-running with the same commit is a no-op past
# the cache point. To force a rebuild, delete firmware/edge-bench/third_party/
# and firmware/edge-bench/build/.
#
# Output:
#   firmware/edge-bench/third_party/tflite-micro/  (gitignored)
#   firmware/edge-bench/build/edgebench            (gitignored)
#   firmware/edge-bench/models/hello_world.tflite  (symlink to TFLM example,
#                                                   gitignored)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_COMMIT="${TFLM_COMMIT:-51bee03bed4776f1de88dd87226ff8c260f88e3c}"
TFLM_DIR="${SCRIPT_DIR}/third_party/tflite-micro"
BUILD_DIR="${SCRIPT_DIR}/build"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

# TFLM's download_and_extract.sh shells out to `unzip`. If the host doesn't
# have it (this WSL2 dev box, some CI containers), prepend our python shim.
if ! command -v unzip >/dev/null 2>&1; then
  echo "[edge-bench] /usr/bin/unzip not found; using python shim at tools/unzip"
  export PATH="${SCRIPT_DIR}/tools:${PATH}"
fi

mkdir -p "${SCRIPT_DIR}/third_party" "${BUILD_DIR}" "${SCRIPT_DIR}/models"

if [ ! -d "${TFLM_DIR}/.git" ]; then
  echo "[edge-bench] cloning tflite-micro @ ${TFLM_COMMIT}"
  git clone https://github.com/tensorflow/tflite-micro.git "${TFLM_DIR}"
fi

cd "${TFLM_DIR}"
CURRENT="$(git rev-parse HEAD)"
if [ "${CURRENT}" != "${TFLM_COMMIT}" ]; then
  echo "[edge-bench] checking out pinned commit ${TFLM_COMMIT}"
  git fetch --depth=1 origin "${TFLM_COMMIT}" || git fetch
  git checkout "${TFLM_COMMIT}"
fi

echo "[edge-bench] building microlite (TARGET=linux, jobs=${JOBS})"
make -f tensorflow/lite/micro/tools/make/Makefile \
     TARGET=linux microlite -j"${JOBS}"

# TFLM picks gen/<TARGET>_<TARGET_ARCH>_<BUILD_TYPE>_<TOOLCHAIN>; on linux that's
# `linux_x86_64_default_gcc` (the trailing `_gcc` is the toolchain). Earlier
# TFLM revisions used `linux_x86_64_default`; sniff both.
LIB=""
for candidate in \
  "${TFLM_DIR}/gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a" \
  "${TFLM_DIR}/gen/linux_x86_64_default/lib/libtensorflow-microlite.a"; do
  if [ -f "${candidate}" ]; then
    LIB="${candidate}"
    break
  fi
done
if [ -z "${LIB}" ]; then
  echo "[edge-bench] microlite library not found under gen/" >&2
  find "${TFLM_DIR}/gen" -name "libtensorflow-microlite.a" 2>/dev/null >&2 || true
  exit 1
fi
GEN_DIR="$(dirname "$(dirname "${LIB}")")"
echo "[edge-bench] microlite library: ${LIB}"

echo "[edge-bench] linking edgebench"
cd "${SCRIPT_DIR}"
make -f Makefile.bench TFLM_DIR="${TFLM_DIR}" GEN_DIR="${GEN_DIR}" BUILD_DIR="${BUILD_DIR}"

# Locate any small shipped tflite for the self-test.
HW="${SCRIPT_DIR}/models/hello_world.tflite"
if [ ! -e "${HW}" ]; then
  CANDIDATE="$(find "${TFLM_DIR}/tensorflow/lite/micro/examples" \
               -maxdepth 4 -name '*.tflite' -size -200k 2>/dev/null \
               | sort | head -n 1 || true)"
  if [ -n "${CANDIDATE}" ]; then
    ln -s "${CANDIDATE}" "${HW}"
    echo "[edge-bench] hello_world.tflite -> ${CANDIDATE}"
  else
    echo "[edge-bench] WARNING: no shipped <200KB .tflite found; self-test will need an explicit --model" >&2
  fi
fi

echo "[edge-bench] built ${BUILD_DIR}/edgebench"
