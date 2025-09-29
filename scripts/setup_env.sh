#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
ENV_FILE="${PROJECT_ROOT}/env/environment.yml"
ENV_NAME="simple-game"
ROM_DIR="${HOME}/.gymnasium/atari_roms"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "[setup] Creating conda environment ${ENV_NAME}"
  conda env create -f "${ENV_FILE}"
else
  echo "[setup] Conda environment ${ENV_NAME} already exists"
fi

echo "[setup] Ensuring ROM directory exists at ${ROM_DIR}"
mkdir -p "${ROM_DIR}"

echo "[setup] Installing Atari ROMs to ${ROM_DIR}"
conda run -n "${ENV_NAME}" AutoROM --accept-license --install-dir "${ROM_DIR}" --quiet

echo "[setup] Done. Activate the environment with: conda activate ${ENV_NAME}"
