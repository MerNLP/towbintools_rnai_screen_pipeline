#!/bin/bash
# Create the towbintools_rnai_screen conda env from the pinned lock file.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME=towbintools_rnai_screen

if [[ ! -f conda-lock.yml ]]; then
  echo "conda-lock.yml not found. Run ./generate_lock.sh first." >&2
  exit 1
fi

~/.local/bin/micromamba create -n "$ENV_NAME" -f conda-lock.yml -y

echo ""
echo "Done. Activate with:"
echo "  micromamba activate $ENV_NAME"
