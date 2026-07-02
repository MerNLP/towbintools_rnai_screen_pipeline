#!/bin/bash
# Regenerate conda-lock.yml from environment.yml (run from this directory).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

rm -f conda-lock.yml

~/.local/bin/micromamba run -n towbintools conda-lock -f environment.yml -p linux-64 --kind lock
~/.local/bin/micromamba run -n towbintools conda-lock render --kind env

echo "Wrote conda-lock.yml and conda-linux-64.lock.yml"
