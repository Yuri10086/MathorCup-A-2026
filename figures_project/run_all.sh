#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python plot_all.py --project support --output publication_figures --dpi 300
