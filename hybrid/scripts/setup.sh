#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Done. Try: python -m hrm_lm.training.train --dry_run 1"
