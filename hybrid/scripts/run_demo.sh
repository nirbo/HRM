#!/usr/bin/env bash
set -e
source venv/bin/activate 2>/dev/null || source venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m hrm_lm.training.train --dry_run 1
