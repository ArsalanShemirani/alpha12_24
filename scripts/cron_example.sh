#!/usr/bin/env bash
set -euo pipefail
cd /path/to/alpha12_24
export PYTHONPATH=$(pwd)
source .venv/bin/activate
python -m src.eval.daily_score --runs runs --charts >> runs/score_cron.log 2>&1