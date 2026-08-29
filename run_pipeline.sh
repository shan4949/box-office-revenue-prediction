#!/usr/bin/env bash
# Runs the full box-office revenue prediction pipeline end-to-end.
# Each step reads from the previous step's output/ folder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found. Run from project root:"
    echo "  python3 -m venv .venv && .venv/bin/pip install pandas numpy lightgbm scikit-learn shap requests"
    exit 1
fi

run_step() {
    local step="$1"
    echo ""
    echo "=========================================="
    echo "  $step"
    echo "=========================================="
    cd "$SCRIPT_DIR/$step"
    "$PYTHON" run.py
    cd "$SCRIPT_DIR"
}

run_step step0_ingest
run_step step1_universe
run_step step2_cpi
run_step step3_leakage
run_step step4_features
run_step step5_starpower
run_step step6_split
run_step step7_baselines
run_step step8_model
run_step step9_validation

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "  Resume metrics: step9_validation/output/resume_metrics.json"
echo "  Full report   : step9_validation/output/validation_report.md"
echo "=========================================="
