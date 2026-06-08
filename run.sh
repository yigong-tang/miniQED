#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBLEM_FILE="${1:-$SCRIPT_DIR/problem/problem.tex}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/proof_output}"
CONFIG="${3:-$SCRIPT_DIR/config.yaml}"

# Detect mode
if [ "${1:-}" = "--learn" ]; then
    echo "============================================================"
    echo "  miniQED -- Proof Pipeline + Interactive Learning"
    echo "============================================================"
    python -m mini_qed.orchestrator --config "$CONFIG" --input "$PROBLEM_FILE" --output "$OUTPUT_DIR" --learn
elif [ "${1:-}" = "--learn-only" ]; then
    echo "============================================================"
    echo "  miniQED -- Interactive Learning Mode"
    echo "============================================================"
    python -m mini_qed.orchestrator --config "$CONFIG" --input "$PROBLEM_FILE" --output "$OUTPUT_DIR" --learn-only
else
    echo "============================================================"
    echo "  miniQED -- Mathematical Proof Pipeline"
    echo "============================================================"
    echo "  Problem:  $PROBLEM_FILE"
    echo "  Output:   $OUTPUT_DIR"
    echo "  Config:   $CONFIG"
    echo ""
    python -m mini_qed.orchestrator --config "$CONFIG" --input "$PROBLEM_FILE" --output "$OUTPUT_DIR"
fi
