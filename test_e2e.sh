#!/bin/bash
set -euo pipefail
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: Set DEEPSEEK_API_KEY environment variable"
    exit 1
fi
cp config.example.yaml config.yaml
echo "Running miniQED end-to-end test..."
bash run.sh problem/problem.tex test_output/
echo "Test complete. Check test_output/proof.md and test_output/TOKEN_USAGE.md"
