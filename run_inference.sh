#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_inference.sh DATASET_DIR OUTPUT_DIR [Hydra overrides...]" >&2
    exit 2
fi

asr_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Resolve input paths before changing to the repository directory.
asr_dataset_dir="$(realpath -- "$1")"
asr_output_dir="$(realpath -m -- "$2")"
shift 2
cd -- "$asr_repo_dir"
asr_python="${PYTHON_BIN:-python}"

"$asr_python" script/download_resources.py
"$asr_python" inference.py \
    "custom_dir=\"$asr_dataset_dir\"" \
    "inferencer.save_path=\"$asr_output_dir\"" "$@"

shopt -s nullglob
asr_references=("$asr_dataset_dir"/transcriptions/*.txt)
if [[ ${#asr_references[@]} -gt 0 ]]; then
    "$asr_python" calc_metrics.py \
        --ground-truth "$asr_dataset_dir/transcriptions" \
        --predictions "$asr_output_dir/test" \
        --output "$asr_output_dir/metrics.json"
else
    echo "No ground-truth transcripts supplied. Predictions: $asr_output_dir/test"
fi
