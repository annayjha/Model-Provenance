#!/bin/bash
# Download the released S2-CLIP features from Hugging Face into features/.
#
# Usage: bash scripts/download_features.sh sofa     # ~8.3 GB
set -e

HF_REPO="lemonjellybelly/model-provenance-features"
PATTERN=${1:+$1/*}

python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', repo_type='dataset', local_dir='features', allow_patterns='${PATTERN:-*}')
"
