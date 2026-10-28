#!/bin/bash
# Step 3: model distances with the paper's settings (30 trials of k=100 images, 10 Isomap dims).
#
# Usage: bash scripts/compute_distances.sh sofa
set -e

PROMPT=${1:-sofa}

python compute_distances.py \
    --features_dir features/$PROMPT \
    --output_dir results/$PROMPT \
    --n_trials 30 \
    --n_samples 100 \
    --n_components 10 \
    --num_workers 4
