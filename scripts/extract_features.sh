#!/bin/bash
# Step 2: S2-CLIP features for every model folder in images/<prompt>/.
#
# Usage: bash scripts/extract_features.sh sofa
set -e

PROMPT=${1:-sofa}
NUM_GPUS=4

accelerate launch --num_processes=$NUM_GPUS extract_features.py \
    --image_dir images/$PROMPT \
    --output_dir features/$PROMPT
