#!/bin/bash
# Step 1: generate the 3,000 images per model behind the released features.
# The initial seeds below reproduce the exact image seeds in features/<prompt>/*.pt.
#
# Usage: bash scripts/generate_images.sh sofa
set -e

PROMPT=${1:-sofa}
N=3000
NUM_GPUS=4
LAUNCH="accelerate launch --num_processes=$NUM_GPUS generate_images.py --prompt $PROMPT --n $N"

if [ "$PROMPT" = "sofa" ]; then
    $LAUNCH --model sd21        --initial_seed 4567820493    --output_dir images/sofa/sd21
    $LAUNCH --model sdxl        --initial_seed 20934857      --output_dir images/sofa/sdxl
    $LAUNCH --model deepfloyd   --initial_seed 29485763      --output_dir images/sofa/deepfloyd
    $LAUNCH --model flux        --initial_seed 5478692045673 --output_dir images/sofa/flux
    $LAUNCH --model hunyuan     --initial_seed 83475683      --output_dir images/sofa/hunyuan
    $LAUNCH --model kolors      --initial_seed 9234857       --output_dir images/sofa/kolors
    $LAUNCH --model pixartsigma --initial_seed 348756        --output_dir images/sofa/pixartsigma
    $LAUNCH --model sana        --initial_seed 34675893      --output_dir images/sofa/sana

    # Target: SD2.1 fine-tuned for 100k steps (weights not released; point --checkpoint to your own).
    $LAUNCH --model sd21 --checkpoint /path/to/ft-sd21-pcb-100k \
        --initial_seed 85476548432385702 --output_dir images/sofa/ft-sd21-pcb-100k

else
    echo "No seeds recorded for prompt '$PROMPT'; pick any --initial_seed per model." && exit 1
fi
