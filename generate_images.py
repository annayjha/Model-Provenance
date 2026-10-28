"""Step 1: generate images of one prompt from one text-to-image model.

Image seeds are drawn from a RNG seeded with --initial_seed, so a run is fully
determined by (model, prompt, initial_seed, n). Alternatively, --seeds_file
gives an explicit list of image seeds, one per line. Images are saved as
<output_dir>/<seed>.png; existing images are skipped, so an interrupted run can
simply be restarted.

Example (4 GPUs):
    accelerate launch --num_processes=4 generate_images.py \
        --model sdxl --prompt sofa --initial_seed 20934857 --n 3000 \
        --output_dir images/sofa/sdxl
"""

import argparse
import json
import os
import random
from datetime import timedelta

from accelerate import InitProcessGroupKwargs, PartialState
from tqdm.auto import tqdm

from provenance.models import MODELS, load_sampler


def main(args):
    with open(args.prompts_file) as f:
        prompt = json.load(f)[args.prompt]

    if args.seeds_file:
        with open(args.seeds_file) as f:
            seeds = [int(line) for line in f if line.strip()]
    else:
        rng = random.Random(args.initial_seed)
        seeds = [rng.randint(0, 2**32 - 1) for _ in range(args.n)]

    # allow up to 10 hours for the slowest process to finish
    state = PartialState(**InitProcessGroupKwargs(timeout=timedelta(hours=10)).to_kwargs())
    my_seeds = seeds[state.process_index :: state.num_processes]
    os.makedirs(args.output_dir, exist_ok=True)

    sample = load_sampler(args.model, state.device, checkpoint=args.checkpoint)
    if state.is_main_process:
        print(f"Generating {len(seeds)} images from {args.model} for prompt: {prompt}")

    for seed in tqdm(my_seeds, disable=not state.is_main_process):
        output_path = os.path.join(args.output_dir, f"{seed}.png")
        if os.path.exists(output_path) and not args.overwrite:
            continue
        sample(prompt, seed).save(output_path)

    state.wait_for_everyone()
    if state.is_main_process:
        print(f"Done. Images saved in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--prompt", required=True, help="a key of --prompts_file, e.g. sofa")
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--initial_seed", type=int)
    seed_group.add_argument("--seeds_file", help="text file with one image seed per line (ignores --n)")
    parser.add_argument("--n", type=int, default=3000, help="number of images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", help="override the model's default weights, e.g. a fine-tuned model")
    parser.add_argument("--prompts_file", default="configs/prompts.json")
    parser.add_argument("--overwrite", action="store_true", help="regenerate images that already exist")
    main(parser.parse_args())
