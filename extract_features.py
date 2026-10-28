"""Step 2: compute S2-CLIP features for folders of images.

Every subfolder of --image_dir is treated as one model, and its features are
saved to <output_dir>/<subfolder>.pt (subfolders with an existing .pt are skipped).
A .pt file is a list with one {image_filename: 1-D feature tensor} dict per process.

Example (4 GPUs):
    accelerate launch --num_processes=4 extract_features.py \
        --image_dir images/sofa --output_dir features/sofa
"""

import argparse
import os
from datetime import timedelta

import torch
from accelerate import InitProcessGroupKwargs, PartialState
from accelerate.utils import gather_object
from PIL import Image
from tqdm.auto import tqdm

from provenance.features import S2CLIP

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def main(args):
    # allow up to 10 hours for the slowest process to finish
    state = PartialState(**InitProcessGroupKwargs(timeout=timedelta(hours=10)).to_kwargs())
    s2clip = S2CLIP(state.device)
    os.makedirs(args.output_dir, exist_ok=True)

    for model in sorted(os.listdir(args.image_dir)):
        folder = os.path.join(args.image_dir, model)
        output_path = os.path.join(args.output_dir, f"{model}.pt")
        if not os.path.isdir(folder):
            continue
        if os.path.exists(output_path):
            if state.is_main_process:
                print(f"Skipping {model}: {output_path} already exists")
            continue

        image_names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))
        if state.is_main_process:
            print(f"{model}: extracting features for {len(image_names)} images")

        features = {}
        with state.split_between_processes(image_names) as my_image_names:
            for name in tqdm(my_image_names, disable=not state.is_main_process):
                features[name] = s2clip(Image.open(os.path.join(folder, name)))

        features = gather_object([features])
        if state.is_main_process:
            torch.save(features, output_path)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", required=True, help="folder with one subfolder of images per model")
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
