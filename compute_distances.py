"""Step 3: compute distances between models and rank base models for each target.

Loads <features_dir>/<model>.pt for every base model and every target model
(by default, targets are all other .pt files in --features_dir), then writes
    <output_dir>/distances_per_trial.csv   one row per (trial, model pair)
    <output_dir>/summary.csv               mean/std over trials per model pair
and prints, for each target, the base models from closest to farthest.

Runs on CPU. Example:
    python compute_distances.py --features_dir features/sofa --output_dir results/sofa
"""

import argparse
import os

from provenance.distance import compute_distances, load_features, summarize

BASE_MODELS = ["flux", "sd21", "deepfloyd", "sana", "sdxl", "hunyuan", "kolors", "pixartsigma"]


def main(args):
    targets = args.targets
    if targets is None:
        all_models = sorted(f[: -len(".pt")] for f in os.listdir(args.features_dir) if f.endswith(".pt"))
        targets = [m for m in all_models if m not in args.base_models]
    print(f"Base models: {args.base_models}")
    print(f"Targets:     {targets}")

    features = {}
    for model in args.base_models + targets:
        print(f"Loading {model} features...")
        features[model] = load_features(os.path.join(args.features_dir, f"{model}.pt"))

    distances = compute_distances(
        features,
        args.base_models,
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        n_components=args.n_components,
        num_workers=args.num_workers,
    )
    summary = summarize(distances)

    os.makedirs(args.output_dir, exist_ok=True)
    distances.to_csv(os.path.join(args.output_dir, "distances_per_trial.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print(f"\nSaved results to {args.output_dir}")

    for target in targets:
        pairs = summary[(summary.model_1 == target) | (summary.model_2 == target)]
        pairs = pairs[pairs.model_1.isin(args.base_models) | pairs.model_2.isin(args.base_models)]
        print(f"\nDistance from {target} to each base model (closest = predicted provenance):")
        for _, row in pairs.iterrows():
            base = row.model_2 if row.model_1 == target else row.model_1
            print(f"  {base:<12} {row['mean']:8.3f} +- {row['std']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features_dir", required=True, help="folder of <model>.pt feature files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_models", nargs="+", default=BASE_MODELS)
    parser.add_argument("--targets", nargs="+", help="default: every other .pt file in --features_dir")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=100, help="images per model per trial (k)")
    parser.add_argument("--n_components", type=int, default=10, help="Isomap dimensions")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="parallel trials; each needs extra memory, lower this if the run stalls")
    main(parser.parse_args())
