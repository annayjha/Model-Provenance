"""Distances between text-to-image models, computed from their image features.

In each trial `n_samples` of images per model:
  1. fit Isomap on the features of all base models together, projecting them
     to `n_components` dims. Every other (target) model, e.g. a fine-tuned
     model of unknown origin, is projected with that same fitted Isomap;
  2. fit a Gaussian (mean, covariance) to each model's projected features;
  3. compute the geometric Jensen-Shannon divergence between every pair of
     Gaussians.
Distances are then averaged over trials. A target model's closest base model
is its predicted provenance.
"""

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


def load_features(path, shuffle_seed=54321):
    """Load a features file as an (n_images, dim) array, in a fixed random order."""
    import torch  # imported here so the worker processes do not each load torch

    gathered = torch.load(path, weights_only=True)  # a list with one {image_name: feature} dict per process
    features = torch.stack([f for per_process in gathered for f in per_process.values()])
    return np.random.RandomState(shuffle_seed).permutation(features.numpy())


def geometric_jensen_shannon(p, q, alpha=0.5):
    """Geometric Jensen-Shannon divergence between two Gaussians given as (mean, cov).

    Unlike the regular JS divergence, which compares each distribution to their
    arithmetic mixture, this compares them to their weighted geometric mean,
    which is again a Gaussian and gives a closed form.
    """
    (mean_p, cov_p), (mean_q, cov_q) = p, q
    cov_p_inv = np.linalg.inv(cov_p)
    cov_q_inv = np.linalg.inv(cov_q)

    # the geometric mean Gaussian
    cov_a_inv = (1 - alpha) * cov_p_inv + alpha * cov_q_inv
    cov_a = np.linalg.inv(cov_a_inv)
    mean_a = cov_a @ ((1 - alpha) * cov_p_inv @ mean_p + alpha * cov_q_inv @ mean_q)

    log_det_ratio = math.log2(
        np.linalg.det(cov_p) ** (1 - alpha) * np.linalg.det(cov_q) ** alpha / np.linalg.det(cov_a)
    )
    return 0.5 * (
        (1 - alpha) * mean_p @ cov_p_inv @ mean_p
        + alpha * mean_q @ cov_q_inv @ mean_q
        - mean_a @ cov_a_inv @ mean_a
        + log_det_ratio
    )


def run_trial(trial, samples, base_models, n_components):
    """Pairwise distances for one trial. `samples` maps model name -> (n_samples, dim)."""
    isomap = Isomap(n_components=n_components)
    projected_base = isomap.fit_transform(np.vstack([samples[m] for m in base_models]))
    projected = dict(zip(base_models, np.split(projected_base, len(base_models))))
    for model, features in samples.items():
        if model not in projected:
            projected[model] = isomap.transform(features)

    gaussians = {m: (x.mean(axis=0), np.cov(x, rowvar=False)) for m, x in projected.items()}
    models = list(gaussians)
    return [
        {
            "trial": trial,
            "model_1": m1,
            "model_2": m2,
            "distance": geometric_jensen_shannon(gaussians[m1], gaussians[m2]),
        }
        for i, m1 in enumerate(models)
        for m2 in models[i + 1 :]
    ]


_thread_limit = None


def _init_worker(n_threads):
    # Split the CPUs between workers so their math libraries do not oversubscribe the machine.
    global _thread_limit
    _thread_limit = threadpool_limits(n_threads)


def compute_distances(features, base_models, n_trials=30, n_samples=100, n_components=10, num_workers=4):
    """Run `n_trials` trials in parallel and return one row per (trial, model pair).

    `features` maps model name -> (n_images, dim) array; every model not in
    `base_models` is treated as a target.
    """
    for model, f in features.items():
        if len(f) < n_trials * n_samples:
            raise ValueError(
                f"{model} has {len(f)} images but {n_trials} trials x {n_samples} samples "
                f"needs {n_trials * n_samples}."
            )

    models = base_models + [m for m in features if m not in base_models]
    trial_samples = [
        {m: features[m][t * n_samples : (t + 1) * n_samples] for m in models} for t in range(n_trials)
    ]
    # "spawn" starts fresh worker processes; forking a process that has already used
    # torch's threads can deadlock on Linux.
    with ProcessPoolExecutor(
        num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(max(1, os.cpu_count() // num_workers),),
    ) as pool:
        results = pool.map(
            run_trial, range(n_trials), trial_samples, repeat(base_models), repeat(n_components)
        )
        rows = [row for trial_rows in tqdm(results, total=n_trials, desc="trials") for row in trial_rows]
    return pd.DataFrame(rows)


def summarize(distances):
    """Mean and std of each pair's distance over trials, closest pairs first."""
    summary = (
        distances.groupby(["model_1", "model_2"], sort=False)["distance"]
        .agg(mean="mean", std=lambda d: np.std(d))
        .reset_index()
    )
    return summary.sort_values("mean", ignore_index=True)
