"""
VISOR Evaluation Script for STORM
Computes OA (Object Accuracy) and VISOR uncond/cond metrics
using OWL-ViT as the object detector, matching the paper's methodology.

Usage:
    python eval_visor.py --image_root outputs/ --model_name [og_storm|new_storm]

Expected folder structure (matching run.py output):
    outputs/
        og_storm/
            a cake to the left of a suitcase/
                6143.png
                7792.png
                ...
        new_storm/
            a cake to the left of a suitcase/
                6143.png
                ...
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import pipeline as hf_pipeline
from tqdm import tqdm


# ── Prompts ────────────────────────────────────────────────────────────────────
# Covering all 4 directions, matching objects from paper figures
EVAL_PROMPTS = [
    # left
    "a cake to the left of a suitcase",
    "a bottle to the left of a suitcase",
    "a dog to the left of a wine glass",
    "an elephant to the left of a laptop",
    "a spoon to the left of a teddy bear",
    # right
    "a bottle to the right of a suitcase",
    "an elephant to the right of a clock",
    "a bicycle to the right of a bear",
    "a train to the right of a vase",
    "a car to the right of a traffic light",
    # above
    "a horse above a cat",
    "an orange above a cat",
    "a refrigerator above a couch",
    "a cat above a pillow",
    "a suitcase above a car",
    # below
    "a handbag below an umbrella",
    "a skateboard below an apple",
    "a giraffe below a sheep",
    "a potted plant below a bench",
    "a sports ball below a sandwich",
]

SEEDS = [6143, 7792, 8892, 9010]


# ── Spatial relation parsing ────────────────────────────────────────────────────
def parse_prompt(prompt: str):
    """
    Returns (obj_a, relation, obj_b) from a prompt like
    'a cake to the left of a suitcase'
    Handles: left, right, above, below
    """
    patterns = [
        (r"^a[n]? (.+?) to the left of a[n]? (.+)$",   "left"),
        (r"^a[n]? (.+?) to the right of a[n]? (.+)$",  "right"),
        (r"^a[n]? (.+?) above a[n]? (.+)$",             "above"),
        (r"^a[n]? (.+?) below a[n]? (.+)$",             "below"),
    ]
    for pattern, relation in patterns:
        m = re.match(pattern, prompt.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip(), relation, m.group(2).strip()
    raise ValueError(f"Could not parse prompt: {prompt}")


# ── Detection ───────────────────────────────────────────────────────────────────
def get_box_center(box):
    """Returns (cx, cy) from [x1, y1, x2, y2]."""
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def detect_objects(detector, image: Image.Image, obj_a: str, obj_b: str, threshold: float = 0.1):
    """
    Runs OWL-ViT on the image for obj_a and obj_b.
    Returns best box for each or None if not detected.
    """
    results = detector(image, candidate_labels=[obj_a, obj_b])

    best = {obj_a: None, obj_b: None}
    scores = {obj_a: -1, obj_b: -1}

    for r in results:
        label = r["label"]
        score = r["score"]
        if label in best and score > threshold and score > scores[label]:
            best[label] = r["box"]  # {"xmin", "ymin", "xmax", "ymax"}
            scores[label] = score

    return best[obj_a], best[obj_b]


def check_spatial_relation(box_a, box_b, relation: str) -> bool:
    """
    Centroid-based spatial check, exactly as described in the paper.
    Returns True if the detected relation matches the ground truth.
    """
    cx_a, cy_a = get_box_center([box_a["xmin"], box_a["ymin"], box_a["xmax"], box_a["ymax"]])
    cx_b, cy_b = get_box_center([box_b["xmin"], box_b["ymin"], box_b["xmax"], box_b["ymax"]])

    if relation == "left":
        return cx_a < cx_b
    elif relation == "right":
        return cx_a > cx_b
    elif relation == "above":
        return cy_a < cy_b   # y increases downward in image coords
    elif relation == "below":
        return cy_a > cy_b
    return False


# ── Per-image evaluation ────────────────────────────────────────────────────────
def evaluate_image(detector, image_path: Path, obj_a: str, relation: str, obj_b: str):
    """
    Returns dict with:
        both_detected: bool
        spatial_correct: bool (only meaningful if both_detected)
    """
    image = Image.open(image_path).convert("RGB")
    box_a, box_b = detect_objects(detector, image, obj_a, obj_b)

    both_detected = box_a is not None and box_b is not None
    spatial_correct = False

    if both_detected:
        spatial_correct = check_spatial_relation(box_a, box_b, relation)

    return {"both_detected": both_detected, "spatial_correct": spatial_correct}


# ── VISOR metrics ───────────────────────────────────────────────────────────────
def compute_visor_metrics(results_per_prompt: dict):
    """
    results_per_prompt: {
        prompt: [
            {"both_detected": bool, "spatial_correct": bool},  # seed 1
            ...                                                  # seed 2,3,4
        ]
    }

    Returns OA, VISOR_uncond, VISOR_cond, VISOR_1, VISOR_2, VISOR_3, VISOR_4
    """
    total_images = 0
    oa_count = 0

    visor_uncond_count = 0
    visor_cond_count = 0
    visor_cond_total = 0

    # For VISOR_n: count prompts with at least n spatially correct out of 4
    visor_n = {1: 0, 2: 0, 3: 0, 4: 0}
    total_prompts = len(results_per_prompt)

    for prompt, seed_results in results_per_prompt.items():
        n_seeds = len(seed_results)
        total_images += n_seeds

        n_both_detected = sum(r["both_detected"] for r in seed_results)
        n_spatial_correct = sum(
            r["both_detected"] and r["spatial_correct"] for r in seed_results
        )

        # OA: both objects present
        oa_count += n_both_detected

        # VISOR uncond: spatial correct AND both detected, over all images
        visor_uncond_count += n_spatial_correct

        # VISOR cond: spatial correct / both detected (conditioned on presence)
        if n_both_detected > 0:
            visor_cond_count += n_spatial_correct
            visor_cond_total += n_both_detected

        # VISOR_n: at least n out of 4 seeds spatially correct
        for n in [1, 2, 3, 4]:
            if n_spatial_correct >= n:
                visor_n[n] += 1

    oa = 100 * oa_count / total_images if total_images > 0 else 0
    visor_uncond = 100 * visor_uncond_count / total_images if total_images > 0 else 0
    visor_cond = 100 * visor_cond_count / visor_cond_total if visor_cond_total > 0 else 0
    visor_1 = 100 * visor_n[1] / total_prompts if total_prompts > 0 else 0
    visor_2 = 100 * visor_n[2] / total_prompts if total_prompts > 0 else 0
    visor_3 = 100 * visor_n[3] / total_prompts if total_prompts > 0 else 0
    visor_4 = 100 * visor_n[4] / total_prompts if total_prompts > 0 else 0

    return {
        "OA (%)":           round(oa, 2),
        "VISOR_uncond (%)": round(visor_uncond, 2),
        "VISOR_cond (%)":   round(visor_cond, 2),
        "VISOR_1 (%)":      round(visor_1, 2),
        "VISOR_2 (%)":      round(visor_2, 2),
        "VISOR_3 (%)":      round(visor_3, 2),
        "VISOR_4 (%)":      round(visor_4, 2),
    }


# ── Main ────────────────────────────────────────────────────────────────────────
def run_eval(image_root: str, model_name: str, seeds: list, prompts: list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OWL-ViT on {device}...")
    detector = hf_pipeline(
        "zero-shot-object-detection",
        model="google/owlvit-base-patch32",
        device=0 if device == "cuda" else -1
    )

    root = Path(image_root) / model_name
    results_per_prompt = {}

    for prompt in tqdm(prompts, desc=f"Evaluating {model_name}"):
        try:
            obj_a, relation, obj_b = parse_prompt(prompt)
        except ValueError as e:
            print(f"Skipping: {e}")
            continue

        prompt_dir = root / prompt
        if not prompt_dir.exists():
            print(f"Missing: {prompt_dir}")
            continue

        seed_results = []
        for seed in seeds:
            img_path = prompt_dir / f"{seed}.png"
            if not img_path.exists():
                print(f"  Missing image: {img_path}")
                continue
            result = evaluate_image(detector, img_path, obj_a, relation, obj_b)
            seed_results.append(result)

        if seed_results:
            results_per_prompt[prompt] = seed_results

    metrics = compute_visor_metrics(results_per_prompt)

    print(f"\n{'='*50}")
    print(f"Results for: {model_name}")
    print(f"Prompts evaluated: {len(results_per_prompt)}/{len(prompts)}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    print(f"{'='*50}\n")

    # Save detailed results
    out_path = Path(image_root) / f"eval_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "per_prompt": results_per_prompt}, f, indent=2)
    print(f"Saved detailed results to {out_path}")

    return metrics


def compare_models(image_root: str, model_names: list, seeds: list, prompts: list):
    all_metrics = {}
    for name in model_names:
        all_metrics[name] = run_eval(image_root, name, seeds, prompts)

    if len(model_names) > 1:
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        header = f"{'Metric':<22}" + "".join(f"{n:>15}" for n in model_names)
        print(header)
        print("-"*70)
        for metric in all_metrics[model_names[0]].keys():
            row = f"{metric:<22}" + "".join(
                f"{all_metrics[n][metric]:>15}" for n in model_names
            )
            print(row)
        print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="outputs",
                        help="Root folder containing model subfolders")
    parser.add_argument("--models", nargs="+", default=["og_storm", "new_storm"],
                        help="Model folder names to evaluate (space separated)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds used during generation")
    args = parser.parse_args()

    compare_models(
        image_root=args.image_root,
        model_names=args.models,
        seeds=args.seeds,
        prompts=EVAL_PROMPTS,
    )