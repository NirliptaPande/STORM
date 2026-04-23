"""
Compare performance of pipeline vs poisson_pipeline on spatial accuracy metrics.

Two-phase workflow:
  Phase 1 — Generation:  Run both pipelines, save images + attention maps.
  Phase 2 — Evaluation:  Reuse eval_visor.py (OWL-ViT detector + correct VISOR
                          metric computation) to compute OA, VISOR uncond/cond,
                          and VISOR 1/2/3/4 on the generated images.

Output structure (compatible with eval_visor.py):
    <output_dir>/
        poisson_pipeline/           # model folder name
            <prompt string>/        # raw prompt as folder name
                <seed>.png
        pipeline/
            <prompt string>/
                <seed>.png
        eval_poisson_pipeline.json
        eval_pipeline.json

Usage:
    python compare.py                         # generate + evaluate
    python compare.py --eval_only             # skip generation, just evaluate
    python compare.py --save_attention        # also save attention maps
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from config import RunConfig
from run import load_model, get_noun_indices_to_alter, run_on_prompt
from utils.ptp_utils import AttentionStore
from utils.attention_utils import AttentionConfig
from eval_visor import EVAL_PROMPTS, compare_models as eval_compare_models


MODEL_VARIANTS: List[Dict[str, Any]] = [
    {
        'key': 'poisson',
        'label': 'poisson_pipeline',
        'model_name': 'poisson_pipeline',
        'run_standard_sd': False,
    },
    {
        'key': 'storm',
        'label': 'pipeline',
        'model_name': 'pipeline',
        'run_standard_sd': False,
    },
    {
        'key': 'sd',
        'label': 'basic_sd',
        'model_name': 'pipeline',
        'run_standard_sd': True,
    },
]

MODEL_VARIANTS_BY_KEY: Dict[str, Dict[str, Any]] = {
    variant['key']: variant for variant in MODEL_VARIANTS
}


def resolve_model_variants(selected_models: List[str]) -> List[Dict[str, Any]]:
    """Resolve user-selected model keys into compare.py variant configs."""
    normalized = [model.strip().lower() for model in selected_models if model and model.strip()]
    if not normalized:
        raise ValueError("No models selected. Choose any of: sd, storm, poisson")

    invalid = sorted({model for model in normalized if model not in MODEL_VARIANTS_BY_KEY})
    if invalid:
        allowed = ", ".join(MODEL_VARIANTS_BY_KEY.keys())
        raise ValueError(f"Invalid model(s): {', '.join(invalid)}. Allowed values: {allowed}")

    deduped_models = list(dict.fromkeys(normalized))
    return [MODEL_VARIANTS_BY_KEY[model] for model in deduped_models]


# ── Phase 1: Generation ────────────────────────────────────────────────────────

def generate_images(
    prompts: List[str],
    seeds: List[int],
    output_dir: Path,
    model_variants: List[Dict[str, Any]],
    save_attention: bool = False,
):
    """
    Generate images for both pipeline versions.

    Images are saved as  <output_dir>/<model_name>/<prompt>/<seed>.png
    so that eval_visor.py can find them directly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timing_summary: Dict[str, Dict[str, float]] = {}  # label -> {total, per_prompt}

    for variant in model_variants:
        label = variant['label']
        model_name = variant['model_name']
        run_standard_sd = variant['run_standard_sd']

        print(f"\n{'='*70}")
        print(f"  GENERATING: {label} (backend={model_name}, run_standard_sd={run_standard_sd})")
        print(f"{'='*70}")

        model_output = output_dir / label

        attn_config = AttentionConfig(
            save=save_attention,
            save_steps=[0, 10, 17, 25, 30, 37, 42, 49],
            save_dir=(model_output / 'attention') if save_attention else None,
            display=False,
        )

        config = RunConfig(
            model_name=model_name,
            seeds=seeds,
            output_path=model_output,
            run_standard_sd=run_standard_sd,
            attention_config=attn_config,
        )

        model = load_model(config)

        variant_start = time.perf_counter()
        prompt_times: List[float] = []

        for prompt in tqdm(prompts, desc=f"Prompts ({label})"):
            token_indices = get_noun_indices_to_alter(model, prompt)
            prompt_start = time.perf_counter()

            for seed in seeds:
                g = torch.Generator('cuda').manual_seed(seed)
                controller = AttentionStore()
                image = None

                try:
                    image = run_on_prompt(
                        prompt=prompt,
                        model=model,
                        controller=controller,
                        token_indices=token_indices,
                        seed=g,
                        config=config,
                    )

                    # Save with folder/file layout that eval_visor.py expects
                    prompt_dir = model_output / prompt
                    prompt_dir.mkdir(parents=True, exist_ok=True)
                    image.save(prompt_dir / f"{seed}.png")

                except torch.cuda.OutOfMemoryError as e:
                    print(f"  OOM: prompt='{prompt}' seed={seed}: {e}")
                except Exception as e:
                    print(f"  ERROR: prompt='{prompt}' seed={seed}: {e}")
                finally:
                    # Aggressive cleanup after each seed to prevent memory accumulation
                    controller.reset()
                    if image is not None:
                        del image
                    del controller, g
                    torch.cuda.empty_cache()

            prompt_times.append(time.perf_counter() - prompt_start)

        variant_total = time.perf_counter() - variant_start
        timing_summary[label] = {
            'total': variant_total,
            'per_prompt': sum(prompt_times) / len(prompt_times) if prompt_times else 0.0,
        }

    del model
    torch.cuda.empty_cache()

    print(f"\nGeneration complete. Images saved under {output_dir}/")

    # ── Timing summary ────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"{'Model':<20} {'Total (s)':>12} {'Per-prompt (s)':>16}")
    print(f"{'─'*52}")
    for lbl, t in timing_summary.items():
        print(f"{lbl:<20} {t['total']:>12.1f} {t['per_prompt']:>16.1f}")
    print(f"{'─'*52}")

    timing_path = output_dir / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"Timing saved to {timing_path}")


# ── Phase 2: Evaluation ────────────────────────────────────────────────────────

def evaluate(
    prompts: List[str],
    seeds: List[int],
    output_dir: Path,
    model_variants: List[Dict[str, Any]],
):
    """
    Evaluate generated images using eval_visor.py's OWL-ViT detection and
    VISOR metric computation.  Prints a side-by-side comparison table.
    """
    eval_compare_models(
        image_root=str(output_dir),
        model_names=[variant['label'] for variant in model_variants],
        seeds=seeds,
        prompts=prompts,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    default_config = RunConfig()

    parser = argparse.ArgumentParser(
        description="Generate with both pipelines and evaluate VISOR metrics."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./compare_output",
        help="Root directory for generated images and metrics."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=default_config.seeds,
        help="Seeds to use for generation (default: 5 seeds from config)."
    )
    parser.add_argument(
        "--models", nargs="+", type=str, default=None,
        help="Model keys to run (choose from: sd storm poisson). Defaults to config.models_to_run."
    )
    parser.add_argument(
        "--save_attention", action="store_true",
        help="Save attention maps at steps [0, 10, 17, 25, 30, 37, 42, 49]."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip generation, only run evaluation on existing images."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prompts = EVAL_PROMPTS  # reuse the canonical prompt list from eval_visor.py
    selected_models = args.models if args.models is not None else default_config.models_to_run
    model_variants = resolve_model_variants(selected_models)

    print(f"Selected models: {[variant['key'] for variant in model_variants]}")

    if not args.eval_only:
        generate_images(
            prompts=prompts,
            seeds=args.seeds,
            output_dir=output_dir,
            model_variants=model_variants,
            save_attention=args.save_attention,
        )

    evaluate(
        prompts=prompts,
        seeds=args.seeds,
        output_dir=output_dir,
        model_variants=model_variants,
    )


if __name__ == '__main__':
    main()
