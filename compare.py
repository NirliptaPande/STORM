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
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from config import RunConfig
from run import load_model, get_noun_indices_to_alter, run_on_prompt
from utils.ptp_utils import AttentionStore
from utils.attention_utils import AttentionConfig
from eval_visor import EVAL_PROMPTS, compare_models as eval_compare_models


MODEL_NAMES = ['poisson_pipeline', 'pipeline']


# ── Phase 1: Generation ────────────────────────────────────────────────────────

def generate_images(
    prompts: List[str],
    seeds: List[int],
    output_dir: Path,
    save_attention: bool = False,
):
    """
    Generate images for both pipeline versions.

    Images are saved as  <output_dir>/<model_name>/<prompt>/<seed>.png
    so that eval_visor.py can find them directly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}")
        print(f"  GENERATING: {model_name}")
        print(f"{'='*70}")

        model_output = output_dir / model_name

        attn_config = AttentionConfig(
            save=save_attention,
            save_steps=[0, 10, 17, 25],
            save_dir=(model_output / 'attention') if save_attention else None,
            display=False,
        )

        config = RunConfig(
            model_name=model_name,
            seeds=seeds,
            output_path=model_output,
            attention_config=attn_config,
        )

        model = load_model(config)

        for prompt in tqdm(prompts, desc=f"Prompts ({model_name})"):
            token_indices = get_noun_indices_to_alter(model, prompt)

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

    del model
    torch.cuda.empty_cache()

    print(f"\nGeneration complete. Images saved under {output_dir}/")


# ── Phase 2: Evaluation ────────────────────────────────────────────────────────

def evaluate(
    prompts: List[str],
    seeds: List[int],
    output_dir: Path,
):
    """
    Evaluate generated images using eval_visor.py's OWL-ViT detection and
    VISOR metric computation.  Prints a side-by-side comparison table.
    """
    eval_compare_models(
        image_root=str(output_dir),
        model_names=MODEL_NAMES,
        seeds=seeds,
        prompts=prompts,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate with both pipelines and evaluate VISOR metrics."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./compare_output",
        help="Root directory for generated images and metrics."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 6143, 7792, 8892, 9010],
        help="Seeds to use for generation (default: 5 seeds from config)."
    )
    parser.add_argument(
        "--save_attention", action="store_true",
        help="Save attention maps at steps [0, 10, 17, 25]."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip generation, only run evaluation on existing images."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prompts = EVAL_PROMPTS  # reuse the canonical prompt list from eval_visor.py

    if not args.eval_only:
        generate_images(
            prompts=prompts,
            seeds=args.seeds,
            output_dir=output_dir,
            save_attention=args.save_attention,
        )

    evaluate(
        prompts=prompts,
        seeds=args.seeds,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()
