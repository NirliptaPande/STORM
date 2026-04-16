# compare.py - Current Usage

This file documents the **current** behavior of `compare.py`.

## What compare.py does

`compare.py` runs in two phases:

1. **Generation**
     - Runs both models: `poisson_pipeline` and `pipeline`
     - Generates images for the VISOR prompt set (20 prompts)
     - Uses the requested seeds (default 5 seeds)
     - Optionally saves attention snapshots if `--save_attention` is passed

2. **Evaluation**
     - Calls `eval_visor.py` evaluation logic
     - Computes OA, VISOR_uncond, VISOR_cond, VISOR_1, VISOR_2, VISOR_3, VISOR_4
     - Prints a side-by-side comparison table

## Defaults

Running:

```bash
python compare.py
```

uses:

- **Prompts**: all 20 prompts from `eval_visor.py` (`EVAL_PROMPTS`)
- **Seeds**: `42 6143 7792 8892 9010`
- **Output dir**: `./compare_output`
- **Attention snapshots**: **OFF** by default

## Save attention snapshots

To save attention maps, you must pass:

```bash
python compare.py --save_attention
```

If you do not pass `--save_attention`, no attention files are written.

## Useful command examples

Run full default comparison:

```bash
python compare.py
```

Run only one seed for quick testing:

```bash
python compare.py --seeds 42
```

Run one-seed test and save attention:

```bash
python compare.py --seeds 42 --save_attention
```

Evaluate existing generated outputs only (skip generation):

```bash
python compare.py --eval_only
```

Use a different output folder:

```bash
python compare.py --output_dir ./my_compare_output --save_attention
```

## Output structure

Generated images:

```text
compare_output/
    poisson_pipeline/
        <prompt>/
            42.png
            6143.png
            ...
    pipeline/
        <prompt>/
            42.png
            6143.png
            ...
```

Evaluation JSON files:

```text
compare_output/
    eval_poisson_pipeline.json
    eval_pipeline.json
```

Attention snapshots (only with `--save_attention`):

```text
compare_output/
    poisson_pipeline/
        attention/
            <prompt>/
                seed_42/
                    step_000.png
                    step_010.png
                    step_017.png
                    step_025.png
                seed_6143/
                    ...
    pipeline/
        attention/
            <prompt>/
                seed_42/
                    step_000.png
                    step_010.png
                    step_017.png
                    step_025.png
```

## Note about interrupted runs

Exit code `130` means the run was interrupted (Ctrl+C / stop signal). In that case, only files produced before interruption will exist.
