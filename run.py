import pprint
from typing import List, Optional, Tuple, Union

import pyrallis
import torch
from PIL import Image

from config import RunConfig
from utils import ptp_utils
from utils.ptp_utils import AttentionStore
from utils.attention_utils import AttentionConfig
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import spacy
import importlib
from pathlib import Path

def load_pipeline_from_config(config: RunConfig):
    module_name = config.model_name
    module = importlib.import_module(module_name)
    pipeline_class = getattr(module, 'StormPipeline')
    return pipeline_class

def load_model(config: RunConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"

    StormPipeline = load_pipeline_from_config(config)
    if device.type == 'cuda':
        stable = StormPipeline.from_pretrained(
            stable_diffusion_version,
            torch_dtype=torch.float16,
        ).to(device)
    else:
        stable = StormPipeline.from_pretrained(stable_diffusion_version).to(device)
    stable.use_distance = False
    return stable

def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                        for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                        if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                        "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def unique_indices(indices: List[int]) -> List[int]:
    if not indices:
        return []
    return [current for current, next_val in zip(indices, indices[1:] + [None])
        if next_val is None or current + 1 != next_val]


def fill_adj_indices(noun_indices: List[int], adj_indices: List[int]) -> Tuple[List[int], List[Optional[int]]]:
    result_adj_indices: List[int | None] = [None] * len(noun_indices)

    if not adj_indices:
        return noun_indices, result_adj_indices

    adj_tensor = torch.tensor(adj_indices)
    noun_tensor = torch.tensor(noun_indices)
    differences = noun_tensor.unsqueeze(1) - adj_tensor.unsqueeze(0)

    for i, adj_idx_val in enumerate(adj_indices):
        valid_diffs = differences[:, i]
        positive_diff_mask = valid_diffs > 0
        if positive_diff_mask.any():
            closest_noun_pos_in_noun_tensor = torch.nonzero(positive_diff_mask).squeeze(1)[0].item()
            result_adj_indices[closest_noun_pos_in_noun_tensor] = adj_idx_val

    return noun_indices, result_adj_indices



def get_noun_indices_to_alter(stable, prompt: str) -> Tuple[List[int], List[Optional[int]]]:
    tokens = stable.tokenizer(prompt)['input_ids']
    token_idx_to_word = {idx: stable.tokenizer.decode(t).strip() for idx, t in enumerate(tokens) if 0 < idx < len(tokens) - 1}
    pprint.pprint(token_idx_to_word)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(prompt)

    nouns = {token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.text not in ['left', 'right']}
    adjs = {token.text for token in doc if token.pos_ in ['ADJ']}

    noun_indices = [idx for idx, word in token_idx_to_word.items() if word in nouns]
    adj_indices = [idx for idx, word in token_idx_to_word.items() if word in adjs]

    unique_noun_indices = unique_indices(noun_indices)
    unique_adj_indices = unique_indices(adj_indices)

    token_noun_indices, token_adj_indices = fill_adj_indices(unique_noun_indices, unique_adj_indices)
    return token_noun_indices, token_adj_indices
            

def run_on_prompt(prompt: Union[str, List[str]],
                model,
                controller: AttentionStore,
                token_indices: Union[List[List[Optional[int]]], Tuple[List[int], List[Optional[int]]]],
                seed: torch.Generator,
                config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    
    # Build attention config from RunConfig
    prompt_for_save = prompt if isinstance(prompt, str) else prompt[0]
    seed_for_save = seed.initial_seed() if hasattr(seed, "initial_seed") else None
    save_dir = config.attention_config.save_dir
    if save_dir is not None:
        save_dir = save_dir / prompt_for_save
        if seed_for_save is not None:
            save_dir = save_dir / f"seed_{seed_for_save}"

    attention_config = AttentionConfig(
        save=config.attention_config.save,
        save_steps=list(config.attention_config.save_steps),
        save_dir=save_dir,
        token_indices=config.attention_config.token_indices,
        display=config.attention_config.display,
    )
    
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    use_distance=getattr(model, 'use_distance', False),
                    attention_config=attention_config)
    image = outputs.images[0]
    # Cleanup to prevent memory leaks
    del outputs, attention_config
    torch.cuda.empty_cache()
    return image
    
@pyrallis.wrap()
def main(config: RunConfig):
    """
    Main function to run STORM with configurable pipeline.
    
    Usage examples:
    1. Run with default poisson_pipeline:
       python run.py --model_name poisson_pipeline
    
    2. Run with standard pipeline:
       python run.py --model_name pipeline
    
    3. Save attention maps at steps 0, 10, 20:
       python run.py --model_name poisson_pipeline --attention_config.save true --attention_config.save_steps [0,10,20]
    
    4. Display attention maps inline:
       python run.py --model_name pipeline --attention_config.display true
    """
    
    # Load the model (pipeline or poisson_pipeline based on config.model_name)
    stable = load_model(config)
    print(f"\n{'='*60}")
    print(f"Running with model: {config.model_name}")
    print(f"Attention config: save={config.attention_config.save}, steps={config.attention_config.save_steps}, display={config.attention_config.display}")
    print(f"{'='*60}\n")
    
    prompt_list = [
        "a cake to the left of a suitcase",
        "a bottle to the left of a suitcase",
        "a dog to the left of a wine glass",
        "an elephant to the left of a laptop",
        "a spoon to the left of a teddy bear",
        "a bottle to the right of a suitcase",
        "an elephant to the right of a clock",
        "a bicycle to the right of a bear",
        "a train to the right of a vase",
        "a car to the right of a traffic light",
        "a horse above a cat",
        "an orange above a cat",
        "a refrigerator above a couch",
        "a cat above a pillow",
        "a suitcase above a car",
        "a handbag below an umbrella",
        "a skateboard below an apple",
        "a giraffe below a sheep",
        "a potted plant below a bench",
        "a sports ball below a sandwich",
    ]

    for prompt in prompt_list:
        token_indices = get_noun_indices_to_alter(stable, prompt)
        print(f"Prompt: {prompt} | Tokens: {token_indices}")
        
        for seed in config.seeds:
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=prompt,
                                model=stable,
                                controller=controller,
                                token_indices=token_indices,
                                seed=g,
                                config=config)
            
            # Save image to output directory
            # Layout: <output_path>/<model_name>/<prompt>/<seed>.png
            # Compatible with eval_visor.py expectations
            prompt_output_path = config.output_path / config.model_name / f'{prompt}'
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f'{seed}.png')
            print(f"  Saved: {prompt_output_path / f'{seed}.png'}")

if __name__ == '__main__':
    main()
