from poisson_pipeline import PoissonPipeline
# everything else identical to run.py

import pprint
import pyrallis
import torch
from PIL import Image

from config import RunConfig
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import spacy

# reuse all helper functions from run.py unchanged
from run import (
    get_indices_to_alter,
    get_noun_indices_to_alter,
    unique_indices,
    fill_adj_indices,
    run_on_prompt,
)


def load_model(config: RunConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    stable_diffusion_version = (
        "stabilityai/stable-diffusion-2-1-base"
        if config.sd_2_1
        else "CompVis/stable-diffusion-v1-4"
    )

    # Only change from run.py: PoissonPipeline instead of StormPipeline
    stable = PoissonPipeline.from_pretrained(stable_diffusion_version).to(device)
    return stable


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    prompt_list = ["a cake to the left of a suitcase"]

    for prompt in prompt_list:
        token_indices = get_noun_indices_to_alter(stable, prompt)
        print(token_indices)
        for seed in config.seeds:
            g = torch.Generator("cuda").manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(
                prompt=prompt,
                model=stable,
                controller=controller,
                token_indices=token_indices,
                seed=g,
                config=config,
            )
            prompt_output_path = config.output_path / f"{prompt}"
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f"{seed}.png")


if __name__ == "__main__":
    main()