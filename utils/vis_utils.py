import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch

import os
from utils import ptp_utils
from utils.ptp_utils import AttentionStore, aggregate_attention


def show_cross_attention(prompt: str,
                        attention_store: AttentionStore,
                        tokenizer,
                        indices_to_alter: List[int],
                        res: int,
                        from_where: List[str],
                        select: int = 0,
                        orig_image=None,
                        time_step = None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    images = []
    flattened_list = [item for sublist in indices_to_alter for item in sublist if item is not None]
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in flattened_list:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    if not images:
        return None

    grid = ptp_utils.view_images(images, num_rows=1, display_image=False)
    if time_step is not None:
        os.makedirs("./cross_attention", exist_ok=True)
        grid.save(os.path.join("./cross_attention", f"step_{time_step}.png"))
    return grid


def visualize_cross_attention_maps(prompt: str,
                                   attention_store: AttentionStore,
                                   tokenizer,
                                   res: int = 16,
                                   from_where: List[str] = ["up", "down", "mid"],
                                   select: int = 0,
                                   token_indices: List[int] = None,
                                   orig_image: Image.Image = None,
                                   save_path: str | None = None,
                                   display_image: bool = True) -> Image.Image:
    """
    Build a token-wise cross-attention visualization grid.

    If token_indices is provided, only those token maps are included.
    If orig_image is provided, each map is rendered as a heatmap overlay.
    Returns a PIL image grid, or None when no maps are available.
    """
    tokens = tokenizer.encode(prompt)
    decoded_tokens = [tokenizer.decode(int(t)).strip() for t in tokens]
    attention_maps = aggregate_attention(
        attention_store=attention_store,
        res=res,
        from_where=from_where,
        is_cross=True,
        select=select,
    ).detach().cpu()

    if token_indices is None:
        token_indices = list(range(min(len(tokens), attention_maps.shape[-1])))

    images = []
    tile_size = res ** 2
    for idx in token_indices:
        if idx < 0 or idx >= attention_maps.shape[-1]:
            continue

        attn_map = attention_maps[:, :, idx]
        if orig_image is not None:
            rendered = show_image_relevance(attn_map, orig_image, relevnace_res=res)
            rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        else:
            attn_np = attn_map.numpy()
            denom = attn_np.max() - attn_np.min()
            if denom < 1e-8:
                denom = 1.0
            attn_np = (attn_np - attn_np.min()) / denom
            rendered = cv2.applyColorMap(np.uint8(255 * attn_np), cv2.COLORMAP_VIRIDIS)
            rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        rendered = np.array(Image.fromarray(rendered).resize((tile_size, tile_size)))
        label = decoded_tokens[idx] if idx < len(decoded_tokens) else str(idx)
        rendered = ptp_utils.text_under_image(rendered, f"{idx}: {label}")
        images.append(rendered)

    if not images:
        return None

    num_rows = max(1, int(np.sqrt(len(images))))
    grid = ptp_utils.view_images(images, num_rows=num_rows, display_image=display_image)

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        grid.save(save_path)

    return grid


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
