import schemas as _schemas
import os
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
import base64
from gfpgan import GFPGANer
from codeformer import CodeFormer
import torch


def enhance_image_gfpgan(image: np.ndarray, background_enhancement: bool = True, upscale: int = 2):
    try:
        restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=upscale, arch='clean', channel_multiplier=2, bg_upsampler=None)

        cropped_faces, restored_faces, restored_image = restorer.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5
        )
        return restored_image
    except Exception as e:
        print(f"Error in enhance_image_gfpgan: {e}")
        return image


def enhance_image_codeformer(image: np.ndarray, background_enhancement: bool = True, upscale: int = 2):
    try:
        net = CodeFormer(model_path='https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth', w=0.5, device='cpu')
        _, restored_image = net.process(image)
        return restored_image
    except Exception as e:
        print(f"Error in enhance_image_codeformer: {e}")
        return image


def read_image_from_base64(encoded_string: str) -> np.ndarray:
    try:
        img = Image.open(BytesIO(base64.b64decode(encoded_string)))
        return np.array(img)
    except UnidentifiedImageError as e:
        raise ValueError("Invalid image format") from e
    except Exception as e:
        raise Exception(f"Error in read_image_from_base64: {e}")


async def enhance(enhanceBase: _schemas._EnhanceBase) -> Image:
    init_image = read_image_from_base64(enhanceBase.encoded_base_img[0])

    method = enhanceBase.method
    background_enhancement = enhanceBase.background_enhancement
    upscale = enhanceBase.upscale

    if method == 'gfpgan':
        restored_image = enhance_image_gfpgan(init_image, background_enhancement, upscale)
    elif method == 'codeformer':
        restored_image = enhance_image_codeformer(init_image, background_enhancement, upscale)
    elif method == 'RestoreFormer':
        print('RestoreFormer is not yet supported')
        restored_image = init_image
    else:
        raise ValueError(f"Invalid method: {method}")

    final_image = Image.fromarray(restored_image.astype(np.uint8))
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue())
    
    return encoded_img
        
