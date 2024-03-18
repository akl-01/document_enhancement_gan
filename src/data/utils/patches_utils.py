import numpy as np
from numpy.typing import NDArray
from typing import Any

def get_patches(
    noisy_image: NDArray[np.uint8],
    gt_image: NDArray[np.uint8],
    stride: int = 192
) -> tuple[NDArray[Any], NDArray[Any]]:
    noisy_patches=[]
    gt_patches=[]

    h =  ((noisy_image.shape [0] // 256) + 1) * 256 
    w =  ((noisy_image.shape [1] // 256) + 1) * 256
    image_padding=np.ones((h,w))
    image_padding[:noisy_image.shape[0],:noisy_image.shape[1]] = noisy_image

    for j in range (0, h - 256, stride):
        for k in range (0, w - 256, stride):
            noisy_patches.append(image_padding[j:j+256,k:k+256])
    
    h =  ((gt_image.shape [0] // 256) + 1) * 256 
    w =  ((gt_image.shape [1] // 256 ) + 1) * 256
    image_padding=np.ones((h,w))
    image_padding[:gt_image.shape[0],:gt_image.shape[1]] = gt_image

    for j in range (0, h - 256, stride):
        for k in range (0, w - 256, stride):
            gt_patches.append(image_padding[j:j+256,k:k+256])  

    if not noisy_patches or not gt_patches:
        return np.array(noisy_patches).reshape(0, 256, 256), np.array(gt_patches).reshape(0, 256, 256)  

    return np.array(noisy_patches), np.array(gt_patches)

def patch_image(
    image: NDArray[np.uint8],
    h: int,
    w: int
) -> NDArray[Any]:
    patches = []
    patch_size = 256

    image_padding = np.ones((h, w))
    image_padding[:image.shape[0], :image.shape[1]] = image

    for j in range(0, h, patch_size):
        for k in range(0, w, patch_size): 
            patches.append(image_padding[j:j+patch_size, k:k+patch_size])
    
    if not patches:
        return np.array(patches).reshape(0, 256, 256)
    
    return np.array(patches)

def merge_patches(
    patches: NDArray[Any],
    h: int,
    w: int
) -> NDArray[Any]:
    image = np.zeros((1, h, w))
    patch_size = 256
    i = 0
    for j in range(0, h, patch_size):
        for k in range(0, w, patch_size):
            image[:, j:j+patch_size, k:k+patch_size] = patches[i]
            i += 1
    
    return image