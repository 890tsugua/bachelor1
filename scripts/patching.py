import torch


def image_to_patches(image, patch_size=64, overlap=4):
    """
    Splits an image into patches of a given size.

    Args:
        image (torch.Tensor): The input image to be split into patches. [C, H, W]
        patch_size (tuple): The size of the patches (height, width).

    Returns:
        list of tensors: a list of patches extracted from the input image.
    """
    # Calculate number of patches in each dimension
    img_h, img_w = image.shape[1], image.shape[2]
    median = torch.median(image)
    
    # Compute padding
    stride = patch_size - overlap
    pad_w = (stride - (img_w - overlap) % stride) % stride
    pad_h = (stride - (img_h - overlap) % stride) % stride

    # Apply padding (right, bottom)
    if pad_w > 0 or pad_h > 0:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=median)
        img_h = image.shape[1]
        img_w = image.shape[2]

    patches = []
    patch_origins = []

    for j in range(0,img_h-patch_size+1, stride):
        for i in range(0, img_w-patch_size+1, stride):
            patch = image[:, j:j+patch_size, i:i+patch_size]
            patches.append(patch)
            patch_origins.append((i,j))

    return patches, patch_origins, (pad_w, pad_h)

import torch

def patches_to_image(patches, patch_origins, image_shape, patch_size=64, pad=(0, 0)):
    """
    Reconstructs an image from patches.

    Args:
        patches (list of tensors): The list of patches to be reconstructed.
        patch_origins (list of tuples): The origin coordinates of each patch.
        image_shape (tuple): The shape of the original image (C, H, W).
        patch_size (int): The size of the patches.

    Returns:
        torch.Tensor: The reconstructed image.
    """

    c, img_h, img_w = image_shape
    pad_w, pad_h = pad[0], pad[1]

    recon = torch.zeros((c, img_h+pad_h, img_w+pad_w), dtype=patches[0].dtype)
    
    for (x,y), patch in zip(patch_origins, patches):
        recon[:,y:y+patch_size,x:x+patch_size] = patch

    # Unpad the image
    recon = recon[:, 0:img_h, 0:img_w]
    
    return recon

