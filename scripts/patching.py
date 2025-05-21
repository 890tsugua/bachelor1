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

def patches_to_image(patches, patch_origins, image_shape, patch_size=64, overlap=4, pad=(0, 0)):
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
    if overlap % 2 != 0:
        raise ValueError("Overlap must be even.")

    c, img_h, img_w = image_shape
    pad_w, pad_h = pad[0], pad[1]

    patches_in_row = ((img_w+pad_w)-patch_size) // (patch_size - overlap) +1
    patches_in_column = ((img_h+pad_h)-patch_size) // (patch_size - overlap) +1
    recon = torch.zeros((c, img_h+pad_h, img_w+pad_w), dtype=patches[0].dtype)
    
    print(patches_in_row, patches_in_column)

    # Insert very first patch
    recon[:,0:patch_size, 0:patch_size] = patches.pop(0)[:,0:patch_size, 0:patch_size]
    patch_origins.pop(0)

    # Insert the rest of the patches in the first row
    for i in range(1, patches_in_row):
        x, y = patch_origins.pop(0)
        patch = patches.pop(0)
        recon[:, 0:patch_size, x+overlap:x + patch_size] = patch[:,0:patch_size, overlap:patch_size]
    
    # Insert the rest of the patches
    for row in range(1, int(patches_in_column)):
        # Insert very first patch in each row
        x, y = patch_origins.pop(0)
        recon[:,y+overlap:y+patch_size, 0:patch_size] += patches.pop(0)[:,overlap:patch_size, 0:patch_size]
        
        # Insert the rest of the patches in the row
        for i in range(1, patches_in_row):
            x, y = patch_origins.pop(0)
            patch = patches.pop(0)
            recon[:, y+overlap:y + patch_size, x+overlap:x + patch_size] += patch[:,overlap:patch_size, overlap:patch_size]
    
    # Unpad the image
    recon = recon[:, 0:img_h, 0:img_w]
    
    return recon


img = torch.randn(3, 123, 137)  # RGB image
patches = image_to_patches(img, patch_size=64, overlap=4)
print(len(patches))  # Should be num_patches_h * num_patches_w
print(patches[0].shape)  # torch.Size([3, 64, 64])

# Import a jpg as a tensor [3,H,W]
from PIL import Image
import torchvision.transforms as T

from torchvision.transforms import ToPILImage

img_path = '/Users/august/Desktop/bachelor/bachelor1/Monkey-Main-1280x720.jpg'
pil_img = Image.open(img_path).convert('RGB')

pil_img.show()

# Convert to tensor [3, H, W]
to_tensor = T.ToTensor()
img_tensor = to_tensor(pil_img)
patches, origins, pad = image_to_patches(img_tensor, patch_size=64, overlap=8)
recon = patches_to_image(patches, origins, img_tensor.shape, patch_size=64, overlap=8, pad=pad)
recon_pil = ToPILImage()(recon)
recon_pil.show()
print(pad)
print(origins)


l = [1,2,3,4]
print(l.pop(0))