# Data simulator

# Basically bare smid ind fra datasimulator...

import numpy as np
import matplotlib.pyplot as plt

# Import PyTorch dependencies
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchtnt.utils import get_module_summary
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.v2 import functional as TF

""" NOISE GENERATION """
def add_poisson(array):
    return np.random.poisson(array).astype(np.float32)

def make_background(base: float = 100, use_gauss_noise: bool = True, sigma: float = 10, pad: int = 20, img_w=512, img_h=512) -> np.float32:
    img_w += 2*pad
    img_h += 2*pad
    if use_gauss_noise:
        array = np.clip(np.random.normal(base,sigma,(img_h,img_w)), 0, None)
    else:
        array = np.full((img_h,img_w),int(base))
    return array.astype(np.float32)

""" PSF GENERATION """
def generate_positions(seed: int, num_spots: int, image_width: int, image_height: int):
    # Set the seed for reproducibility
    np.random.seed(seed)
    x = (image_width) * np.random.rand(num_spots)
    y = (image_height) * np.random.rand(num_spots)
    positions = np.column_stack((x,y))
    return positions

def calculate_signal_value(background_mean, snr):
    return snr * np.sqrt(background_mean)

def make_psf(sigma, intensity, subpos: tuple): 
    
    radius = int(np.ceil(3*sigma))
    array = np.zeros((radius*2+1,radius*2+1))
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = xx**2 + yy**2 <= radius**2
    subpos = (subpos[0] - 0.5, subpos[1] - 0.5)

    r2 = (xx-subpos[0])**2 + (yy-subpos[1])**2

    gauss = lambda r2: np.exp(-r2/(2*sigma**2)) # 2D gauss function
    signals = gauss(r2)
    # Sum of all pixel values is not always the same... Need to normalize?

    for i in array[mask]:
        array[mask] = signals[mask] * intensity
    return array.astype(np.float32), mask


""" ADDING PSFS TO AN ARRAY AND DRAWING PICTURE """
def calculate_background_mean(input_array, sigma, x, y):
    start_x, start_y = int(np.ceil(x-sigma*3*2)), int(np.ceil(y-sigma*3*2))  # Where to round up??
    end_x, end_y = int(np.ceil(start_x + sigma*3*2)), int(np.ceil(start_y + sigma*3*2))
    mean = np.mean(input_array[max(0,start_y):end_y, max(0,start_x):end_x]) # Minimum 0 so that np doesnt access outside array
    return mean

def round_position(pos):
    sub_offset = pos % 1
    shift = sub_offset > 0.5
    rounded_pos = pos.astype(int) + shift
    return rounded_pos

def draw_array(positions, sigmas, input_array: np.float32, snrs, pad):
    h, w = input_array.shape
    masks = np.zeros((positions.shape[0],h,w)) # Initialize empty masks

    for i, ((x, y), sigma, snr) in enumerate(zip(positions, sigmas, snrs)):
        subpos = (x%1, y%1)
        mean = calculate_background_mean(input_array,sigma,pad+x,pad+y)
        psf, mask = make_psf(sigma, calculate_signal_value(mean, snr), subpos)
        x, y = x.astype(int), y.astype(int)
        start_x, start_y = pad+x - int(np.ceil(sigma*3)), pad+y - int(np.ceil(sigma*3))  # Insert PSF input_arrays from their top left corner.
        input_array[start_y:start_y + psf.shape[0], start_x:start_x + psf.shape[1]] += psf

    # Remove padding and clip
    unpadded_array = np.clip(input_array[pad:-pad, pad:-pad],0,255)
    return unpadded_array.astype(np.uint8)

""" MAKING THE TARGETS """
def make_targets(positions, sigmas, masks, img_w, img_h):
    """
    Create targets for the mdoel. Right now boxes are float values. 2x2 sizes."""
    
    labels = torch.ones(positions.shape[0]).to(torch.int64)

    bboxes = []
    #ps = np.floor(positions) # Round down positions.
    for i in range(positions.shape[0]):
        #p = ps[i]
        p = positions[i]
        #box = [p[0]-1,p[1]-1,p[0]+1,p[1]+1] #xyxy format. Bottom left corner, top right corner.
        box = [p[0]-1,p[1]-1,p[0]+1,p[1]+1] #xyxy format. Bottom left corner, top right corner.
        bboxes.append(box)
    bboxes = torch.clamp(torch.tensor(bboxes).to(torch.float32), min=0, max=img_w)

    targets = {
        'boxes': bboxes,
        'labels': labels, #torch.Tensor([class_names.index(label) for label in labels]),
        'positions': torch.tensor(positions).to(torch.float32),
        'subpixel_positions': torch.tensor(positions%1).to(torch.float32)#torch.tensor(positions).to(torch.float32) / img_w # Between 0 and 1. For now assume square pictures...
    }
    return targets

def make_one_data(
    seed: int = 0,
    num_spots: int = 5,
    sigma_mean: float = 2,
    sigma_std: float = 0.3,
    snr_mean: float = 10,
    snr_std: float = 0.3,
    base_noise: int = 100,
    use_gauss_noise: bool = True,
    gauss_noise_std: float = 15,
    img_w: int = 64,
    img_h: int = 64):
    
    if num_spots == 0:
            raise ValueError("Number of spots must be more than 0")

    positions = generate_positions(seed, num_spots, img_w, img_h)
    sigmas = np.clip(np.random.normal(sigma_mean,sigma_std,num_spots),0,None)
    snrs = np.clip(np.random.normal(snr_mean,snr_std,num_spots),0,None)
    pad = int(np.ceil(3*np.max(sigmas)*2))

    background = make_background(base_noise,False,gauss_noise_std,pad,img_w, img_h)  # FALSE ADDED
    image_array = draw_array(positions, sigmas, background, snrs, pad)       
    #image_array = add_poisson(image_array)                        
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    image = torch.tensor(np.stack([image_array.astype(np.float32) / 255] * 3, axis=0))  

    targets = make_targets(positions, None, None, img_w, img_h)

    return image, targets

class PsfDataset(Dataset):
    """
    This class represents a PyTorch Dataset for a collection of images and their annotations.
    The class is designed to load images along with their corresponding segmentation masks, bounding box annotations, and labels.
    """
    def __init__(self, seed, num_datapoints, 
                    num_spots_min,
                    num_spots_max,
                    sigma_mean,
                    sigma_std,
                    snr_mean,
                    snr_std,
                    base_noise_min,
                    base_noise_max,
                    use_gauss_noise,
                    gauss_noise_std,
                    img_w, 
                    img_h):
        """
        Constructor for the HagridDataset class.

        Parameters:
        img_keys (list): List of unique identifiers for images.
        annotation_df (DataFrame): DataFrame containing the image annotations.
        img_dict (dict): Dictionary mapping image identifiers to image file paths.
        class_to_idx (dict): Dictionary mapping class labels to indices.
        transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()
        
        self._len = num_datapoints
        self._seed = seed

        self._num_spots_min = num_spots_min
        self._num_spots_max = num_spots_max
        self._sigma_mean = sigma_mean
        self._sigma_std = sigma_std
        self._snr_mean = snr_mean
        self._snr_std = snr_std
        self._base_noise_min = base_noise_min
        self._base_noise_max = base_noise_max
        self._use_gauss_noise = use_gauss_noise
        self._gauss_noise_std = gauss_noise_std
        self._img_w = img_w
        self._img_h = img_h
        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: The number of items in the dataset.
        """
        return self._len
        
    def __getitem__(self, index):

        #seed = self._seed * self._len + index # Make sure there is a different but unique seed for each data point across epochs. No need...
        seed = np.random.randint(1,1000000)
        num_spots = np.random.randint(self._num_spots_min, self._num_spots_max+1)
        base_noise = np.random.randint(self._base_noise_min, self._base_noise_max+1)

        image, target = make_one_data(  seed,
                                        num_spots,
                                        self._sigma_mean,
                                        self._sigma_std,
                                        self._snr_mean,
                                        self._snr_std,
                                        base_noise,
                                        self._use_gauss_noise,
                                        self._gauss_noise_std,
                                        self._img_w, 
                                        self._img_h
                                        )
        
        return image, target
