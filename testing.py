from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torch

import sys, os
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions, evaluate_prediction
from PsfSimulator import PsfDataset, PsfSimulator
from scripts.plotting import PlotController
import PIL
from PIL import Image
import torchvision.transforms as T
import pickle
import time
import matplotlib.pyplot as plt
import tifffile
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, laplace

device = 'cuda'

### LOAD THE MODEL ###
backbone = resnet_fpn_backbone("resnet50", pretrained=False, trainable_layers=5)
kwargs = {"nms_thresh": 0.1, "detections_per_img": 200, "score_thresh": 0.1, "image_mean":[0,0,0], "image_std":[1,1,1]}

model = SubpixRCNN(backbone, num_classes=2, device=device, **kwargs)
model.to(device=device)
path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\notebooks\subpix_rcnn_models\2025-06-02_19-15-51\with_filters_long.pth"
model.load_state_dict(torch.load(path, map_location=device))

## FOR SIMULATING DATA AND TESTING IT ON THE MODEL ###
# seed = None
# num_datapoints = 1
# num_spots_min = 6
# num_spots_max = 6
# sigma_mean= 1.5
# sigma_std = 0.0
# snr_min = 5
# snr_max = 5
# snr_std = 0.0
# base_noise_min = 2000
# base_noise_max = 2000
# use_gauss_noise = False
# gauss_noise_std = 0
# use_perlin_noise = True
# perlin_min_max = (0.4,0.6)
# img_w = 64
# img_h = 64
# dataset = PsfDataset(seed, num_datapoints, num_spots_min, num_spots_max, sigma_mean, sigma_std,
#                       snr_min, snr_max, snr_std, base_noise_min, base_noise_max, use_gauss_noise,
#                       gauss_noise_std, use_perlin_noise, perlin_min_max, img_w, img_h)


# arr, image, target = dataset[0]

# image = move_data_to_device(image, device)
# target = move_data_to_device(target, device)
# model.eval()
# with torch.no_grad():
#     pred = model([image])
# pred = pred[0]
# pred = move_dict_to_cpu(pred)
# target = move_dict_to_cpu(target)
# print(target['positions'], target['true_snrs'])
# ev = evaluate_prediction(pred, target, 0.3)
# print(ev)
# # # bcimage = image.clone()
# # # bcimage = torch.clamp(bcimage-0.3, 0,1)
# # # bcimage = (bcimage - torch.min(bcimage)) / (torch.max(bcimage) - torch.min(bcimage))  # Normalize to [0, 1]

# # # bcimage = move_data_to_device(bcimage, device)

# img = torch.stack([image[0] * 3], dim=0) # Make it 3 channels

# PlotController(img, target, pred, 'eval', 1, 1, 1)

# # Plot array
# plt.imshow(image[2].cpu().numpy(), cmap='gray')
# plt.show()


# ### FOR TESTING MODEL ON REAL DATA ###
# Open the tif file
tif_path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\images\theo_frame_504.tif"
image = tifffile.imread(tif_path)
bcimage = image.copy()

# image -= np.min(image) 
# image /= np.max(image)  # min max normalization to [0, 1]

image = np.clip(image, None, 8000)
image = (image - np.min(image)) / (np.max(image) - np.min(image))  # min max normalization to [0, 1]
#image = (image - np.mean(image)) / (np.std(image))  # z-score normalization

blurred_channel = gaussian_filter(image, sigma=1.0)
lap = laplace(image)
array = np.stack([image, blurred_channel, lap], axis=0)

image_tensor = torch.tensor(array, dtype=torch.float32)
original_image_tensor = torch.stack([image_tensor[0] * 3], dim=0)  # Make it 3 channels

image_tensor = move_data_to_device(image_tensor, device)
model.eval()
with torch.no_grad():
    pred = model([image_tensor])
pred = pred[0]
pred = move_dict_to_cpu(pred)

# # # Save the prediction
# # save_path = r'd:\zeiss\Desktop\coding\Hilger\bachelor\pred_2_on_theo.pkl'
# # with open(save_path, 'wb') as f:
# #     pickle.dump(pred, f)


# # Increase contrast and brightness for visualization. Not sure how its done but it works
bcimage = np.clip(bcimage, 0, 4200)
bcimage = np.clip(bcimage - 2300, 0, None) # Increase brightness
bcimage /= np.max(bcimage)  # Normalize the image to [0, 1]
bcimage_tensor = torch.tensor(bcimage, dtype=torch.float32)
bcimage_tensor = bcimage_tensor.unsqueeze(0).repeat(3, 1, 1)  # [C, H, W]
bcimage_tensor = move_data_to_device(bcimage_tensor, device)

# with open(r"/Users/august/Desktop/bachelor/bachelor1/measurements/pred_2_on_theo.pkl", 'rb') as f:
#     pred = pickle.load(f)

PlotController(bcimage_tensor, None, pred, 'buttons', 1, 0, 1)



# # # Perlin Noise
# # # Patching
# # # performance vs density and snr
# # # Edge cases... extremely high background noise
# # # Run on some real data
# # # Size prediction
# # # Future prospects: Sammenlign med andre metoder
# # # Hvad ville jeg have gjort anderledes?
# # # Hvad hvis jeg havde mere tid? 3d... psf vs non-psf
# # # SPTnet bruger perlin noise
# # # Nikos spørger: Jaccard og IOU