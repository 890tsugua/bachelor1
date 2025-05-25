from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torch

import sys, os
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions, evaluate_prediction
from PsfSimulator import PsfDataset, PsfSimulator
from scripts.plotting import PlotController
from scripts.evaluating import test_model_fixed_snr, test_model_once_plot
import PIL
from PIL import Image
import torchvision.transforms as T
import pickle
import time
import matplotlib.pyplot as plt
import tifffile
import numpy as np

device = 'cuda'

### LOAD THE MODEL ###
backbone = resnet_fpn_backbone("resnet50", pretrained=True)
kwargs = {"nms_thresh": 0.1, "detections_per_img": None, "score_thresh": 0.9}
model = SubpixRCNN(backbone, num_classes=2, device=device, **kwargs)
model.to(device=device)
path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\subpix_rcnn_models\2025-05-24_13-46-15\second_long_run.pth"
model.load_state_dict(torch.load(path, map_location=device))

### FOR SIMULATING DATA AND TESTING IT ON THE MODEL ###
# seed = None
# num_datapoints = 1
# num_spots_min = 100
# num_spots_max = 100
# sigma_mean= 1.0
# sigma_std = 0.1
# snr_min = 3
# snr_max = 3
# snr_std = 0.0
# base_noise_min = 5000
# base_noise_max = 5000
# use_gauss_noise = False
# gauss_noise_std = 0.02
# use_perlin_noise = False
# perlin_min_max = (0.4, 0.6)
# img_w = 64
# img_h = 64
# dataset = PsfDataset(seed, num_datapoints, num_spots_min, num_spots_max, sigma_mean, sigma_std,
#                       snr_min, snr_max, snr_std, base_noise_min, base_noise_max, use_gauss_noise,
#                       gauss_noise_std, use_perlin_noise, perlin_min_max, img_w, img_h)
# image, target = dataset[0]
# image = move_data_to_device(image, device)
# target = move_data_to_device(target, device)
# model.eval()
# with torch.no_grad():
#     pred = model([image.to(device)])
# pred = pred[0]
# pred = move_dict_to_cpu(pred)
# target = move_dict_to_cpu(target)
# print(target['positions'], target['true_snrs'])
# PlotController(image, target, pred, 'eval', 1, 1, 1)


### FOR TESTING MODEL ON REAL DATA ###
# Open the tif file
tif_path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\theo_frame_504.tif"
image = tifffile.imread(tif_path)
bcimage = image.copy()
#image = np.clip(image, 0, 10000)
image /= np.max(image)  # Normalize the image to [0, 1]

image_tensor = torch.tensor(image, dtype=torch.float32)
# Make it three channels
image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)  # [C, H, W]
image_tensor = move_data_to_device(image_tensor, device)
model.eval()
with torch.no_grad():
    pred = model([image_tensor])
pred = pred[0]
pred = move_dict_to_cpu(pred)

# Save the prediction
save_path = r'd:\zeiss\Desktop\coding\Hilger\bachelor\pred_2_on_theo.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(pred, f)


# Increase contrast and brightness for visualization. Not sure how its done but it works
bcimage = np.clip(bcimage, 0, 4200)
bcimage /= np.max(bcimage)  # Normalize the image to [0, 1]
bcimage_tensor = torch.tensor(bcimage, dtype=torch.float32)
bcimage_tensor = bcimage_tensor.unsqueeze(0).repeat(3, 1, 1)  # [C, H, W]
bcimage_tensor = move_data_to_device(bcimage_tensor, device)

PlotController(bcimage_tensor, None, pred, 'buttons', 1, 0, 1)



# Perlin Noise
# Patching
# performance vs density and snr
# Edge cases... extremely high background noise
# Run on some real data
# Size prediction
# Future prospects: Sammenlign med andre metoder
# Hvad ville jeg have gjort anderledes?
# Hvad hvis jeg havde mere tid? 3d... psf vs non-psf
# SPTnet bruger perlin noise
# Nikos spørger: Jaccard og IOU