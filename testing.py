from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torch

import sys, os
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions, evaluate_prediction
from PsfSimulator import PsfDataset
from scripts.plotting import PlotController
from scripts.evaluating import test_model_fixed_snr, test_model_once_plot
import PIL
from PIL import Image
import torchvision.transforms as T
import pickle


device = 'cuda'
# seed = None
# num_datapoints = 1
# num_spots_min = 20
# num_spots_max = 20
# sigma_mean= 1.0
# sigma_std = 0.1
# snr_min = 2
# snr_max = 2
# snr_std = 0.0
# base_noise_min = 50
# base_noise_max = 100
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



backbone = resnet_fpn_backbone("resnet50", pretrained=True)
kwargs = {"nms_thresh": 0.1, "detections_per_img": None, "score_thresh": 0.8}
model = SubpixRCNN(backbone, num_classes=2, device=device, **kwargs)
model.to(device=device)
path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\subpix_rcnn_models\2025-05-22_18-47-47\first_long_run.pth"
model.load_state_dict(torch.load(path, map_location=device))
model.eval()



img_path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\screenshot.png"
pil_img = Image.open(img_path).convert("L")  # "L" = grayscale

# Downscale by 0.5x to revert retina doubling
w, h = pil_img.size
pil_img = pil_img.resize((w // 2, h // 2), Image.LANCZOS)

# Convert to 3 channels by repeating
to_tensor = T.ToTensor()
img_tensor = to_tensor(pil_img)  # shape: (1, H, W)
img_tensor = img_tensor.repeat(3, 1, 1)  # shape: (3, H, W)


with torch.no_grad():
    pred = model([img_tensor.to(device)])
pred = pred[0]
theo_pred = move_dict_to_cpu(pred)


save_path = r'd:\zeiss\Desktop\coding\Hilger\bachelor\theo_pred.pkl'

with open(save_path, 'wb') as f:
    pickle.dump(theo_pred, f)

PlotController(img_tensor, None, pred, 'buttons', 1,0,1)

# target = move_dict_to_cpu(target)
# print(target['positions'], target['true_snrs'])
# PlotController(image, target, pred, 'eval', 1, 1, 1)