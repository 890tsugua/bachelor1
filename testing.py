from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch

import sys, os
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions
from data_simulator import PsfDataset
from scripts.plotting import PlotController
from scripts.evaluating import test_model_fixed_snr, test_model_once_plot

# Instantiate the datasets.
num_spots_min = 10
num_spots_max = 10
sigma_mean = 1.0
sigma_std = 0.0
snr_mean = 3
snr_std = 0.0
base_noise_min = 10
base_noise_max = 100
use_gauss_noise = False
gauss_noise_std = 10
img_w, img_h = 64, 64

# Instantiate the datasets.
train_dataset = PsfDataset( 1,
                            200,
                            num_spots_min, num_spots_max,
                            sigma_mean, sigma_std,
                            snr_mean, snr_std,
                            base_noise_min, base_noise_max,
                            use_gauss_noise, gauss_noise_std,
                            img_w, img_h
                          )

valid_dataset = PsfDataset( 2,
                            5,
                            num_spots_min, num_spots_max,
                            sigma_mean, sigma_std,
                            snr_mean, snr_std,
                            base_noise_min, base_noise_max,
                            use_gauss_noise, gauss_noise_std,
                            img_w, img_h
                          )

img, tar = valid_dataset[1]

device = 'cuda'
img = move_data_to_device(img, device)
tar = move_data_to_device(tar, device)

backbone = resnet_fpn_backbone("resnet50", pretrained=True)
kwargs = {"nms_tresh": 0.1, "detections_per_img": 20}
model = SubpixRCNN(backbone, num_classes=2, device=device, **kwargs)
model.to(device=device)
path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\subpix_rcnn_models\2025-05-20_16-05-31\first_real_run.pth"
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

with torch.no_grad():
    out = model([img])
eval = evaluate_predictions(out, [tar], 0.5)
print(eval)
out = out[0]
move_dict_to_cpu(out)
move_dict_to_cpu(tar)


from scripts.plotting import PlotController
# Create an instance of the PlotController
plot_controller = PlotController(img, tar, out, 'eval', 1, 1)