from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torch

import sys, os
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions, evaluate_prediction
from data_simulator import PsfDataset
from scripts.plotting import PlotController
from scripts.evaluating import test_model_fixed_snr, test_model_once_plot


# Instantiate the datasets.
num_spots_min = 5
num_spots_max = 5
sigma_mean = 1.0
sigma_std = 0.0
snr_mean = 8
snr_std = 0.0
base_noise_min = 100
base_noise_max = 100
use_gauss_noise = False
gauss_noise_std = 10
img_w, img_h = 128,128

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
img2, tar2 = valid_dataset[2]
print(img.shape)

device = 'cpu'
img = move_data_to_device(img, device)
tar = move_data_to_device(tar, device)
img2 = move_data_to_device(img2, device)
tar2 = move_data_to_device(tar2, device)


backbone = resnet_fpn_backbone("resnet50", pretrained=True)

kwargs = {"nms_thresh": 0.2, "detections_per_img": 5, "score_thresh":0.9, "device":torch.device(device)}
model = SubpixRCNN(backbone, num_classes=2, **kwargs)
model.to(device=device)
path = r"/Users/august/Desktop/bachelor/bachelor1/first_real_run.pth"
model.load_state_dict(torch.load(path, map_location=device))
model.eval()
images = [img, img2]
with torch.no_grad():
    out = model([img])

eval = evaluate_predictions(out,[tar], 0.5)
print(eval)
out = out[0]
print(out)
move_dict_to_cpu(out)
move_dict_to_cpu(tar)


from scripts.plotting import PlotController
# Create an instance of the PlotController
plot_controller = PlotController(img, tar, out, 'eval', 1, 1, 1)



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
