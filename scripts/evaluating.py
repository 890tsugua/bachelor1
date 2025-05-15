from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes
from data_simulator import PsfDataset

# Instantiate the datasets.
num_spots_min = 2
num_spots_max = 6
sigma_mean = 2.0
sigma_std = 0.0
snr_mean = 10
snr_std = 0.2
base_noise_min = 50
base_noise_max = 150
use_gauss_noise = True
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
                            40,
                            num_spots_min, num_spots_max,
                            sigma_mean, sigma_std,
                            snr_mean, snr_std,
                            base_noise_min, base_noise_max,
                            use_gauss_noise, gauss_noise_std,
                            img_w, img_h
                          )

backbone = resnet_fpn_backbone("resnet50", pretrained=True)
kwargs = {"nms_tresh": 0.1, "detections_per_img": 6}
model = SubpixRCNN(backbone, num_classes=2, device='cuda', **kwargs)
model.to(device='cuda')
path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\subpix_rcnn_models\2025-05-15_15-43-05\testmodel_noise_added.pth"
model.load_state_dict(torch.load(path))


device = 'cuda'
model.eval()


# ...existing code to load model...

while True:
    idx = input("Enter validation image index (or 'q' to quit): ")
    if idx.lower() == 'q':
        break
    idx = int(idx)
    img, tar = valid_dataset[idx]
    img = move_data_to_device(img, device)
    tar = move_data_to_device(tar, device)
    with torch.no_grad():
        out = model([img])[0]
    move_dict_to_cpu(tar)
    move_dict_to_cpu(out)
    plot_image_boxes(img, tar, out, True, True)
    print(out)
    print(tar)




img, tar = valid_dataset[2]
img = move_data_to_device(img,device)
tar = move_data_to_device(tar,device)
imgs = []
imgs.append(img)
with torch.no_grad():
  out = model(imgs)

out = out[0]
move_dict_to_cpu(tar)
move_dict_to_cpu(out)
plot_image_boxes(img,tar,out,True,True)
print(out)
print(tar)