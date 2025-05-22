from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torch

import sys, os
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions, evaluate_prediction
from PsfSimulator import PsfDataset
from scripts.plotting import PlotController
from scripts.evaluating import test_model_fixed_snr, test_model_once_plot

# Load a png image

from PIL import Image
import torchvision.transforms as T

# Load the image as grayscale
img_path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\screenshot.png"
pil_img = Image.open(img_path).convert("L")  # "L" = grayscale
w, h = pil_img.size
pil_img = pil_img.resize((w // 2, h // 2), Image.LANCZOS)
# Convert to 3 channels by repeating
to_tensor = T.ToTensor()
img_tensor = to_tensor(pil_img)  # shape: (1, H, W)
img_tensor = img_tensor.repeat(3, 1, 1)     # shape: (3, H, W)
print(img_tensor.shape)

device = 'cuda'
img_tensor = img_tensor.to(device=device)
backbone = resnet_fpn_backbone("resnet50", pretrained=True)

kwargs = {"nms_thresh": 0.2, "detections_per_img": None, "score_thresh":0.8, "device":torch.device(device)}
model = SubpixRCNN(backbone, num_classes=2, **kwargs)
model.to(device=device)
path = r"D:\zeiss\Desktop\coding\Hilger\bachelor\subpix_rcnn_models\2025-05-20_16-05-31\first_real_run.pth"
model.load_state_dict(torch.load(path, map_location=device))
model.eval()
print("ding")
with torch.no_grad():
    out = model([img_tensor])
print("dong")
#eval = evaluate_predictions(out,[tar], 0.5)
#print(eval)
out = out[0]
print(out)
move_dict_to_cpu(out)
#move_dict_to_cpu(tar)


from scripts.plotting import PlotController
# Create an instance of the PlotController
plot_controller = PlotController(img_tensor, None, out, 'buttons', 1, 0, 1)
