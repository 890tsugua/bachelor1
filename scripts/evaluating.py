from torchvision.models.detection import resnet_fpn_backbone
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu
import torch
from data_simulator import PsfDataset, plot_image_boxes
from scripts.training import valid_dataset

backbone = resnet_fpn_backbone("resnet50", pretrained=True)
kwargs = {"nms_tresh": 0.1, "detections_per_img": 5}
model = SubpixRCNN(backbone, num_classes=2, device='cpu', **kwargs)
model.load_state_dict(torch.load('sp_no_noise_justboxes_model.pth',map_location=torch.device('cpu')))

model.to(device='cuda')
device = 'cuda'
model.eval()
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
plot_image_boxes(img,None,out,True,True)
print(tar)
print(out)