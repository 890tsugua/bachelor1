from collections import OrderedDict
from typing import Any, Callable, Optional

import math
import pandas as pd
import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
import torch.optim as optim
from data_simulator import PsfDataset, plot_image, plot_image_boxes

from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms._presets import ObjectDetection
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead
from torchvision.models.detection.mask_rcnn import MaskRCNN
from tqdm.auto import tqdm
from subpix_rcnn_2 import SubpixRCNN
from model_prep import move_data_to_device, move_dict_to_cpu
import datetime
from pathlib import Path
import os

# Instantiate the datasets.
num_spots_min = 5
num_spots_max = 5
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


# Print the number of samples in the training and validation datasets
print(pd.Series({
    'Training dataset size:': len(train_dataset),
    'Validation dataset size:': len(valid_dataset)}))

""" DATA LOADER """
data_loader_params = {'batch_size':4,    'collate_fn': lambda batch: tuple(zip(*batch)),}

training_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_params)
validation_loader = torch.utils.data.DataLoader(valid_dataset, **data_loader_params)

def run_epoch(model, dataloader, optimizer, device, is_training):
  model.train()
  epoch_loss = 0
  progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")

  for batch_id, (images, targets) in enumerate(dataloader):
    inputs = torch.stack(images).to(device)
    inputs = move_data_to_device(inputs, device)
    targets = move_data_to_device(targets, device)

    if is_training:
      losses = model(inputs, targets)
    else:
      with torch.no_grad():
        losses = model(inputs, targets) # Validation / inference mode

    loss = sum(loss for loss in losses.values())

    # If in training, now backpropagate error and update weights.
    if is_training:
      loss.backward() # Backward pass
      optimizer.step() # Update weights
      optimizer.zero_grad() # Zero the gradients.

    epoch_loss += loss.item()

    # Update the progress bar.
    progress_bar_dict = dict(loss=epoch_loss, avg_loss = epoch_loss/(batch_id+1))
    progress_bar.set_postfix(progress_bar_dict)
    progress_bar.update()
  progress_bar.close()

  return epoch_loss / (batch_id+1) # Returns average loss for this epoch.

def train_loop(model, training_loader, validation_loader, optimizer, device, epochs, checkpoint_path):
  best_loss = torch.inf

  for epoch in tqdm(range(epochs), desc="Epochs"):
    train_loss = run_epoch(model, training_loader, optimizer, device, True)

    # Run a validation epoch.
    with torch.no_grad():
      valid_loss = run_epoch(model,validation_loader,optimizer, device, False)
      if valid_loss < best_loss:
        best_loss = valid_loss
        print(f"New best loss: {best_loss}")
        torch.save(model.state_dict(), checkpoint_path)

  # If the device is a GPU, empty the cache
  if device.type != 'cpu':
    getattr(torch, device.type).empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = resnet_fpn_backbone("resnet50", pretrained=True)
kwargs = {"nms_tresh": 0.1, "detections_per_img": 5}
model = SubpixRCNN(backbone, num_classes=2, device=device, **kwargs)
model.to(device)
model.name = "testmodel"
optimizer = optim.Adam(model.parameters(), lr=0.001)

import os
import datetime

def generate_checkpoint_path(model_name, project_name="Subpix_models"):
  """Generates a timestamped checkpoint path for saving model state dicts.

  Args:
    model_name: The name of the model.
    project_name: The name of the project (optional, defaults to "Subpix_models").

  Returns:
    A pathlib.Path object representing the checkpoint path.
  """
  # 1. Define the project directory within Colab's content area
  folder_path = os.path.join('/content/', project_name)

  # 2. Create the directory if it doesn't exist
  os.makedirs(folder_path, exist_ok=True)

  # 3. Generate a timestamped subdirectory
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  checkpoint_dir = os.path.join(folder_path, timestamp)
  os.makedirs(checkpoint_dir, exist_ok=True)

  # 4. Construct the checkpoint path
  checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pth")

  return checkpoint_path

checkpoint_path = generate_checkpoint_path("sp_no_noise_justboxes_model")
num_epochs = 2
checkpoint_path = "/Users/august/Desktop/bachelor"
train_loop(model, training_loader, validation_loader, optimizer, device, num_epochs, checkpoint_path)