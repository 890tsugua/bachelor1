
import pandas as pd
import torch
#from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor, _validate_trainable_layers
from tqdm.auto import tqdm
from utils import move_data_to_device, move_dict_to_cpu
import time

def run_epoch(model, dataloader, optimizer, device, is_training):
  model.train()
  epoch_loss = 0
  progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")

  load_batch_time = 0
  train_time = 0
  t0 = time.time()
  
  for batch_id, (images, targets) in enumerate(dataloader):
    t1 = time.time()
    load_batch_time += t1 - t0
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

    t2 = time.time()
    train_time += t2 - t1

    epoch_loss += loss.item()

    # Update the progress bar.
    progress_bar_dict = dict(loss=epoch_loss, avg_loss = epoch_loss/(batch_id+1),
            data_time=f"{load_batch_time:.3f}s",
            train_time=f"{train_time:.3f}s")
    progress_bar.set_postfix(progress_bar_dict)
    progress_bar.update()
  progress_bar.close()

  return epoch_loss / (batch_id+1) # Returns average loss for this epoch.

def train_loop(model, training_loader, validation_loader, optimizer, device, epochs, checkpoint_path, writer):
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
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', valid_loss, epoch)
  # If the device is a GPU, empty the cache
  if device.type != 'cpu':
    getattr(torch, device.type).empty_cache()
  writer.close()
