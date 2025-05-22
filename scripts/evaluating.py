from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions
from utils import evaluate_prediction
from data_simulator import PsfDataset   # USE NEW ONE
import numpy as np
from scripts.plotting import PlotController


# I want functions:
# 1. Run model across densities with a fixed SNR

def test_model_once_plot(model, dataset, device, type='simple'):
    img, tar = dataset[1]
    img = move_data_to_device(img, device)
    tar = move_data_to_device(tar, device)
    with torch.no_grad():
        out = model([img])
    eval = evaluate_predictions(out, [tar], 0.5)
    print(eval)
    out = out[0]
    move_dict_to_cpu(tar)
    move_dict_to_cpu(out)
    PlotController(img, tar, out, type, True, True)


def density_to_num_spots(density, img_w, img_h, um_per_pixel):
    """
    Convert the density to number of spots. Depends on image dimensions and conversion factor (nm per pixel)
    """
    # If 1 pixel is 100nm x 100nm then the area is 10^4 nm^2.
    # Then 10 pixels by 10 pixels is 1 um^2.
    # So 100 pixels is 1 um^2.
    # 64 x 64 = 4096 pixels = 40.96 um^2
    # 6.4 x 6.4 = 40.96
    area_in_um = (img_w * img_h) * um_per_pixel**2
    num_spots = density * area_in_um
    return int(num_spots)

def test_model_fixed_snr(model, snr, start_density, end_density, step_density, num_images, device, **kwargs):
    """
    Test the model with a fixed SNR across a range of densities.
    
    Parameters:
    - model: The model to be tested.
    - snr: The fixed SNR value.
    - start_density: The starting density for the test.
    - end_density: The ending density for the test.
    - step_density: The step size for the density.
    - num_images: The number of images to generate for each density.
    - device: The device to run the model on (CPU or GPU).
    
    Returns:
    - results: A dictionary containing the evaluation metrics for each density.
    """
    seed = kwargs.get('seed', 0)
    sigma_mean = kwargs.get('sigma_mean', 1.0)
    sigma_std = kwargs.get('sigma_std', 0.2)
    snr_std = kwargs.get('snr_std', 0.1)
    base_noise_min = kwargs.get('base_noise_min', 20)
    base_noise_max = kwargs.get('base_noise_max', 150)
    use_gauss_noise = kwargs.get('use_gauss_noise', True)
    gauss_noise_std = kwargs.get('gauss_noise_std', 0.05)
    img_w = kwargs.get('img_w', 64)
    img_h = kwargs.get('img_h', 64)
    um_per_pixel = kwargs.get('um_per_pixel', 0.1) # Standard 100nm per pixel

    results = {}
    model.eval()
    model.to(device)

    for density in np.arange(start_density, end_density + step_density, step_density):
        num_spots = density_to_num_spots(density, img_w, img_h, um_per_pixel)
        dataset = PsfDataset(seed=seed,
                             num_datapoints=num_images,
                             num_spots_min=num_spots,
                             num_spots_max=num_spots,
                             sigma_mean=sigma_mean,
                             sigma_std=sigma_std,
                             snr_mean=snr,
                             snr_std=snr_std,
                             base_noise_min=base_noise_min,
                             base_noise_max=base_noise_max,
                             use_gauss_noise=use_gauss_noise,
                             gauss_noise_std=gauss_noise_std,
                             img_w=img_w, 
                             img_h=img_h)
        all_predictions = []
        all_targets = []
        for i in range(num_images):
            image, target = dataset[0]
            image = move_data_to_device(image, device)
            target = move_data_to_device(target, device)
            with torch.no_grad():
                prediction = model([image])
            all_predictions.append(prediction[0])   # THIS IS WEIRD
            all_targets.append(target)
        
        # Evaluate predictions
        metrics = evaluate_predictions(all_predictions, all_targets)
        results[density] = metrics
        print(f"Density: {density}, Metrics: {metrics}")
    return results
