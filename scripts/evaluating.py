from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.subpix_rcnn import SubpixRCNN
from utils import move_data_to_device, move_dict_to_cpu, plot_image_boxes, evaluate_predictions
from utils import evaluate_prediction
from PsfSimulator import PsfDataset 
import numpy as np
from scripts.plotting import PlotController
from torchvision.ops import box_iou
from collections import defaultdict


# LOC ERROR SHOULD BE ROOT MEAN SQUARED ERROR

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

def evaluate_recall_per_snr(model, reps, device, iou_thresh, **kwargs):
    """
    Evaluate the recall.
    Args:
        model: The model to be evaluated.
        reps: Number of images for the evaluation.
        device: The device to run the model on (CPU or GPU).
        iou_thresh: IoU threshold for matching predictions to ground truth.
        **kwargs: Additional arguments for the dataset.
    """
    results = defaultdict(lambda: {'TP':0, 'FN':0})
    model.eval()
    model.to(device)
    dp = kwargs.get('dp', 1)  # Decimal places for rounding SNRs
    dataset = PsfDataset(
        seed = kwargs.get('seed', None),
        num_datapoints=1,
        num_spots_min=1,
        num_spots_max=1,
        sigma_mean = kwargs.get('sigma_mean', 1.0),
        sigma_std = kwargs.get('sigma_std', 0.1),
        snr_min = kwargs.get('snr_min', 1),
        snr_max = kwargs.get('snr_max', 15),
        snr_std = kwargs.get('snr_std', 0.0),
        base_noise_min=10,
        base_noise_max=6000,
        use_gauss_noise=False,
        gauss_noise_std=0.05,
        use_perlin_noise=False,
        perlin_min_max=(0.4, 0.6),
        img_w=64, 
        img_h=64)

    for i in range(reps):
        if i%100 == 0:
            print(f"Processing image {i+1}/{reps}")
        image, target = dataset[0]
        image = move_data_to_device(image, device)
        target = move_data_to_device(target, device)
        with torch.no_grad():
            prediction = model([image])
        true_snr = target['true_snrs']
        pred_boxes = prediction[0]['boxes']  # Get the boxes from the prediction 
        gt_box = target['boxes'] # Only one box per image in this case
        boxfound = False
        for box in pred_boxes:
            if box_iou(box.unsqueeze(0), gt_box).item() > iou_thresh:
                # If the prediction box matches the ground truth box
                results[round(true_snr.item(),dp)]['TP'] += 1
                boxfound = True
        if not boxfound:
            results[round(true_snr.item(),dp)]['FN'] += 1

    return results


    #     # Matching logic
    #     pred_boxes = prediction[0]['boxes']
    #     gt_boxes = target['boxes']
    #     iou_matrix = box_iou(pred_boxes, gt_boxes)
    #     matches = []
    #     used_preds = set()
    #     used_gts = set()
    #     pairs = [(i, j, iou_matrix[i,j].item())
    #             for i in range(iou_matrix.shape[0])
    #             for j in range(iou_matrix.shape[1])] # Makes a list of tuples. All possible pairs
    #     # Sort pairs by IoU in descending order
    #     pairs.sort(key=lambda x: x[2], reverse=True)
    #     # Now greedy matching.
    #     for i, j, iou in pairs:
    #         if iou < iou_thresh:
    #             continue
    #         if i not in used_preds and j not in used_gts:
    #             matches.append((i,j))
    #             used_preds.add(i)
    #             used_gts.add(j)
        
    #     for j in range(len(gt_boxes)):
    #         if j not in used_gts:  # Ground truth box not matched
    #             results[round(true_snrs[j].item(), 0)]['FN'] += 1  # Increment FN
    #         else:
    #             results[round(true_snrs[j].item(), 0)]['TP'] += 1  # Increment TP
    # return results


def test_model_fixed_snr(model, snr, num_images, device, **kwargs):
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
    seed = kwargs.get('seed', None)
    sigma_std = kwargs.get('sigma_std', 0.1)
    sigma_mean = kwargs.get('sigma_mean', 1.0)
    snr_std = kwargs.get('snr_std', 0.1)
    base_noise_min = kwargs.get('base_noise_min', 20)
    base_noise_max = kwargs.get('base_noise_max', 150)
    use_gauss_noise = kwargs.get('use_gauss_noise', False)
    gauss_noise_std = kwargs.get('gauss_noise_std', 0.05)
    use_perlin_noise = kwargs.get('use_perlin_noise', False)
    perlin_min_max = kwargs.get('perlin_min_max', (0.4, 0.6))
    img_w = kwargs.get('img_w', 64)
    img_h = kwargs.get('img_h', 64)
    um_per_pixel = kwargs.get('um_per_pixel', 0.1) # Standard 100nm per pixel

    results = {}
    model.eval()
    model.to(device)

    # Fine steps for small values
    small = np.arange(0.04, 0.1, 0.01)      # 0.04, 0.05, ..., 0.10
    medium = np.arange(0.1, 1.0, 0.1)        # 0.2, 0.3, ..., 0.9
    large = np.arange(1, 5, 1)               # 1, 2, 3
    densities = np.unique(np.concatenate([small, medium, large]))

    for density in densities:
        print(f"Testing density: {density}")
        num_spots = density_to_num_spots(density, img_w, img_h, um_per_pixel)
        dataset = PsfDataset(seed=seed,
                             num_datapoints=num_images,
                             num_spots_min=num_spots,
                             num_spots_max=num_spots,
                             sigma_mean=sigma_mean,
                             sigma_std=sigma_std,
                             snr_min=snr,
                             snr_max=snr,
                             snr_std=snr_std,
                             base_noise_min=base_noise_min,
                             base_noise_max=base_noise_max,
                             use_gauss_noise=use_gauss_noise,
                             gauss_noise_std=gauss_noise_std,
                             use_perlin_noise=use_perlin_noise,
                             perlin_min_max=perlin_min_max,
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
            all_predictions.append(prediction[0]) 
            all_targets.append(target)
        
        # Evaluate predictions
        metrics = evaluate_predictions(all_predictions, all_targets)
        results[density] = metrics
        print(f"Density: {density}, Metrics: {metrics}")
    return results
