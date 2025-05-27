import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou
import sys
import PIL
from PIL import Image
import tifffile

# Open the TIFF file
def extract_frame_from_tiff(tiff_file, frame_index):
    try:
        with tifffile.TiffFile(tiff_file) as tif:
            if frame_index < 0 or frame_index >= len(tif.pages):
                raise IndexError("Frame index out of range.")
            page = tif.pages[frame_index]
            image = page.asarray()
            return Image.fromarray(image)
    except Exception as e:
        print(f"Error extracting frame: {e}")
        sys.exit(1)
        # Then: image.save('/Users/august/Downloads/frame_t_0_extracted.tif', format='TIFF')

def evaluate_prediction(prediction, target, iou_thresh = 0.5):
    """
    Evaluate a single image prediction against the target. 
    Prediction and target are dictionaries containing keys...
    """
    pred_boxes = prediction['boxes']
    gt_boxes = target['boxes']

    # Match each prediction to at most one ground truth using greedy matching
    
    iou_matrix = box_iou(pred_boxes, gt_boxes)

    matches = []
    used_preds = set()
    used_gts = set()

    pairs = [(i, j, iou_matrix[i,j].item())
             for i in range(iou_matrix.shape[0])
             for j in range(iou_matrix.shape[1])] # Makes a list of tuples. All possible pairs

    # Sort pairs by IoU in descending order
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Now greedy matching.
    for i, j, iou in pairs:
        if iou < iou_thresh:
            continue
        if i not in used_preds and j not in used_gts:
            matches.append((i,j))
            used_preds.add(i)
            used_gts.add(j)

    # Calculate TP, FP, FN
    tp = len(matches)
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - tp

    ji = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2*(precision*recall))/(precision+recall) if (precision + recall) > 0 else 0

    # ADD LOCALIZATION ERROR
    loc_error = 0
    for (i,j) in matches:
        dx = target['positions'][j][0] - prediction['box_centers'][i][0]
        dy = target['positions'][j][1] - prediction['box_centers'][i][1]
        loc_error += (dx**2 + dy**2)**0.5
    loc_error /= tp if tp > 0 else 1
    loc_error = loc_error.item() if isinstance(loc_error, torch.Tensor) else loc_error

    return precision, recall, f1, ji, loc_error

def evaluate_predictions(predictions, targets, iou_thresh = 0.5):
    """
    A method that is able to evaluate a batch (list) of predictions
    """

    precision = 0
    recall = 0
    total_f1 = 0
    total_ji = 0
    total_le = 0

    for prediction, target in zip(predictions, targets):
        p, r, f1, ji, le = evaluate_prediction(prediction, target, iou_thresh)
        precision += p
        recall += r
        total_f1 += f1
        total_ji += ji
        total_le += le

    num_images = len(predictions)
    precision /= num_images
    recall /= num_images
    avg_f1 = total_f1 / num_images
    avg_ji = total_ji / num_images
    avg_le = total_le / num_images

    return {"precision": precision, "recall": recall, "avg f1": avg_f1, "avg ji": avg_ji, "avg loc error": avg_le}


def move_data_to_device(data, # Data to move to the device.
                        device:torch.device # The PyTorch device to move the data to.
                       ): # Moved data with the same structure as the input but residing on the specified device.
    """
    Recursively move data to the specified device.

    This function takes a data structure (could be a tensor, list, tuple, or dictionary)
    and moves all tensors within the structure to the given PyTorch device.
    """
    # If the data is a tuple, iterate through its elements and move each to the device.
    if isinstance(data, tuple):
        return tuple(move_data_to_device(d, device) for d in data)
    
    # If the data is a list, iterate through its elements and move each to the device.
    if isinstance(data, list):
        return list(move_data_to_device(d, device) for d in data)
    
    # If the data is a dictionary, iterate through its key-value pairs and move each value to the device.
    elif isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    
    # If the data is a tensor, directly move it to the device.
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    
    # If the data type is not a tensor, list, tuple, or dictionary, it remains unchanged.
    else:
        return data

def move_dict_to_cpu(data_dict):
  """Moves all tensors in a dictionary to CPU.

  Args:
    data_dict: A dictionary containing tensors.

  Returns:
    A dictionary with all tensors moved to CPU.
  """
  for key, value in data_dict.items():
    if isinstance(value, torch.Tensor):
      data_dict[key] = value.cpu()
  return data_dict

def plot_image(image, targets=None, proposals=None, plot_boxes=True, plot_pos=True):
    # Add a batch dimension
    image = image.unsqueeze(0)  
    enlarged_image = F.interpolate(image, scale_factor=10, mode='nearest')
    img = enlarged_image.squeeze(0) 
    img = img.cpu()
    if targets is not None and plot_boxes == True:
        clrs = []
        for i in range(targets['boxes'].shape[0]):
            clrs.append(2)
        t_boxes = (targets['boxes']*10)
        t_boxes[:,[2,3]] += 9
        img = draw_bounding_boxes(img, t_boxes,fill=False, width=1,colors=clrs)
    if proposals is not None and plot_boxes==True:
        p_boxes = (proposals['boxes']*10)
        p_boxes[:,[2,3]] += 9
        img = draw_bounding_boxes(img,p_boxes,fill=False,width=1)
    image = img[0].cpu()

    plt.imshow(image)
    if targets is not None and plot_pos==True:
        for p, b in zip(targets['subpixel_positions'], targets['boxes']):
            plt.scatter([(b[0].item()+1+p[0].item())*10],[(b[1].item()+1+p[1].item())*10], color="red", marker="x")
            #plt.scatter([p[0].item()*64*10],[p[1].item()*64*10], color="red", marker="x")  # Plot a red cross
    if proposals is not None and plot_pos==True:
        for p, b in zip(proposals['subpixel_positions'], proposals['boxes']):
                plt.scatter([(b[0].item()+1+p[0].item())*10],[(b[1].item()+1+p[1].item())*10], edgecolors="blue", marker="o",facecolors="none")
            #plt.scatter([q[0].item()*64*10],[q[1].item()*64*10], edgecolors="blue", marker="o",facecolors="none") 
    plt.show()

def plot_image_norm(image, targets=None, proposals=None, plot_boxes=True, plot_pos=True):
    # Add a batch dimension
    image = image.unsqueeze(0)  
    enlarged_image = F.interpolate(image, scale_factor=10, mode='nearest')
    img = enlarged_image.squeeze(0) 
    img = img.cpu()
    if targets is not None and plot_boxes == True:
        clrs = []
        for i in range(targets['boxes'].shape[0]):
            clrs.append(2)
        t_boxes = (targets['boxes']*10)
        t_boxes[:,[2,3]] += 9
        img = draw_bounding_boxes(img, t_boxes,fill=False, width=1,colors=clrs)
    if proposals is not None and plot_boxes==True:
        p_boxes = (proposals['boxes']*10)
        p_boxes[:,[2,3]] += 9
        img = draw_bounding_boxes(img,p_boxes,fill=False,width=1)
    image = img[0].cpu()

    plt.imshow(image)
    if targets is not None and plot_pos==True:
        for p, b in zip(targets['subpixel_positions'], targets['boxes']):
            p *= get_box_sizes(b)
            plt.scatter([(b[0].item()+p[0].item())*10],[(b[1].item()+p[1].item())*10], color="red", marker="x")
    if proposals is not None and plot_pos==True:
        for p, b in zip(proposals['subpixel_positions'], proposals['boxes']):
                p *= get_box_sizes(b)
                plt.scatter([(b[0].item()+p[0].item())*10],[(b[1].item()+p[1].item())*10], edgecolors="blue", marker="o",facecolors="none")
            #plt.scatter([q[0].item()*64*10],[q[1].item()*64*10], edgecolors="blue", marker="o",facecolors="none") 
    plt.show()

def plot_image_boxes(image, targets=None, proposals=None, plot_boxes=True, plot_pos=True):
    image = image.unsqueeze(0)  
    enlarged_image = F.interpolate(image, scale_factor=10, mode='nearest')
    img = enlarged_image.squeeze(0) 
    img = img.cpu()
    
    if targets is not None and plot_boxes == True:
        clrs = []
        for i in range(targets['boxes'].shape[0]):
            clrs.append(2)
        t_boxes = (targets['boxes']*10)
        img = draw_bounding_boxes(img, t_boxes,fill=False, width=1,colors=clrs)
    if proposals is not None and plot_boxes==True:
        p_boxes = (proposals['boxes']*10)
        img = draw_bounding_boxes(img,p_boxes,fill=False,width=1)
    image = img[0].cpu()
    plt.imshow(image)

    if targets is not None and plot_pos==True:
        for p in targets['positions']:
            plt.scatter([p[0].item()*10],[p[1].item()*10], color="red", marker="x")
    if proposals is not None and plot_pos==True:
        for p in proposals['box_centers']:
            plt.scatter([p[0].item()*10],[p[1].item()*10], edgecolors="blue", marker="o",facecolors="none")
    plt.show()

def get_box_sizes(boxes):
    if boxes.dim() > 1:
        widths = boxes[:,2] - boxes[:,0]
        heights = boxes[:,3] - boxes[:1]
        box_sizes = torch.stack([widths, heights], dim=1)
    elif boxes.dim() == 1:
        widths = boxes[2] - boxes[0]
        heights = boxes[3] - boxes[1]
        box_sizes = torch.tensor([widths, heights])
    else: 
        raise ValueError("boxes must be 1D or 2D tensor")
    return box_sizes
