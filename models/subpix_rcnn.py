# subpix_rcnn_2

# Jeg skal i virkeligheden starte ud fra faster_rcnn.py, dvs extende generalized rcnn?? 
# Problemet er at roi_heads bliver kaldt og defineret inde i faster__rcnn
# Så skal jeg lave min egen RoIHead. Denne skal nemlig gives som input til generalizedrcnn.
# Men jeg skal stadig bruge noget af det roihead der findes for faster rcnn... Vil jo stadig godt have bounding boxes til når det
# skal generaliseres...

# Obs på roi_align. Jeg skal også roi_aligne ground truth subpixel pos for at kunne lave loss function?? 
# se funktion "project_masks_on_boxes"

from collections import OrderedDict
from typing import Any, Callable, Optional
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign

from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms._presets import ObjectDetection
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
import torch.nn.functional as F
import torch


class SubpixHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduces to [N, 64, 1, 1]
            nn.Flatten(),                  # [N, 64]
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),              # Output x, y subpixel positions
            nn.Sigmoid()                   # Values in [0, 1]
        )

    def forward(self, x):
        return self.head(x)
    

class SubpixRoIHeads(RoIHeads):
    def __init__(self, *args, subpixel_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.subpixel_head = subpixel_head
        self.device = kwargs.get('device', None)
        #dev = kwargs.get('device', None)
        #self.device = torch.device(dev) if dev is not None else torch.device("cpu")
        print("Custom SubpixRoIHeads successfully initialized!")

    def forward(self, features, proposals, image_shapes, targets = None):
        result, losses = super().forward(features, proposals, image_shapes, targets) # Results and losses after backbone, rpn and bbox head
        # Now the subpixel head logic.
        
        if 1 == 2 and self.subpixel_head is not None: # Make sure there is a subpixel head.

            bbox_proposals = [r["boxes"] for r in result] # Extracts bounding boxes. r is for each image.

            # Check if any images don't have boxes... Make the actual lists (not empty) that will be used for training and inference.
            pooled_boxes = []
            pooled_image_shapes=[]
            pooled_offset_targets=[]
            pooled_indices=[]
            # Pooled targets: List of tensors shapes [N,2]
            # Pooled boxes: List of boxes shaped [N,4]
            # Bbox proposals: List of boxes shaped [N,4] ?? [x1, y1, x2, y2]
            for i, boxes in enumerate(bbox_proposals):
                if boxes.shape[0]>0:
                    pooled_indices.append(i)
                    pooled_boxes.append(boxes)
                    pooled_image_shapes.append(image_shapes[i])
                    if self.training:
                        if targets is None:
                            raise ValueError("Targets should not be none when training.")
                        #pooled_targets.append(targets[i]['subpixel_positions'])
                        pooled_offset_targets.append(targets[i]['positions']) # [N,2]

                        # I am gonna try implementing normalized offset inside the whole box.
                        #offsets = (targets[i]['positions'] - boxes[:,[0,1]])
                        #widths = boxes[:,2] - boxes[:,0] # x2 - x1
                        #heights = boxes[:,3] - boxes[:,1] # y2 - y1
                        #box_sizes = torch.stack([widths, heights], dim=1) # [N,2]
                        #pooled_offset_targets.append(offsets / box_sizes) 

            if len(pooled_boxes) > 0:
                #subpixel_features = self.box_roi_pool(features,pooled_boxes,pooled_image_shapes) 
                #subpixel_offsets = self.subpixel_head(subpixel_features)
                proposed_box_centers = []
                for boxes in pooled_boxes:
                    if boxes.dim() == 1:
                        cx = (boxes[0] + boxes[2]) / 2
                        cy = (boxes[1] + boxes[3]) / 2
                    else:
                        cx = (boxes[:, 0] + boxes[:, 2]) / 2
                        cy = (boxes[:, 1] + boxes[:, 3]) / 2
                    pos = torch.stack([cx, cy], dim=1)  # shape [N, 2]
                    proposed_box_centers.append(pos)
                proposed_box_centers = torch.cat(proposed_box_centers, dim=0) # shape [N, 2]

                if self.training:
                    # Compute subpixel loss.
                    # subpixel_loss = F.mse_loss(subpixel_offsets, torch.cat(pooled_offset_targets,dim=0)) # Combine targets into one tensor
                    #losses.update({"loss_subpixel": subpixel_loss})
                    # Compute position loss
                    position_loss = F.mse_loss(proposed_box_centers, torch.cat(pooled_offset_targets,dim=0)) # Combine targets into one tensor
                    losses.update({"loss_subpixel": position_loss})
                else:
                    # Inference mode. Add the subpixel offset proposals to the results.
                    num_boxes_per_image = [boxes.shape[0] for boxes in pooled_boxes]
                    #split_offsets = torch.split(subpixel_offsets,num_boxes_per_image) # Split the subpixel proposals
                    split_offsets = torch.split(proposed_box_centers,num_boxes_per_image)
                    for i, offsets in zip(pooled_indices,split_offsets):
                        result[i]['subpixel_positions'] = offsets

            else: # In case that there are no images with any detections
                if self.training:
                    losses.update({"loss_subpixel": torch.tensor(0.0, device=self.device)})
            
        return result, losses


class SubpixRCNN(FasterRCNN):
    def __init__(
        self, 
        backbone, 
        num_classes=None, 
        subpix_in_channels=256, 
        **kwargs):
       
        """
        Basically the same implementation as the FasterRCNN class, but adds a custom head.
        kwargs: set specific rpn, box head, transform etc.
        """

        super().__init__(backbone, num_classes) # Initialize FasterRCNN with default rpn, box head, transform and roi_heads
        
        brp=self.roi_heads.box_roi_pool
        bh=self.roi_heads.box_head
        bp=self.roi_heads.box_predictor

        # Override the roi_heads
        subpixel_head = SubpixHead(subpix_in_channels)
        new_roi_heads = SubpixRoIHeads(
            box_roi_pool=kwargs.get("box_roi_pool",brp),
            box_head=kwargs.get("box_head",bh),
            box_predictor=kwargs.get("box_predictor",bp),
            fg_iou_thresh=kwargs.get("fg_iou_thresh", 0.5),
            bg_iou_thresh=kwargs.get("bg_iou_thresh", 0.5),
            batch_size_per_image=kwargs.get("batch_size_per_image", 512),
            positive_fraction=kwargs.get("positive_fraction", 0.25),
            bbox_reg_weights=kwargs.get("bbox_reg_weights", None),
            score_thresh=kwargs.get("score_thresh", 0.05),
            nms_thresh=kwargs.get("nms_thresh", 0.5),
            detections_per_img=kwargs.get("detections_per_img", 100), # No need to pass mask variables, they default to None
            subpixel_head=subpixel_head
        )
        self.roi_heads = new_roi_heads

    def forward(self, images, targets=None):
        """
        Compute the forward pass of the model and add box centers to the results.
        """
        outputs = super().forward(images, targets)
        # Standard training behavior inherited from the parent class
        if self.training:
            return outputs
        
        # Inference. Add the box centers to the results
        for output in outputs:
            boxes = output['boxes']
            if boxes.numel() > 0:
                if boxes.dim() == 1:
                    cx = (boxes[0] + boxes[2]) / 2
                    cy = (boxes[1] + boxes[3]) / 2
                else:
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2
                centers = torch.stack([cx, cy], dim=1)  # shape [N, 2]
                output['box_centers'] = centers
            else:
                output['box_centers'] = torch.empty((0, 2), device=boxes.device)
        return outputs

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
        # Call the parent class to load everything else
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

        # Try to load subpixel_head if present
        subpixel_head_state_dict_keys = [k for k in state_dict.keys() if k.startswith(prefix + "roi_heads.subpixel_head.")]
        if len(subpixel_head_state_dict_keys) > 0:
            subpixel_prefix = prefix + "roi_heads.subpixel_head."
            self.roi_heads.subpixel_head._load_from_state_dict(
                state_dict, subpixel_prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs
            )
        else:
            # If the keys aren't present but strict=True, treat it as missing
            if strict and self.roi_heads.subpixel_head is not None:
                m_keys = [prefix + "roi_heads.subpixel_head." + k for k, _ in self.roi_heads.subpixel_head.named_parameters()]
                missing_keys.extend(m_keys)
