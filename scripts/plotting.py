import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as F
from utils import evaluate_prediction, move_dict_to_cpu
from datetime import datetime
import numpy as np
import PIL
from PIL import Image

class PlotController:
    def __init__(self, image, targets, predictions, type, show_preds, show_targets, show_scores):
        plt.style.use('dark_background')
        self.original_image = image.clone().cpu()  # Keep the original image
        if targets is not None:
            self.targets = move_dict_to_cpu(targets)
        else:
            self.targets = None
        if predictions is not None:
            self.predictions = move_dict_to_cpu(predictions)
        else:
            self.predictions = None
        self.show_boxes = True
        self.show_positions = True
        self.show_predictions = show_preds
        self.show_targets = show_targets
        self.type = type
        self.show_scores = show_scores

        # Prepare enlarged image for display
        img = self.original_image.unsqueeze(0)
        img -= torch.min(img)
        img /= torch.max(img)
        img = F.interpolate(img, scale_factor=10, mode='nearest')
        self.enlarged_image = img.squeeze(0).cpu()
        self.fig, self.ax = plt.subplots()
        if self.type == "buttons" or self.type == "eval":

            plt.subplots_adjust(bottom=0.3) # Make room for buttons

            self.ax_metrics = plt.axes([0.7, 0.075, 0.2, 0.125])
            self.metrics_text_obj = None

        # Initial plot
        if self.type == "buttons" or self.type == "eval":
            plt.subplots_adjust(bottom=0.3) # Make room for buttons

            self.ax_metrics = plt.axes([0.7, 0.075, 0.2, 0.125])
            self.metrics_text_obj = None

            # Add buttons
            ax_box = plt.axes([0.1, 0.075, 0.2, 0.125])
            ax_center = plt.axes([0.4, 0.075, 0.2, 0.125])
            ax_saveimg = plt.axes([0.7, 0.01, 0.2, 0.05])  # Save image button

            self.btn_box = Button(ax_box, 'Toggle Boxes')
            self.btn_center = Button(ax_center, 'Toggle Centers')
            self.btn_saveimg = Button(ax_saveimg, 'Save Image')

            self.btn_box.on_clicked(self.toggle_boxes)
            self.btn_center.on_clicked(self.toggle_centers)
            self.btn_saveimg.on_clicked(self.save_img_tensor)  # Connect to save method


        # Initial plot
        self.plot()
        plt.show()

    def save_img_tensor(self, event):
        # Save only the image tensor (not the whole figure)
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        img = self.enlarged_image.cpu()
        # Convert tensor [C, H, W] to numpy [H, W, C] and scale to [0, 255]
        img_np = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        # If grayscale, remove channel dimension
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
        pil_img = Image.fromarray(img_np)
        pil_img.save(filename)
        print(f"Image tensor saved as {filename}")

    def plot(self):
        self.ax.clear()
        # Always start from the clean, enlarged image
        img = self.enlarged_image.clone()

        if self.show_boxes:
            # Draw bounding boxes
            if self.show_targets:
                t_boxes = self.targets['boxes'] * 10
                clrs = ["red"] * len(t_boxes)
                img = draw_bounding_boxes(img, t_boxes, fill=False, width=1, colors=clrs)
            if self.show_predictions:
                p_boxes = self.predictions['boxes'] * 10
                clrs = ["blue"] * len(p_boxes)
                img = draw_bounding_boxes(img, p_boxes, fill=False, width=1, colors=clrs)
                if self.show_scores:
                    for i, score in enumerate(self.predictions['scores']):
                        self.ax.text(p_boxes[i][0].item(), p_boxes[i][1].item()-4, f"{score:.2f}", color="blue", fontsize=6)
        
        self.ax.imshow(img.permute(1, 2, 0).cpu().numpy(), vmin = 0, vmax=1, cmap='gray')
        
        if self.show_positions:
            if self.show_targets:
                for p in self.targets['positions']:
                    self.ax.scatter([p[0].item() * 10], [p[1].item() * 10], color="red", marker="x")
            if self.show_predictions:
                for p in self.predictions['box_centers']:
                    self.ax.scatter([p[0].item() * 10], [p[1].item() * 10], edgecolors="blue", marker="o", facecolors="none")
        
        if self.type == "eval":
            # Display metrics
            precision, recall, f1, ji, me, stde, mse, stdse = evaluate_prediction(self.predictions, self.targets)
            metrics_text = f"Precision: {precision:.2f} \nRecall: {recall:.2f} \nF1 score: {f1:.2f} \nJaccard index: {ji:.2f} \nLocalization error: {me:.2f}px"
            
            self.ax_metrics.clear()
            self.ax_metrics.axis('off')
            self.ax_metrics.text(0.5,0.5, metrics_text, ha='center', va='center',
                        transform = self.ax_metrics.transAxes,fontsize=10,
                        bbox=dict(facecolor='white',alpha=0.7,edgecolor='black'))

        self.fig.canvas.draw_idle()

    def toggle_boxes(self, event):
        self.show_boxes = not self.show_boxes
        self.plot()

    def toggle_centers(self, event):
        self.show_positions = not self.show_positions
        self.plot()

