import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as F

class PlotController:
    def __init__(self, image, targets, predictions):
        self.original_image = image.clone().cpu()  # Keep the original image
        self.targets = targets
        self.predictions = predictions
        self.show_boxes = True
        self.show_positions = True
        self.show_predictions = True
        self.show_targets = True

        # Prepare enlarged image for display
        img = self.original_image.unsqueeze(0)
        img = F.interpolate(img, scale_factor=10, mode='nearest')
        self.enlarged_image = img.squeeze(0).cpu()

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2) # Make room for buttons

        # Initial plot
        self.plot()

        # Add buttons
        ax_box = plt.axes([0.1, 0.05, 0.2, 0.075])
        ax_center = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.btn_box = Button(ax_box, 'Toggle Boxes')
        self.btn_center = Button(ax_center, 'Toggle Centers')

        self.btn_box.on_clicked(self.toggle_boxes)
        self.btn_center.on_clicked(self.toggle_centers)
        plt.show()

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
        
        self.ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        
        if self.show_positions:
            if self.show_targets:
                for p in self.targets['positions']:
                    self.ax.scatter([p[0].item() * 10], [p[1].item() * 10], color="red", marker="x")
            if self.show_predictions:
                for p in self.predictions['box_centers']:
                    self.ax.scatter([p[0].item() * 10], [p[1].item() * 10], edgecolors="blue", marker="o", facecolors="none")
        self.fig.canvas.draw_idle()

    def toggle_boxes(self, event):
        self.show_boxes = not self.show_boxes
        self.plot()

    def toggle_centers(self, event):
        self.show_positions = not self.show_positions
        self.plot()

