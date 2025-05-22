import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as F
from utils import evaluate_prediction, move_dict_to_cpu

class PlotController:
    def __init__(self, image, targets, predictions, type, show_preds, show_targets, show_scores):
        self.original_image = image.clone().cpu()  # Keep the original image
        self.targets = move_dict_to_cpu(targets)
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
        img = F.interpolate(img, scale_factor=10, mode='nearest')
        self.enlarged_image = img.squeeze(0).cpu()
        self.fig, self.ax = plt.subplots()
        if self.type == "buttons" or self.type == "eval":

            plt.subplots_adjust(bottom=0.3) # Make room for buttons

            self.ax_metrics = plt.axes([0.7, 0.075, 0.2, 0.125])
            self.metrics_text_obj = None

        # Initial plot
        self.plot()
        if self.type == "buttons" or self.type == "eval":
            # Add buttons
            ax_box = plt.axes([0.1, 0.075, 0.2, 0.125])
            ax_center = plt.axes([0.4, 0.075, 0.2, 0.125])
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
                if self.show_scores:
                    for i, score in enumerate(self.predictions['scores']):
                        self.ax.text(p_boxes[i][0].item(), p_boxes[i][1].item()-4, f"{score:.2f}", color="blue", fontsize=6)
        
        self.ax.imshow(img.permute(1, 2, 0).cpu().numpy(), vmin=0, vmax=1)
        
        if self.show_positions:
            if self.show_targets:
                for p in self.targets['positions']:
                    self.ax.scatter([p[0].item() * 10], [p[1].item() * 10], color="red", marker="x")
            if self.show_predictions:
                for p in self.predictions['box_centers']:
                    self.ax.scatter([p[0].item() * 10], [p[1].item() * 10], edgecolors="blue", marker="o", facecolors="none")
        
        if self.type == "eval":
            # Display metrics
            precision, recall, f1, ji, le = evaluate_prediction(self.predictions, self.targets)
            metrics_text = f"Precision: {precision:.2f} \nRecall: {recall:.2f} \nF1 score: {f1:.2f} \nJaccard index: {ji:.2f} \nLocalization error: {le:.2f}px"
            
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

