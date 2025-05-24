# from scripts.evaluating import evaluate_recall_per_snr
# from models.subpix_rcnn import SubpixRCNN
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# import torch

# device = 'cpu'
# backbone = resnet_fpn_backbone("resnet50", pretrained=True)

# kwargs = {"nms_thresh": 0.2, "detections_per_img": None, "score_thresh":0.8, "device":torch.device(device)}
# model = SubpixRCNN(backbone, num_classes=2, **kwargs)
# model.to(device=device)
# path = r"/Users/august/Desktop/bachelor/bachelor1/first_real_run.pth"
# model.load_state_dict(torch.load(path, map_location=device))
# model.eval()

# result = evaluate_recall_per_snr(model, 1000, device, 0.5)

# print(result)
# for snr in result.keys():
#     print(snr, result[snr]['TP'], result[snr]['FN'])

# # Make a plot of the recall vs SNR
# import matplotlib.pyplot as plt
# import numpy as np
# snrs = np.array(list(result.keys()))
# recalls = np.array([result[snr]['TP'] / (result[snr]['TP'] + result[snr]['FN']) for snr in snrs])
# plt.scatter(snrs, recalls)
# plt.xlabel('SNR')
# plt.ylabel('Recall')
# plt.title('Recall vs SNR')
# plt.grid()
# plt.show()

import numpy as np
yy, xx = np.ogrid[-2:3,-2:3]
r2 = xx**2 + yy**2 < 6
print(r2)

a = np.zeros((11,11))
a[6][6] = r2