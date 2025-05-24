from PsfSimulator import PsfSimulator, PsfDataset
from scripts.plotting import PlotController
import PIL
from PIL import Image
import torchvision.transforms as T


#sim = PsfSimulator(use_perlin_noise=False, perlin_min_max=(0.4,0.6), sigma_mean=1, snr_mean=5)
#image, target = sim.generate(num_spots=5)

#print(target)

#PlotController(image, target, None, 'buttons', False, 1, False)

# Instatiate the dataset
seed = None
num_datapoints = 100
num_spots_min = 5
num_spots_max = 5
sigma_mean= 1.0
sigma_std = 0.1
snr_min = 5
snr_max = 5
snr_std = 0.0
base_noise_min = 2000
base_noise_max = 2000
use_gauss_noise = False
gauss_noise_std = 0.02
use_perlin_noise = False
perlin_min_max = (0.4, 0.6)
img_w = 64
img_h = 64

dataset = PsfDataset(seed, num_datapoints, num_spots_min, num_spots_max, sigma_mean, sigma_std,
                      snr_min, snr_max, snr_std, base_noise_min, base_noise_max, use_gauss_noise,
                      gauss_noise_std, use_perlin_noise, perlin_min_max, img_w, img_h)

# Get a sample
img, target = dataset[0]

# Tensor to pil image
img = img.clone().cpu()
to_pil = T.ToPILImage()
im = to_pil(img)
# Convert to grayscale
im = im.convert("L")
im.save("/Users/august/Desktop/bachelor/bachelor1/testhighbg.png")

print(target)
PlotController(img, target, None, 'buttons', False, 1, False)

# Save the image



#import numpy as np
#a = np.zeros((10,10))
#a[1:3, 1:3] = 1
#print(a)

#sim = PsfSimulator(use_perlin_noise=True, perlin_min_max=(0.4,0.6), sigma_mean=1, snr_mean=5)
#psf = sim.make_psf(0.8, 250, (0.5,0.5))
#import PIL
#from PIL import Image
#im = Image.fromarray(psf)
#im = im.convert("L")
#im.show()