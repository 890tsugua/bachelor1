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
num_datapoints = 1
num_spots_min = 20
num_spots_max = 20
sigma_mean= 1.0
sigma_std = 0.1
snr_min = 2
snr_max = 2
snr_std = 0.0
base_noise_min = 100
base_noise_max = 100
use_gauss_noise = False
gauss_noise_std = 0.02
use_perlin_noise = True
perlin_min_max = (0.4, 0.6)
img_w = 64
img_h = 64

dataset = PsfDataset(seed, num_datapoints, num_spots_min, num_spots_max, sigma_mean, sigma_std,
                      snr_min, snr_max, snr_std, base_noise_min, base_noise_max, use_gauss_noise,
                      gauss_noise_std, use_perlin_noise, perlin_min_max, img_w, img_h)

image, target = dataset[0]
PlotController(image, target, None, 'buttons', False, 1, False)