from PsfSimulator import PsfSimulator, PsfDataset
from scripts.plotting import PlotController

sim = PsfSimulator(use_perlin_noise=True, perlin_min_max=(0.4,0.6), sigma_mean=1, snr_mean=5)
image, target = sim.generate(num_spots=5)

PlotController(image, target, None, 'buttons', False, 1, False)

# Instatiate the dataset
seed = None
num_datapoints = 100
num_spots_min = 3
num_spots_max = 8
sigma_min = 1.0
sigma_max = 1.0
sigma_std = 0.2
snr_min = 10
snr_max = 10
snr_std = 0.3
base_noise_min = 50
base_noise_max = 100
use_gauss_noise = False
gauss_noise_std = 0.02
use_perlin_noise = True
perlin_min_max = (0.4, 0.6)
img_w = 64
img_h = 64

dataset = PsfDataset(seed, num_datapoints, num_spots_min, num_spots_max, sigma_min, sigma_max, sigma_std,
                      snr_min, snr_max, snr_std, base_noise_min, base_noise_max, use_gauss_noise,
                      gauss_noise_std, use_perlin_noise, perlin_min_max, img_w, img_h)

# Get a sample
img, target = dataset[0]
PlotController(img, target, None, 'buttons', False, 1, False)