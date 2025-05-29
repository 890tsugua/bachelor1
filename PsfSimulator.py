import numpy as np
import torch
import matplotlib.pyplot as plt
from perlin_numpy import generate_perlin_noise_2d
from torch.utils.data import Dataset, DataLoader


class PsfSimulator:
    def __init__(self,
                 img_w=64,
                 img_h=64,
                 sigma_mean=1,
                 sigma_std=0.1,
                 snr_mean=10,
                 snr_std=0.2,
                 base_noise=100,
                 use_gauss_noise=False,
                 gauss_noise_std=15,
                 use_perlin_noise=False,
                 perlin_min_max=(0.4, 0.5)
                 ):
        
        self.img_w = img_w
        self.img_h = img_h

        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std

        self.snr_mean = snr_mean
        self.snr_std = snr_std

        self.base_noise = base_noise

        self.use_gauss_noise = use_gauss_noise
        self.gauss_noise_std = gauss_noise_std

        self.use_perlin_noise = use_perlin_noise
        self.perlin_min_max = perlin_min_max

    def add_poisson(self, array):
        return np.random.poisson(array).astype(np.float32)

    def make_background(self, base, pad):
        img_w, img_h = self.img_w + 2 * pad, self.img_h + 2 * pad
        if self.use_gauss_noise:
            array = np.clip(np.random.normal(base, self.gauss_noise_std, (img_h, img_w)), 0, None)
        
        elif self.use_perlin_noise:
            # Draw a random number 1,2,4,8
            d1 = np.random.choice([1, 2, 4])
            d2 = np.random.choice([1, 2, 4])
            perlin = generate_perlin_noise_2d((img_h - 2 * pad, img_w - 2 * pad), (d1,d2), (0, 0))
            if self.perlin_min_max is not None:
                min_perlin, max_perlin = self.perlin_min_max
            else:
                min_perlin = np.random.uniform(0, 0.5)
                max_perlin = np.random.uniform(0.5, 1)

            perlin = ((perlin+1)/2) * (max_perlin-min_perlin) + min_perlin # Normalize to [min_perlin, max_perlin]
            perlin = (perlin * base) + base/2
            array = np.zeros((img_h, img_w)).astype(np.int64)
            array[pad:-pad, pad:-pad] += perlin.astype(np.int64)
        
        else:
            array = np.full((img_h, img_w), int(base))

        return array.astype(np.float32)

    def generate_positions(self, num_spots):
        x = self.img_w * np.random.rand(num_spots)
        y = self.img_h * np.random.rand(num_spots)
        return np.column_stack((x, y))

    def calculate_signal_value(self, background_mean, snr):
        return snr * np.sqrt(background_mean)

    def make_psf(self, sigma, intensity, subpos):
        radius = int(np.ceil(3 * sigma))
        array = np.zeros((radius * 2 + 1, radius * 2 + 1))
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        subpos = (subpos[0] - 0.5, subpos[1] - 0.5)
        r2 = (xx - subpos[0]) ** 2 + (yy - subpos[1]) ** 2
        signals = np.exp(-r2 / (2 * sigma ** 2)) 
        array = signals * intensity
        return array.astype(np.float32)

    def calculate_background_mean(self, array, sigma, x, y):
        start_x, start_y = int(np.ceil(x - sigma * 3)), int(np.ceil(y - sigma * 3))
        end_x, end_y = int(np.ceil(start_x + sigma * 3)), int(np.ceil(start_y + sigma * 3))
        return np.mean(array[max(0, start_y):end_y, max(0, start_x):end_x])

    def draw_array(self, positions, sigmas, input_array, snrs, pad):
        copied_array = input_array.copy()
        for i, ((x, y), sigma, snr) in enumerate(zip(positions, sigmas, snrs)):
            subpos = (x%1, y%1)
            mean = self.calculate_background_mean(copied_array,sigma,pad+x,pad+y)

            psf = self.make_psf(sigma, self.calculate_signal_value(mean, snr), subpos)
            x, y = x.astype(int), y.astype(int)
            start_x, start_y = pad+x - int(np.ceil(sigma*3)), pad+y - int(np.ceil(sigma*3))  # Insert PSF input_arrays from their top left corner.
            input_array[start_y:start_y + psf.shape[0], start_x:start_x + psf.shape[1]] += psf

        # Remove padding
        unpadded_array = input_array[pad:-pad, pad:-pad]
        return unpadded_array.astype(np.float32)

    def find_true_snrs(self, arr, pos, img_w, img_h):
        rad = 6
        yy, xx = np.ogrid[-2:3,-2:3]
        psf_mask = xx**2 + yy**2 < 6
        signals = []
        snrs = []
        # Cut out all the PSFs from the array
        array = np.copy(arr)
        positions = np.copy(pos)
        array = np.pad(array, [[rad,rad],[rad,rad]], mode='median')
        positions += rad

        for (x,y) in positions:
            x1, x2 = int(x-2), int(x+3)
            y1, y2 = int(y-2), int(y+3)
            signal = np.nanmax(array[y1:y2, x1:x2])
            signals.append(signal)
            array[y1:y2, x1:x2][psf_mask] = np.nan

        for i, (x,y) in enumerate(positions):
            x1, x2 = int(x-rad+1), int(x+rad)
            y1, y2 = int(y-rad+1), int(y+rad)
            meanbg = np.nanmedian(array[y1:y2, x1:x2])
            #std = np.nanstd(array[y1:y2, x1:x2])
            snrbg = (signals[i]-meanbg) / np.sqrt(meanbg)
            snrsig = (signals[i]-meanbg) / np.sqrt(signals[i])
            snrs.append([snrbg, snrsig])

        return snrs

    def make_targets(self, positions, snrs):
        labels = torch.ones(positions.shape[0], dtype=torch.int64)
        bboxes = []
        for p in positions:
            box = [p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1]
            bboxes.append(box)
        bboxes = torch.clamp(torch.tensor(bboxes, dtype=torch.float32), min=0, max=self.img_w + 2)

        return {
            'boxes': bboxes,
            'labels': labels,
            'positions': torch.tensor(positions, dtype=torch.float32),
            'true_snrs': torch.tensor(snrs, dtype=torch.float32)
        }

    def generate(self, seed=None, num_spots=5):
        if seed is not None:
            np.random.seed(seed)

        if num_spots <= 0:
            raise ValueError("Number of spots must be more than 0")

        positions = self.generate_positions(num_spots)
        
        sigmas = np.clip(np.random.normal(self.sigma_mean, self.sigma_std, num_spots), 0, None)
        snrs = np.clip(np.random.normal(self.snr_mean, self.snr_std, num_spots), 0, None)
        pad = int(np.ceil(3 * np.max(sigmas) * 2))
        
        background = self.make_background(self.base_noise, pad)
        array = self.draw_array(positions, sigmas, background, snrs, pad)
        array = self.add_poisson(array)

        true_snrs = self.find_true_snrs(array, positions, self.img_w, self.img_h)

        array = np.pad(np.clip(array.astype(np.float32),0,None), ((1, 1), (1, 1)), mode='median')
        normalization = 'minmax'  # 'minmax', 'absolute', 'standard'
        
        if normalization == 'minmax':
            array -= np.min(array)
            array /= np.max(array)
        elif normalization == 'absolute':
            max_scale = 10000
            array /= max_scale
        elif normalization == 'standard':
            array = (array - np.mean(array)) / np.std(array)

        # minmaxarray = torch.from_numpy((array - np.min(array)) / (np.max(array) - np.min(array))).float()
        # standardarray = torch.from_numpy((array - np.mean(array)) / np.std(array)).float()
        # minmaximage = torch.stack([minmaxarray] * 3, axis=0)
        # standardimage = torch.stack([standardarray] * 3, axis=0)

        array = torch.from_numpy(array).float()
        image = torch.stack([array] * 3, axis=0)
        targets = self.make_targets(positions + 1, true_snrs)

        return image, targets

class PsfDataset(Dataset):
    """
    This class represents a PyTorch Dataset for a collection of images and their annotations.
    The class is designed to load images along with their corresponding segmentation masks, bounding box annotations, and labels.
    """
    def __init__(self, 
                seed=None, 
                num_datapoints=1, 
                num_spots_min=1,
                num_spots_max=80,
                sigma_mean=1.0,
                sigma_std=0.1,
                snr_min=2,
                snr_max=20,
                snr_std=0.2,
                base_noise_min=20,
                base_noise_max=150,
                use_gauss_noise=False,
                gauss_noise_std=0.05,
                use_perlin_noise=False,
                perlin_min_max=(0.4, 0.6),
                img_w=64, 
                img_h=64):

        super(Dataset, self).__init__()
        
        self._len = num_datapoints
        self._seed = seed # Should not be used

        self._num_spots_min = num_spots_min
        self._num_spots_max = num_spots_max

        self._sigma_mean = sigma_mean
        self._sigma_std = sigma_std

        self._snr_min = snr_min
        self._snr_max = snr_max
        self._snr_std = snr_std

        self._base_noise_min = base_noise_min
        self._base_noise_max = base_noise_max
        
        self._use_gauss_noise = use_gauss_noise
        self._gauss_noise_std = gauss_noise_std
        
        self._use_perlin_noise = use_perlin_noise
        self._perlin_min_max = perlin_min_max

        self._img_w = img_w
        self._img_h = img_h
        self.psf_simulator = None
        
    def __len__(self):
        return self._len
        
    def __getitem__(self, idx):
        # In here generate random parameters for the image generation:
        
        num_spots = np.random.randint(self._num_spots_min, self._num_spots_max+1)
        snr = np.random.uniform(self._snr_min, self._snr_max)
        base_noise = np.random.randint(self._base_noise_min, self._base_noise_max+1)
        

        # Istantiate the simulator
        self.psf_simulator = PsfSimulator(img_w=self._img_w,
                                          img_h=self._img_h,
                                          sigma_mean=self._sigma_mean,
                                          sigma_std=self._sigma_std,
                                          snr_mean=snr,
                                          snr_std=self._snr_std,
                                          base_noise=base_noise,
                                          use_gauss_noise=self._use_gauss_noise,
                                          gauss_noise_std=self._gauss_noise_std,
                                          use_perlin_noise=self._use_perlin_noise,
                                          perlin_min_max=self._perlin_min_max
                                          )

        image, target = self.psf_simulator.generate(seed=None,
                                    num_spots=num_spots)
        
        return image, target