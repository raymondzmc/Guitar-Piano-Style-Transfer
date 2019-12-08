import torch
import torch.nn as nn
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import librosa
import random
import argparse
import pdb


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

import random
import torch


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def save_heatmap(data, path):
    """
    Save spectrogram as heatmap
    """
    # data = x_mel
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())

    # map the normalized data to colors
    # image is now RGBA (512x512x4) 
    image = cmap(norm(data))
    plt.imsave(path, image)


def reconstruct2(spectrogram, phase, spec_range):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()

    spec_min, spec_max = spec_range

    spectrogram = spectrogram.squeeze((0, 1))
    spectrogram = (spec_max - spec_min) * ((spectrogram + 1) / 2) + spec_min

    spectrogram = 10**(spectrogram)
    s_db_inv = librosa.feature.inverse.mel_to_stft(spectrogram, sr=cfg.sampling_rate, n_fft=cfg.n_fft)
    # s_power_inv = librosa.core.db_to_amplitude(s_db_inv, ref=1.0)
    # reconstruct without phase information
    stft = s_db_inv[:, :phase.shape[1]] * phase

    data1 = librosa.core.istft(stft, hop_length=cfg.hop_length)
    # data2 = librosa.griffinlim(s_db_inv, hop_length=cfg.hop_length)
    # # s_istft = librosa.istft(s_power_inv)
    # # reconstruct with random phase using Griffin-Lim algorithm
    # s_istft = librosa.griffinlim(s_power_inv)
    return data1, data1


def reconstruct(spectrogram, phase):
    """
    Reconstruct the wavefrom from Mel-spectrogram
    """

    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()

    spectrogram = spectrogram.squeeze((0, 1))

    # Convert Mel-spectrogram to power series
    stft = librosa.feature.inverse.mel_to_stft(spectrogram,
                                               sr=cfg.sampling_rate,
                                               n_fft=cfg.n_fft)

    # Scale stft to log-amplitude between -15 and 65 dB
    stft = np.interp(stft, (stft.min(), stft.max()), (cfg.db_min, cfg.db_max))

    # Convert decibel to amplitude
    stft = librosa.core.db_to_amplitude(stft)
    
    # Re-construct using input phase
    stft = stft * phase

    data = librosa.core.istft(stft, hop_length=cfg.hop_length)

    return data



def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_ckpt", type=int, default=0, help="Load pre-trained parameters by number of epochs trained")
    args = parser.parse_args()
    return args