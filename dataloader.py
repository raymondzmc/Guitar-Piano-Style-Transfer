import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import config as cfg

import random
import librosa
import os
import pdb




class AudioDataset(Dataset):

    def __init__(self, root_dir, eval=False):

        try:
            piano_path = os.path.join(root_dir, cfg.piano)
            guitar_path = os.path.join(root_dir, cfg.guitar)
        except:
            print("Dataset directories do not exist in root_dir, please check \"config.py\"!")

        piano = [os.path.join(piano_path, f) for f in os.listdir(piano_path)]
        guitar = [os.path.join(guitar_path, f) for f in os.listdir(guitar_path)]

        # Define x to be smaller set, and y the larger set
        self.x = min(piano, guitar, key=len)
        self.y = max(piano, guitar, key=len)

        self.transform = transforms.ToTensor()
        self.eval = eval

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_path = self.x[idx]

        # randomize the index for domain y to avoid fixed pairs
        y_path = self.y[random.randint(0, len(self.y) - 1)]

        x, sr = librosa.core.load(x_path, sr=cfg.sampling_rate)
        y, sr = librosa.core.load(y_path, sr=cfg.sampling_rate)

        if not self.eval:
            x = x * np.random.uniform(low=0.5, high=1.0)
            y = y * np.random.uniform(low=0.5, high=1.0)

        # Randomly sample the audio for training
        sample_size = (cfg.sampling_rate * 10) if self.eval else cfg.n_mels * cfg.hop_length - 1
        x_idx = random.randint(0, len(x) - sample_size - 1)
        y_idx = random.randint(0, len(y) - sample_size - 1)
        x = x[x_idx:x_idx + sample_size]
        y = y[y_idx:y_idx + sample_size]

        if len(x) != len(y):
            raise ValueError("Length mismatch for the two samples in the dataset!")

        # Transform audio to time-frequency spectrogram
        x_fft = librosa.core.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        y_fft = librosa.core.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        x_mag, x_phase = librosa.magphase(x_fft)
        y_mag, y_phase = librosa.magphase(y_fft)

        # Convert magnitude to decibel
        # x_db = librosa.core.amplitude_to_db(x_mag)
        # y_db = librosa.core.amplitude_to_db(y_mag)

        # print("X max: {}".format(x_db.max()))
        # print("X min: {}".format(x_db.min()))
        # print("X mean: {}".format(x_db.mean()))
        # print("Y max: {}".format(y_db.max()))
        # print("Y min: {}".format(y_db.min()))
        # print("Y mean: {}".format(y_db.mean()))

        # # Clip the power spectrogram
        # x_db = np.clip(x_db, cfg.db_min, cfg.db_max)
        # y_db = np.clip(y_db, cfg.db_min, cfg.db_max)

        # Transformed to a Mel-frequency spectrogram
        x_mel = librosa.feature.melspectrogram(S=x_mag,
                                               sr=sr,
                                               n_fft=cfg.n_fft,
                                               hop_length=cfg.hop_length,
                                               n_mels=cfg.n_mels)
        y_mel = librosa.feature.melspectrogram(S=y_mag,
                                               sr=sr,
                                               n_fft=cfg.n_fft,
                                               hop_length=cfg.hop_length,
                                               n_mels=cfg.n_mels)

        # print(x_mel.min(), y_mel.min())
        x_mel = np.log10(np.clip(x_mel, a_min=1e-15, a_max=None))
        y_mel = np.log10(np.clip(y_mel, a_min=1e-15, a_max=None))

        # print(x_mel.min())
        # print(y_mel.min())
        # pdb.set_trace()
        # Transformed to a Mel-frequency spectrogram
        # x_mel = librosa.feature.melspectrogram(x,
        #                                        sr=sr,
        #                                        n_fft=cfg.n_fft,
        #                                        hop_length=cfg.hop_length,
        #                                        n_mels=cfg.n_mels)
        # y_mel = librosa.feature.melspectrogram(y,
        #                                        sr=sr,
        #                                        n_fft=cfg.n_fft,
        #                                        hop_length=cfg.hop_length,
        #                                        n_mels=cfg.n_mels)


        # Normalize Mel-spectrogram
        x_min, x_max, y_min, y_max = x_mel.min(), x_mel.max(), y_mel.min(), y_mel.max()

        if not (x_max == x_min):
            x_mel = 2 * ((x_mel - x_min) / (x_max - x_min)) - 1

        if not (y_max == y_min):
            y_mel = 2 * ((y_mel - y_min) / (y_max - y_min)) - 1

        # Return phase information and filename during evaluation
        if self.eval:
            x_name = os.path.basename(x_path).split('.')[0]
            y_name = os.path.basename(y_path).split('.')[0]
            return x, y, Variable(self.transform(x_mel)), Variable(self.transform(y_mel)), x_phase, y_phase, x_name, y_name, (x_min, x_max, y_min, y_max)
        else:
            return Variable(self.transform(x_mel)), Variable(self.transform(y_mel))


def get_dataloader(dataset):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=True,
                                             num_workers=cfg.n_workers)
    return dataloader


if __name__ == "__main__":

    path = 'data/processed/'
    dataset = AudioDataset(path)
    dataloader = get_dataloader(dataset)

    for epoch in range(2):
        print(dataloader.dataset.y[1:10])
        for i, (x, y) in enumerate(dataloader):
            pdb.set_trace()

    # for i, (x, y) in enumerate(dataset):
    #     pdb.set_trace()

    # path = 'data/youtube_piano'
    # total_minutes = 0.
    # sample_count = 0
    # file_list = os.listdir(path)
    # # file_list.sort()
    # # file_list = file_list[:360]
    # for i, file in enumerate(file_list):
    #     file = os.path.join(path, file)
    #     audio, rate = librosa.core.load(file, sr=None)

    #     for t in range(0, len(audio), rate * 30):
    #         sample = audio[t: t + (rate * 30)]
    #         # pdb.set_trace()
    #         if len(sample) <= 512 * 128:
    #             continue

    #         new_rate = 16000
    #         new_sample = librosa.core.resample(sample, rate, new_rate)
    #         write_path = "data/processed/piano/{}.wav".format(sample_count)
    #         librosa.output.write_wav(write_path, new_sample, new_rate)
    #         sample_count += 1

    #     # librosa.output.write_wav(path, y, sr, norm=False)

    #     # if not (rate == 44100):
    #     #     os.remove(file)

    #     # example_len = rate * 4
    #     # for j in range(len(audio) // example_len):
    #     #     start, end = j * example_len, 
    #     #     example = audio[j * ]

    #     # pdb.set_trace()
    #     total_minutes += len(audio) / rate / 60
    #     print("{} of {}".format(i + 1, len(file_list)))
    # print(total_minutes)