import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import config as cfg

import random
import librosa
import os
import pdb




class AudioDataset(Dataset):

    def __init__(self, root_dir):

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


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]

        # randomize the index for domain y to avoid fixed pairs
        y = self.y[random.randint(0, len(self.y) - 1)]


        x, sr = librosa.core.load(x, sr=cfg.sampling_rate)
        y, sr = librosa.core.load(y, sr=cfg.sampling_rate)

        if len(x) != len(y):
            raise ValueError("Length mismatch for the two samples in the dataset!")

        # Transform audio to time-frequency spectrogram
        x_fft = librosa.core.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        y_fft = librosa.core.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        x_mag, x_phase = librosa.magphase(x_fft)
        y_mag, y_phase = librosa.magphase(y_fft)

        # Convert magnitude to decibel
        x_db = librosa.core.amplitude_to_db(x_mag, top_db=cfg.db_max)
        y_db = librosa.core.amplitude_to_db(y_mag, top_db=cfg.db_max)

        # # Clip the power spectrogram
        # x_db = np.clip(x_db, cfg.db_min, cfg.db_max)
        # y_db = np.clip(y_db, cfg.db_min, cfg.db_max)

        # Transformed to a Mel-frequency
        x_mel = librosa.feature.melspectrogram(S=x_db,
                                               sr=sr,
                                               n_fft=cfg.n_fft,
                                               hop_length=cfg.hop_length,
                                               n_mels=cfg.n_mels)
        y_mel = librosa.feature.melspectrogram(S=y_db,
                                               sr=sr,
                                               n_fft=cfg.n_fft,
                                               hop_length=cfg.hop_length,
                                               n_mels=cfg.n_mels)

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
        x_mel = (x_mel - x_mel.min()) / (x_mel.max() - x_mel.min())
        y_mel = (y_mel - x_mel.min()) / (x_mel.max() - x_mel.min())

        return self.transform(x_mel), self.transform(y_mel)


def get_dataloader(dataset):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=True,
                                             num_workers=8)
    return dataloader


# if __name__ == "__main__":

    # path = 'data/processed/'
    # dataset = AudioDataset(path)
    # dataloader = get_dataloader(dataset)

    # for epoch in range(2):
    #     print(dataloader.dataset.y[1:10])
    #     for i, (x, y) in enumerate(dataloader):
    #         pdb.set_trac
    # for i, (x, y) in enumerate(dataset):
    #     pdb.set_trace()

    # path = 'data/audio_hex-pickup_original'
    # total_minutes = 0.
    # sample_count = 0
    # file_list = os.listdir(path)
    # file_list.sort()
    # file_list = file_list[:360]
    # for i, file in enumerate(file_list):
    #     file = os.path.join(path, file)
    #     audio, rate = librosa.core.load(file, sr=None)

    #     for t in range(0, 44100 * 30, 512 * 256):
    #         sample = audio[t: t + (512 * 256) - 1]
    #         # pdb.set_trace()
    #         if len(sample) != 512 * 256 - 1:
    #             continue
    #         write_path = "data/processed/guitar/{}.wav".format(sample_count)
    #         librosa.output.write_wav(write_path, sample, rate)
    #         sample_count += 1

    #     # librosa.output.write_wav(path, y, sr, norm=False)

    #     # if not (rate == 44100):
    #     #     os.remove(file)

    #     # example_len = rate * 4
    #     # for j in range(len(audio) // example_len):
    #     #     start, end = j * example_len, 
    #     #     example = audio[j * ]

    #     # pdb.set_trace()
    # #     total_minutes += len(audio) / 44100 / 60
    #     print("{} of {}".format(i + 1, len(file_list)))
    # print(total_minutes)