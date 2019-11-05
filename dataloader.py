import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config as cfg

import librosa
import os
import pdb




class AudioDataset(Dataset):

    def __init__(self, root_dir):
        self.files = os.listdir(path)
        self.files = [os.path.join(root_dir, f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        audio, sr = librosa.core.load(file, sr=cfg.sampling_rate)
        s = librosa.feature.melspectrogram(audio, sr=sr,
                                                  n_fft=cfg.n_fft,
                                                  hop_length=cfg.hop_length,
                                                  n_mels=cfg.n_mels)
        s_db = librosa.power_to_db(s, ref=1.0)
        return s_db



if __name__ == "__main__":

    path = 'data/processed/guitar'
    dataset = AudioDataset(path)
    for i, sample in enumerate(dataset):
        pdb.set_trace()

    # path = 'data/maestro-v1.0.0'
    # total_minutes = 0.
    # sample_count = 0
    # file_list = os.listdir(path)
    # for i, file in enumerate(file_list):
    #     file = os.path.join(path, file)
    #     audio, rate = librosa.core.load(file, sr=None)

    #     for t in range(0, 44100 * 30, 512 * 128):
    #         sample = audio[t: t + (512 * 128)]

    #         if len(sample) != 512 * 128:
    #             continue
    #         write_path = "data/processed/piano/{}.wav".format(sample_count)
    #         librosa.output.write_wav(write_path, sample, rate)
    #         sample_count += 1

        # librosa.output.write_wav(path, y, sr, norm=False)

        # if not (rate == 44100):
        #     os.remove(file)

        # example_len = rate * 4
        # for j in range(len(audio) // example_len):
        #     start, end = j * example_len, 
        #     example = audio[j * ]

        # pdb.set_trace()
    #     total_minutes += len(audio) / 44100 / 60
    #     print("{} of {}".format(i + 1, len(file_list)))
    # print(total_minutes)