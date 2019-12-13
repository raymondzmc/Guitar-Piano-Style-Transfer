import torch
from loss import CycleGANModel
import config as cfg
from dataloader import AudioDataset
from utils import reconstruct2, save_heatmap

import numpy as np
import os
import librosa
import pdb


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    # Directory for checkpoints
    cwd = os.getcwd()
    ckpt_path = os.path.join(cwd, cfg.ckpt)

    to_evaluate = [620]

    models = []

    for epoch in to_evaluate:

        heatmap_path = os.path.join('heatmap', str(epoch))
        sample_path = os.path.join('samples', str(epoch))

        if not os.path.exists(heatmap_path):
            os.mkdir(heatmap_path)
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)

        model = CycleGANModel().to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_path, '{}.pth'.format(epoch))))
        model.eval()
        models.append(model)


    dataset = AudioDataset(cfg.data_path, eval=True)


    for i, (xx, yy, x, y, x_phase, y_phase, x_name, y_name) in enumerate(dataset):

        x, y = x.to(device).unsqueeze(0), y.to(device).unsqueeze(0)

        for j, model in enumerate(models):

            with torch.no_grad():
                model(x, y)

            fake_x, fake_y = model.fake_x, model.fake_y
            print(j)
            # re_fake_x1, re_fake_x2 = reconstruct2(fake_x, x_phase, (cfg.x_min, cfg.x_max))
            # re_fake_y1, re_fake_y2 = reconstruct2(fake_y, y_phase, (cfg.y_min, cfg.y_max))
            # re_x1, re_x2 = reconstruct2(x, x_phase, (cfg.x_min, cfg.x_max))
            # re_y1, re_y2 = reconstruct2(y, y_phase, (cfg.y_min, cfg.y_max))

            save_heatmap(x.squeeze().cpu().numpy(), 'heatmap/{}/{}_x.png'.format(to_evaluate[j], x_name))
            save_heatmap(y.squeeze().cpu().numpy(), 'heatmap/{}/{}_y.png'.format(to_evaluate[j], y_name))
            save_heatmap(fake_x.squeeze().cpu().numpy(), 'heatmap/{}/{}_fake_x.png'.format(to_evaluate[j], y_name))
            save_heatmap(fake_y.squeeze().cpu().numpy(), 'heatmap/{}/{}_fake_y.png'.format(to_evaluate[j], x_name))

            # librosa.output.write_wav('samples/{}/{}_x.wav'.format(to_evaluate[j], x_name), xx, cfg.sampling_rate)
            # librosa.output.write_wav('samples/{}/{}_y.wav'.format(to_evaluate[j], y_name), yy, cfg.sampling_rate)
            # librosa.output.write_wav('samples/{}/{}_fake_x_stft.wav'.format(to_evaluate[j], y_name), re_fake_x1, cfg.sampling_rate)
            # librosa.output.write_wav('samples/{}/{}_fake_y_stft.wav'.format(to_evaluate[j], x_name), re_fake_y1, cfg.sampling_rate)

            # librosa.output.write_wav('samples/{}/{}_x_gl.wav'.format(to_evaluate[j], x_name), re_x2, cfg.sampling_rate)
            # librosa.output.write_wav('samples/{}/{}_y_gl.wav'.format(to_evaluate[j], y_name), re_y2, cfg.sampling_rate)
            # librosa.output.write_wav('samples/{}/{}_fake_x_gl.wav'.format(to_evaluate[j], y_name), re_fake_x2, cfg.sampling_rate)
            # librosa.output.write_wav('samples/{}/{}_fake_y_gl.wav'.format(to_evaluate[j], x_name), re_fake_y2, cfg.sampling_rate)
    pdb.set_trace()