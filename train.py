import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time

from dataloader import AudioDataset, get_dataloader
from loss import CycleGANModel
import config as cfg
import pdb


if __name__ == "__main__":
    cuda = torch.cuda.is_available()

    if cuda:
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    # Create directory for checkpoints
    cwd = os.getcwd()
    ckpt_path = os.path.join(cwd, cfg.ckpt)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    # Initialize model
    model = CycleGANModel().to(device)

    # Initialize dataloader
    dataset = AudioDataset(cfg.data_path)
    dataloader = get_dataloader(dataset)

    n = len(dataloader)

    writer = SummaryWriter()


    # Training loop
    for epoch in range(1, cfg.epoch + 1):
        start_time = time.time()

        # Keep track of batch loss for discriminator and generator
        loss_D, loss_G = 0., 0.

        for i, (x, y) in enumerate(dataloader):

            if i + 1 % 100 == 0:
                print("Minibatch {} of {}".format(i + 1, n))

            x, y = x.to(device), y.to(device)

            try:
                model.optimize_parameters(x, y)
            except:
                pdb.set_trace()

            loss_D += model.loss_D.item()
            loss_G += model.loss_G.item()

        # Print epoch training time
        epoch_time = time.time() - start_time   
        print("End of epoch {} / {}, Time Taken: {:.0f} sec".format(epoch, cfg.epoch, epoch_time))

        # Save checkpoint for current epoch
        save_path = os.path.join(ckpt_path, "{}.pth".format(epoch))
        torch.save(model.cpu().state_dict(), save_path)
        model.to(device)

        # Update LR scheduler
        model.update_learning_rate()

        # Write average mini-batch loss
        writer.add_scalar('Loss/loss_D', loss_D / n, epoch)
        writer.add_scalar('Loss/loss_G', loss_G / n, epoch)



