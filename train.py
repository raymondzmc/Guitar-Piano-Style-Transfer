import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time

from dataloader import AudioDataset, get_dataloader
from loss import CycleGANModel
from utils import parse_arg
import config as cfg
import pdb


if __name__ == "__main__":

    args = parse_arg()

    load_ckpt = args.load_ckpt
    
    if torch.cuda.is_available():
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

    # Load previous checkpoint
    if load_ckpt > 0:
        ckpt_file = os.path.join(cfg.ckpt, "{}.pth".format(load_ckpt))
        model.load_state_dict(torch.load(ckpt_file))
        print("Loaded checkpoint at epoch {}".format(load_ckpt))

    # Initialize dataloader
    dataset = AudioDataset(cfg.data_path)
    dataloader = get_dataloader(dataset)

    n = len(dataloader)

    writer = SummaryWriter()


    # Training loop
    for epoch in range(load_ckpt + 1, cfg.epoch + 1):
        start_time = time.time()

        # Keep track of batch loss for discriminator and generator
        loss_Fy, loss_Gx, loss_cyc_x, loss_cyc_y, loss_idt_x, loss_idt_y, loss_Dx, loss_Dy = 8 * [0.]

        for i, (x, y) in enumerate(dataloader):
            if (i + 1) % 100 == 0:
                print("Minibatch {} of {}".format(i + 1, n))

            x, y = x.to(device), y.to(device)

            try:
                model.optimize_parameters(x, y)
            except:
                pdb.set_trace()

            loss_Fy += model.loss_Fy.item()
            loss_Gx += model.loss_Gx.item()
            loss_cyc_x += model.loss_cyc_x.item()
            loss_cyc_y += model.loss_cyc_y.item()
            loss_idt_x += model.loss_idt_x.item()
            loss_idt_y += model.loss_idt_y.item()
            loss_Dx += model.loss_Dx.item()
            loss_Dy += model.loss_Dy.item()

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
        writer.add_scalar('Loss/y_to_x_loss', loss_Fy / n, epoch)
        writer.add_scalar('Loss/x_to_y_loss', loss_Gx / n, epoch)
        writer.add_scalar('Loss/x_cycle_loss', loss_cyc_x / n, epoch)
        writer.add_scalar('Loss/y_cycle_loss', loss_cyc_y / n, epoch)
        writer.add_scalar('Loss/x_identity_loss', loss_idt_x / n, epoch)
        writer.add_scalar('Loss/y_identity_loss', loss_idt_y / n, epoch)
        writer.add_scalar('Loss/x_discriminator_loss', loss_Dx / n, epoch)
        writer.add_scalar('Loss/y_discriminator_loss', loss_Dy / n, epoch)