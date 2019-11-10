import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import itertools
from model import * 
from utils import *
import config as cfg
import pdb


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class CycleGANModel(nn.Module):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, verbose=False, last_epoch=0):
        """Initialize the CycleGAN class."""

        super(CycleGANModel, self).__init__()

        self.verbose = verbose

        # Define generators F(y) = x, G(x) = y
        self.Fy = ResnetGenerator(cfg.input_nc, cfg.output_nc)
        self.Gx = ResnetGenerator(cfg.input_nc, cfg.output_nc)


        # Define discriminators
        self.Dy = NLayerDiscriminator(cfg.input_nc)
        self.Dx = NLayerDiscriminator(cfg.input_nc)

        # Initialize weights
        init_weights(self.Fy)
        init_weights(self.Gx)
        init_weights(self.Dy)
        init_weights(self.Dx)

        # Define image pools
        self.fake_x_pool = ImagePool()  # create image buffer to store previously generated images
        self.fake_y_pool = ImagePool()

        # Define criterions
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionGAN = GANLoss()

        # Define optimizers
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.Fy.parameters(), self.Gx.parameters()), lr=cfg.init_lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.Dy.parameters(), self.Dx.parameters()), lr=cfg.init_lr, betas=(0.5, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        # Lambda parameters for loss
        self.lambda_cyc = cfg.lambda_cyc
        self.lambda_idt = cfg.lambda_idt

        # LR scheduler
        def lr_lambda(epoch):
            lr_l = 1.0 - (max(0, epoch + last_epoch - 100) / float(101))
            return lr_l

        self.schedulers = []
        for optim in self.optimizers:
            self.schedulers.append(lr_scheduler.LambdaLR(optim, lr_lambda))

        print("Initial learning rate: {:.6f}".format(self.optimizers[0].param_groups[0]['lr']))

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x, y):
        self.real_x = x
        self.fake_y = self.Gx(self.real_x)
        self.rec_x = self.Fy(self.fake_y)

        self.real_y = y
        self.fake_x = self.Fy(self.real_y)
        self.rec_y = self.Gx(self.fake_x)

    def optimize_parameters(self, x, y):
        """Calculate losses, gradients, and update network weights"""

        # Forward step
        self.forward(x, y)

        # Optimize for the Generators
        self.set_requires_grad([self.Dx, self.Dy], False)
        self.optimizer_G.zero_grad()

        # Identity Loss: ||Fy(x) - x|| and ||Gx(y) - y||
        idt_x = self.Fy(self.real_x)
        loss_idt_x = self.criterionIdt(idt_x, self.real_x) * self.lambda_cyc * self.lambda_idt
        idt_y = self.Gx(self.real_y)
        loss_idt_y = self.criterionIdt(idt_y, self.real_y) * self.lambda_cyc * self.lambda_idt
        
        
        # GAN Loss: Dx(Fy(y)) and Dy(Gx(x))
        loss_Fy = self.criterionGAN(self.Dx(self.fake_x), True)
        loss_Gx = self.criterionGAN(self.Dy(self.fake_y), True)
        

        # Cycle-Consistency Loss: ||Gx(Dy(y)) - y|| and ||Fy(Gx(x)) - x|| 
        loss_cyc_x = self.criterionCycle(self.rec_x, self.real_x) * self.lambda_cyc
        loss_cyc_y = self.criterionCycle(self.rec_y, self.real_y) * self.lambda_cyc
        

        # Compute gradients and update weights for the generator
        self.loss_G = loss_idt_x + loss_idt_y + loss_Fy + loss_Gx + loss_cyc_x + loss_cyc_y
        self.loss_G.backward()
        self.optimizer_G.step()
        
        ######################################################################################################

        # Optimize for the Discriminators
        self.set_requires_grad([self.Dx, self.Dy], True)
        self.optimizer_D.zero_grad()

        # Loss for Dx
        fake_x = self.fake_x_pool.query(self.fake_x)
        loss_Dx_real = self.criterionGAN(self.Dx(self.real_x), True)
        loss_Dx_fake = self.criterionGAN(self.Dx(fake_x.detach()), False)
        loss_Dx = (loss_Dx_real + loss_Dx_fake) * 0.5

        # Loss for Dy
        fake_y = self.fake_y_pool.query(self.fake_y)
        loss_Dy_real = self.criterionGAN(self.Dx(self.real_y), True)
        loss_Dy_fake = self.criterionGAN(self.Dx(fake_y.detach()), False)
        loss_Dy = (loss_Dy_real + loss_Dy_fake) * 0.5

        # Compute gradients and update weights for the discriminator
        self.loss_D = loss_Dx + loss_Dy
        self.loss_D.backward()
        self.optimizer_D.step()

        if self.verbose:
            print("Identity loss for x: {:.3f}, y: {:.3f}".format(loss_idt_x.item(), loss_idt_y.item()))
            print("Generator loss for Fy: {:.3f}, Gx: {:.3f}".format(loss_Fy.item(), loss_Gx.item()))
            print("Cycle-consistent loss for x: {:.3f}, y: {:.3f}".format(loss_cyc_x.item(), loss_cyc_y.item()))
            print("Discriminator loss for x: {:.3f}, y: {:.3f}".format(loss_Dx.item(), loss_Dy.item()))


    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:.6f}'.format(lr))

# if __name__ == "__main__":
#     cyclegan = CycleGANModel().cuda()
#     x, y = torch.Tensor(1,1,128,128).cuda(), torch.Tensor(1,1,128,128).cuda()
#     pdb.set_trace()
#     cyclegan.optimize_parameters(x,y)