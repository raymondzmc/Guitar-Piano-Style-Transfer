import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import grad
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

    def __init__(self, lr_lambda=None, verbose=False, last_epoch=0):
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
        if last_epoch == 0:
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

        self.schedulers = []

        # LR scheduler
        self.last_epoch = last_epoch
        def lr_lambda(epoch):
            lr_l = 1.0 - (max(0, epoch + self.last_epoch - 2400) / float(cfg.epoch - 2400))
            return lr_l

        if lr_lambda != None:
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
        self.loss_idt_x = self.criterionIdt(idt_x, self.real_x) * self.lambda_cyc * self.lambda_idt
        idt_y = self.Gx(self.real_y)
        self.loss_idt_y = self.criterionIdt(idt_y, self.real_y) * self.lambda_cyc * self.lambda_idt

        # GAN Loss: Dx(Fy(y)) and Dy(Gx(x))
        self.loss_Fy = self.criterionGAN(self.Dx(self.fake_x), True)
        self.loss_Gx = self.criterionGAN(self.Dy(self.fake_y), True)

        # Cycle-Consistency Loss: ||Gx(Dy(y)) - y|| and ||Fy(Gx(x)) - x|| 
        self.loss_cyc_x = self.criterionCycle(self.rec_x, self.real_x) * self.lambda_cyc
        self.loss_cyc_y = self.criterionCycle(self.rec_y, self.real_y) * self.lambda_cyc

        # Compute gradients and update weights for the generator
        self.loss_G = self.loss_idt_x + self.loss_idt_y + self.loss_Fy + self.loss_Gx + self.loss_cyc_x + self.loss_cyc_y
        self.loss_G.backward()
        self.optimizer_G.step()

        ######################################################################################################

        # Optimize for the Discriminators
        self.set_requires_grad([self.Dx, self.Dy], True)
        self.optimizer_D.zero_grad()

        # Loss for Dx
        fake_x = self.fake_x_pool.query(self.fake_x)
        self.loss_Dx_real = self.criterionGAN(self.Dx(self.real_x), True)
        self.loss_Dx_fake = self.criterionGAN(self.Dx(fake_x.detach()), False)

        # Gradient Penalty for Dx
        self.gp_dx, gradients = self.gradient_penalty(self.Dx, self.real_x, fake_x, 'cuda')
        self.loss_Dx = (self.loss_Dx_real + self.loss_Dx_fake + self.gp_dx) * 0.5


        # Loss for Dy
        fake_y = self.fake_y_pool.query(self.fake_y)
        self.loss_Dy_real = self.criterionGAN(self.Dy(self.real_y), True)
        self.loss_Dy_fake = self.criterionGAN(self.Dy(fake_y.detach()), False)

        # Gradient Penalty for Dy
        self.gp_dy, gradients = self.gradient_penalty(self.Dy, self.real_y, fake_y, 'cuda')
        self.loss_Dy = (self.loss_Dy_real + self.loss_Dy_fake + self.gp_dy) * 0.5


        # Compute gradients and update weights for the discriminator
        self.loss_D = self.loss_Dx + self.loss_Dy
        self.loss_D.backward()
        self.optimizer_D.step()

        if self.verbose:
            print("Identity loss for x: {:.3f}, y: {:.3f}".format(loss_idt_x.item(), loss_idt_y.item()))
            print("Generator loss for Fy: {:.3f}, Gx: {:.3f}".format(loss_Fy.item(), loss_Gx.item()))
            print("Cycle-consistent loss for x: {:.3f}, y: {:.3f}".format(loss_cyc_x.item(), loss_cyc_y.item()))
            print("Discriminator loss for x: {:.3f}, y: {:.3f}".format(loss_Dx.item(), loss_Dy.item()))


    def gradient_penalty(self, net, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
        alpha = torch.rand(real_data.shape[0], 1, device=device)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        interpolatesv.requires_grad_()
        disc_interpolates = net(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:.7f}'.format(lr))

# if __name__ == "__main__":
#     cyclegan = CycleGANModel().cuda()
#     x, y = torch.Tensor(1,1,128,128).cuda(), torch.Tensor(1,1,128,128).cuda()
#     pdb.set_trace()
#     cyclegan.optimize_parameters(x,y)