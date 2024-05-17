import math
import lpips
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from custom_nodes.ComfyUI_wav2lip.basicsr.archs.vgg_arch import VGGFeatureExtractor
from custom_nodes.ComfyUI_wav2lip.basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

@LOSS_REGISTRY.register()
class Wav2LipL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Wav2LipL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class Wav2LipMSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Wav2LipMSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class Wav2LipCharbonnierLoss(nn.Module):
    """Charbonnier loss (L1).

    Args:
        loss_weight (float): Loss weight for Charbonnier loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the numerical stability to avoid
            NaNs. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(Wav2LipCharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * charbonnier_loss(pred, target, self.eps, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class Wav2LipPerceptualLoss(nn.Module):
    """Perceptual loss with VGG feature extraction.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
        vgg_type (str): The type of VGG network used for feature extraction.
            Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image before VGG
            feature extraction. Default: True.
        range_norm (bool): If True, normalize the input image to [0, 1].
            Default: False.
        perceptual_weight (float): If True, use perceptual loss.
            Default: 1.0.
        style_weight (float): If True, use style loss. Default: 0.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.):
        super(Wav2LipPerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)
        self.vgg.eval()
        for k, v in self.vgg.named_parameters():
            v.requires_grad = False

    def forward(self, pred, target):
        if self.perceptual_weight > 0:
            percep_loss = self.perceptual_weight * self.vgg(pred)
        else:
            percep_loss = None

        if self.style_weight > 0:
            style_loss = self.style_weight * self.vgg(target)
        else:
            style_loss = None

        return percep_loss + style_loss

@LOSS_REGISTRY.register()
class Wav2LipGANLoss(nn.Module):
    """GAN loss.

    Args:
        gan_type (str): Type of GAN loss. Support 'vanilla', 'lsgan', 'wgan',
            'wgan_softplus'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight for GAN loss. Default: 1.0.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(Wav2LipGANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        else:
            raise NotImplementedError(f'GAN type {gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def forward(self, input, target_is_real):
        target_label = self.real_label_val if target_is_real else self.fake_label_val
        return self.loss(input, torch.full_like(input, target_label)) * self.loss_weight

@LOSS_REGISTRY.register()
class Wav2LipWeightedTVLoss(nn.Module):
    """Weighted Total Variation Loss.
    
    Args:
        loss_weight (float): Loss weight for WeightedTVLoss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Wav2LipWeightedTVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        
    def forward(self, pred, weight=None):
        return self.loss_weight * l1_loss(pred, torch.zeros_like(pred), weight, reduction=self.reduction)

def r1_penalty(real_img, real_pred):
    """R1 regularization for discriminator.

    Args:
        real_img (Tensor): Real images.
        real_pred (Tensor): Real predictions.

    Returns:
        Tensor: R1 penalty.
    """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()

def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """
    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty
