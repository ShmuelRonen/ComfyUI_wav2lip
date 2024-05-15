from copy import deepcopy

from custom_nodes.facerestore_cf.basicsr.utils import get_root_logger
from custom_nodes.facerestore_cf.basicsr.utils.registry import LOSS_REGISTRY
from .losses import (Wav2LipCharbonnierLoss, Wav2LipGANLoss, Wav2LipL1Loss, Wav2LipMSELoss, Wav2LipPerceptualLoss, Wav2LipWeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty)

__all__ = [
    'Wav2LipL1Loss', 'Wav2LipMSELoss', 'Wav2LipCharbonnierLoss', 'Wav2LipWeightedTVLoss', 'Wav2LipPerceptualLoss', 'Wav2LipGANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize'
]

def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
