from .psnr_ssim import Wav2Lip_calculate_psnr, Wav2Lip_calculate_ssim

__all__ = ['Wav2Lip_calculate_psnr', 'Wav2Lip_calculate_ssim']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
