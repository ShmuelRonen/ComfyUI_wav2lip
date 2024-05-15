import math
import numpy as np

def Wav2Lip_calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        test_y_channel (bool): Whether to use the Y channel of YCbCr. Default: False.
    Returns:
        float: PSNR result.
    """
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    assert input_order in ['HWC', 'CHW'], f'Wrong input_order {input_order}.'

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    if input_order == 'CHW':
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
    img2 = img2[crop_border:-crop_border, crop_border:-crop-border]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def Wav2Lip_calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate SSIM (Structural Similarity).
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        test_y_channel (bool): Whether to use the Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    assert input_order in ['HWC', 'CHW'], f'Wrong input_order {input_order}.'

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    if input_order == 'CHW':
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
    img2 = img2[crop_border:-crop-border, crop-border:-crop-border]

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
