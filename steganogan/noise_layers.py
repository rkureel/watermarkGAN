import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


def dropout(cover, encoded, device, p=0.3):
    mask = np.random.choice([0.0, 1.0], encoded.shape[2:], p=[1-p, p])
    mask_tensor = torch.tensor(mask, device=device, dtype=torch.float)
    mask_tensor = mask_tensor.expand_as(encoded)
    noised_image = encoded*mask_tensor + cover*(1-mask_tensor)
    return noised_image

def gaussian_blur(encoded, kernelSize=3,sigma=(0.1, 2.0)):
    kernel = transforms.GaussianBlur(kernelSize, sigma)
    return kernel.forward(encoded)

def random_float(min, max):
    return np.random.rand() * (max - min) + min
  
def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width
  
def cropout(cover, encoded, device, height_ratio=(0.55, 0.6), width_ratio=(0.55,0.6), p=0.3):
    cropout_mask = torch.zeros_like(encoded)
    h_start, h_end, w_start, w_end = get_random_rectangle_inside(encoded, height_ratio, width_ratio)
    cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

    noised_image = encoded * cropout_mask + cover * (1-cropout_mask)
    return noised_image

def random_float(min, max):
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)
    
    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

def crop(cover, encoded, device, height_ratio_range=(0.55, 0.6), width_ratio_range=(0.55, 0.6)):
    cropout_mask = torch.zeros_like(encoded)
    h_start, h_end, w_start, w_end = get_random_rectangle_inside(encoded, height_ratio_range, width_ratio_range)
    cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
    noised_image = encoded * cropout_mask
    return noised_image

def gen_filters(size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:
    tile_size_x = 8
    filters = np.zeros((size_x * size_y, size_x, size_y))
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(n_x,
                                                                                                            k_x,
                                                                                                            size_x)
    return filters

def get_jpeg_yuv_filter_mask(image_shape: tuple, window_size: int, keep_count: int):
    mask = np.zeros((window_size, window_size), dtype=np.uint8)

    index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

    for i, j in index_order[0:keep_count]:
        mask[i, j] = 1

    return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                          int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]


def dct_coeff(n, k, N):
    return np.cos(np.pi / N * (n + 1. / 2.) * k)


def idct_coeff(n, k, N):
    return (int(0 == n) * (- 1 / 2) + np.cos(
        np.pi / N * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * N))


def rgb2yuv(image_rgb, image_yuv_out):
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :].clone() + 0.587 * image_rgb[:, 1, :, :].clone() + 0.114 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :].clone() + -0.28886 * image_rgb[:, 1, :, :].clone() + 0.436 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :].clone() + -0.51499 * image_rgb[:, 1, :, :].clone() + -0.10001 * image_rgb[:, 2, :, :].clone()


def yuv2rgb(image_yuv, image_rgb_out):
    image_rgb_out[:, 0, :, :] = image_yuv[:, 0, :, :].clone() + 1.13983 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 1, :, :] = image_yuv[:, 0, :, :].clone() + -0.39465 * image_yuv[:, 1, :, :].clone() + -0.58060 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 2, :, :] = image_yuv[:, 0, :, :].clone() + 2.03211 * image_yuv[:, 1, :, :].clone()

def create_mask(requested_shape, jpeg_mask, device, yuv_keep_weights):
    if jpeg_mask is None or requested_shape > jpeg_mask.shape[1:]:
        jpeg_mask = torch.empty((3,) + requested_shape, device=device)
        for channel, weights_to_keep in enumerate(yuv_keep_weights):
            mask = torch.from_numpy(get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
            jpeg_mask[channel] = mask
    return jpeg_mask

def get_mask(image_shape, jpeg_mask, device, yuv_keep_weights):
    if jpeg_mask.shape < image_shape:
        jpeg_mask = create_mask(image_shape, jpeg_mask, device, yuv_keep_weights)
    return jpeg_mask[:, :image_shape[1], :image_shape[2]].clone()

def apply_conv(image, filter_type, dct_conv_weights, idct_conv_weights):
    if filter_type == 'dct':
        filters = dct_conv_weights
    elif filter_type == 'idct':
        filters = idct_conv_weights
    else:
        raise('Unknown filter_type value.')

    image_conv_channels = []
    for channel in range(image.shape[1]):
        image_yuv_ch = image[:, channel, :, :].unsqueeze_(1).clone()
        image_conv = F.conv2d(image_yuv_ch, filters, stride=8).clone()
        image_conv = image_conv.permute(0, 2, 3, 1)
        image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2], 8, 8)
        image_conv = image_conv.permute(0, 1, 3, 2, 4)
        image_conv = image_conv.contiguous().view(image_conv.shape[0],
                                                image_conv.shape[1]*image_conv.shape[2],
                                                image_conv.shape[3]*image_conv.shape[4])

        image_conv.unsqueeze_(1)
        image_conv_channels.append(image_conv)

    image_conv_stacked = torch.cat(image_conv_channels, dim=1)
    return image_conv_stacked

def jpeg_compress(cover, encoded, device, yuv_keep_weights = (25, 9, 9)):


    dct_conv_weights = torch.tensor(gen_filters(8, 8, dct_coeff), dtype=torch.float32).to(device)
    dct_conv_weights.unsqueeze_(1)
    idct_conv_weights = torch.tensor(gen_filters(8, 8, idct_coeff), dtype=torch.float32).to(device)
    idct_conv_weights.unsqueeze_(1)
    
    keep_coeff_masks = []
    jpeg_mask = None
    jpeg_mask = create_mask((1000, 1000), jpeg_mask, device, yuv_keep_weights)

    pad_height = (8 - encoded.shape[2] % 8) % 8
    pad_width = (8 - encoded.shape[3] % 8) % 8
    noised_image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(encoded)
    image_yuv = torch.empty_like(noised_image)
    rgb2yuv(noised_image, image_yuv)

    assert image_yuv.shape[2] % 8 == 0
    assert image_yuv.shape[3] % 8 == 0

    # apply dct
    image_dct = apply_conv(image_yuv, 'dct', dct_conv_weights, idct_conv_weights)
    # get the jpeg-compression mask
    mask = get_mask(image_dct.shape[1:], jpeg_mask, device, yuv_keep_weights)
    # multiply the dct-ed image with the mask.
    image_dct_mask = torch.mul(image_dct, mask)

    # apply inverse dct (idct)
    image_idct = apply_conv(image_dct_mask, 'idct', dct_conv_weights, idct_conv_weights)
    # transform from yuv to to rgb
    image_ret_padded = torch.empty_like(image_dct)
    yuv2rgb(image_idct, image_ret_padded)

    # un-pad
    noised = image_ret_padded[:, :, :image_ret_padded.shape[2]-pad_height, :image_ret_padded.shape[3]-pad_width].clone()

    return noised