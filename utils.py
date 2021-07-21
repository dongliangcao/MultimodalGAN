from numbers import Number
import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image

def read_image(filename):
    return np.array(Image.open(filename))

def read_mask(filename):
    return np.array(Image.open(filename))[..., None]

def numpy2torch(array):
    assert isinstance(array, np.ndarray), f'Unknown suppoerted type: {type(array)}'
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array.copy()).float()

def torch2numpy(tensor):
    assert isinstance(tensor, torch.Tensor), f'Unknown suppoerted type: {type(tensor)}'
    if tensor.ndim == 3:
        array = tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return array

def center_crop(img, size):
    if isinstance(size, Number):
        crop_h, crop_w = int(size), int(size)
    else:
        crop_h, crop_w = size
    
    h, w = img.shape[:2]
    pad_h, pad_w = (h - crop_h) // 2, (w - crop_w) // 2

    return img[pad_h:pad_h+crop_h, pad_w:pad_w+crop_w]


def save_img(array, filename):
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array + 1.0) / 2.0 # rescale [-1, 1] -> [0, 1]
        array = (array * 255).astype(np.uint8) # rescale [0, 1] -> [0, 255]
        Image.fromarray(array).save(filename)
    else:
        Image.fromarray(array).save(filename)