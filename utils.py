import torch
import torch.nn.functional as F

import numpy as np
import imageio
from PIL import Image

def read_image(filename):
    # rescale image from [0, 255] -> [-1, 1]
    return imageio.imread(filename).astype(np.float32) / np.float32(127.5) - 1.0

def read_mask(filename):
    return imageio.imread(filename)

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

def resize(img, size):
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=size, mode='bilinear', align_corners=True).squeeze(0)
    return img

def save_img(array, filename):
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array + 1.0) / 2.0 # rescale [-1, 1] -> [0, 1]
        array = (array * 255).astype(np.uint8) # rescale [0, 1] -> [0, 255]
        Image.fromarray(array).save(filename)
    else:
        Image.fromarray(array).save(filename)