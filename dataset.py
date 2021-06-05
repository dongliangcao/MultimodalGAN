import os

import numpy as np

from glob import glob

import torch
import torch.nn.functional as F
import torch.utils.data as data

import imageio

def read_image(filename):
    # rescale image from [0, 255] -> [-1, 1]
    return imageio.imread(filename).astype(np.float32) / np.float32(127.5) - 1.0

def read_mask(filename):
    return imageio.imread(filename)

def numpy2torch(array):
    assert(isinstance(array, np.ndarray))
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array.copy()).float()

def resize(img, output_size):
    """
    resize img/flow/occ to ensure the size is divisible by 64
    """
    # expand one dimension as dimension for batch size
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=output_size, mode='bilinear', align_corners=True)
    return img.squeeze(0)

class MRDataset(data.Dataset):
    """
    Dataset to read MR data in .png
    source data: T1, T2, target data: FLAIR, DIR 
    """
    def __init__(self, root, dtype='train'):
        if not os.path.isdir(root):
            raise ValueError(f'f{root} is not found')
        
        if dtype == 'train':
            self.T1_filenames = sorted(glob(os.path.join(root, 'trainT1', '*.png')))
            self.T2_filenames = sorted(glob(os.path.join(root, 'trainT2', '*.png')))
            self.FLAIR_filenames = sorted(glob(os.path.join(root, 'trainS1', '*.png')))
            self.DIR_filenames = sorted(glob(os.path.join(root, 'trainS2', '*.png')))
            self.mask_filenames = sorted(glob(os.path.join(root, 'train_mask', '*.png')))
        elif dtype == 'test':
            self.T1_filenames = sorted(glob(os.path.join(root, 'testT1', '*.png')))
            self.T2_filenames = sorted(glob(os.path.join(root, 'testT2', '*.png')))
            self.FLAIR_filenames = sorted(glob(os.path.join(root, 'testS1', '*.png')))
            self.DIR_filenames = sorted(glob(os.path.join(root, 'testS2', '*.png')))
            self.mask_filenames = sorted(glob(os.path.join(root, 'test_mask', '*.png')))
        else:
            raise ValueError(f'Unknwon dtype: {dtype}')

        # sanity check
        assert len(self.T1_filenames) != 0 and len(self.T1_filenames) == len(self.T2_filenames)
        assert len(self.T1_filenames) == len(self.FLAIR_filenames) and len(self.T1_filenames) == len(self.FLAIR_filenames)
        assert len(self.T1_filenames) == len(self.DIR_filenames) and len(self.T1_filenames) == len(self.mask_filenames)

        self._size = len(self.T1_filenames)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]
        else:
            T1 = read_image(self.T1_filenames[index])
            T2 = read_image(self.T2_filenames[index])
            FLAIR = read_image(self.FLAIR_filenames[index])
            DIR = read_image(self.DIR_filenames[index])
            mask = read_mask(self.mask_filenames[index])

            T1, T2, FLAIR, DIR, mask = numpy2torch(T1), numpy2torch(T2), numpy2torch(FLAIR), numpy2torch(DIR), numpy2torch(mask)
            real_A = torch.cat((T1, T2), dim=0)
            real_B = torch.cat((FLAIR, DIR), dim=0)
            return real_A, real_B, mask

class MRTrainDataset(MRDataset):
    def __init__(self, root):
        super().__init__(root, dtype='train')

class MRTestDataset(MRDataset):
    def __init__(self, root):
        super().__init__(root, dtype='test')
    