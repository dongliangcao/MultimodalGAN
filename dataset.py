import os
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import *

class MRDataset(data.Dataset):
    """
    Dataset to read MR data in .png
    source data: T1, T2, target data: FLAIR, DIR 
    """
    def __init__(self, root, dtype='train'):
        if not os.path.isdir(root):
            raise ValueError(f'f{root} is not found')
        
        self.dtype = dtype
        
        if dtype == 'train':
            self.T1_filenames = sorted(glob(os.path.join(root, 'trainT1') + '/*/*.png'))
            self.T2_filenames = sorted(glob(os.path.join(root, 'trainT2') + '/*/*.png'))
            self.T1CE_filenames = sorted(glob(os.path.join(root, 'trainT1CE') + '/*/*.png'))
            self.Flair_filenames = sorted(glob(os.path.join(root, 'trainFlair') + '/*/*.png'))
            self.mask_filenames = sorted(glob(os.path.join(root, 'trainMask') + '/*/*.png'))
        elif dtype == 'test':
            self.T1_filenames = sorted(glob(os.path.join(root, 'testT1') + '/*/*.png'))
            self.T2_filenames = sorted(glob(os.path.join(root, 'testT2') + '/*/*.png'))
            self.T1CE_filenames = sorted(glob(os.path.join(root, 'testT1CE') + '/*/*.png'))
            self.Flair_filenames = sorted(glob(os.path.join(root, 'testFlair') + '/*/*.png'))
        else:
            raise ValueError(f'Unknwon dtype: {dtype}')

        # sanity check
        assert len(self.T1_filenames) != 0 and len(self.T1_filenames) == len(self.T2_filenames)
        assert len(self.T1_filenames) == len(self.T1CE_filenames) 

        self._size = len(self.T1_filenames)

        # data transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]
        else:
            T1 = read_image(self.T1_filenames[index])
            T2 = read_image(self.T2_filenames[index])
            T1CE = read_image(self.T1CE_filenames[index])
            Flair = read_image(self.Flair_filenames[index])
            
            if self.dtype == 'train':
                mask = read_mask(self.mask_filenames[index])
                # perform center crop
                size = (196, 196)
                T1, T2, T1CE, Flair, mask = center_crop(T1, size), center_crop(T2, size), center_crop(T1CE, size), center_crop(Flair, size), center_crop(mask, size)
                mask = numpy2torch(mask)

            # perform transform
            img = np.stack((T1, T2, T1CE, Flair), axis=-1)
            aug_img = self.transform(img)
            T1, T2, T1CE, Flair = aug_img[0], aug_img[1], aug_img[2], aug_img[3]
            
            real_A = torch.stack((T1, T1CE), axis=0)
            real_B = torch.stack((T2, Flair), axis=0)

            if self.dtype == 'train':
                return real_A, real_B, mask
            else:
                return real_A, real_B

class MRTrainDataset(MRDataset):
    def __init__(self, root):
        super().__init__(root, dtype='train')

class MRTestDataset(MRDataset):
    def __init__(self, root):
        super().__init__(root, dtype='test')