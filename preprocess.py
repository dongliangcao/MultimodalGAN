"""
Convert NIfTI files to png images
"""

import os
import SimpleITK as sitk
from PIL import Image
import numpy as np
from tqdm import tqdm

def min_max_norm(array):
    '''
    :function: normalize the intensities to [0, 255]
    :input: array with any number of dimensions
    :return: normalized array with int8 format
    '''
    array = (array - np.min(array))/(np.max(array) - np.min(array))
    return np.uint8(255*array)

if __name__ == '__main__':
    # process training data [155, 240, 240]
    train_dir = '../data/train'
    save_dir_t1 = '../data/trainT1'
    save_dir_t2 = '../data/trainT2'
    save_dir_t1ce = '../data/trainT1CE'
    save_dir_flair = '../data/trainFlair'
    save_dir_mask = '../data/trainMask'
    
    os.makedirs(save_dir_t1, exist_ok=True)
    os.makedirs(save_dir_t2, exist_ok=True)
    os.makedirs(save_dir_t1ce, exist_ok=True)
    os.makedirs(save_dir_flair, exist_ok=True)
    os.makedirs(save_dir_mask, exist_ok=True)

    folder_ls = os.listdir(train_dir)
    for folder in tqdm(folder_ls):
        if os.path.isdir(os.path.join(train_dir, folder)):
            # make dir
            os.makedirs(os.path.join(save_dir_t1, folder), exist_ok=True)
            os.makedirs(os.path.join(save_dir_t2, folder), exist_ok=True)
            os.makedirs(os.path.join(save_dir_t1ce, folder), exist_ok=True)
            os.makedirs(os.path.join(save_dir_flair, folder), exist_ok=True)
            os.makedirs(os.path.join(save_dir_mask, folder), exist_ok=True)
            
            # find data
            t1 = os.path.join(train_dir, folder, folder + '_t1.nii')
            t1ce = os.path.join(train_dir, folder, folder + '_t1ce.nii')
            t2 = os.path.join(train_dir, folder, folder + '_t2.nii')
            flair = os.path.join(train_dir, folder, folder + '_flair.nii')
            mask = os.path.join(train_dir, folder, folder + '_seg.nii')

            assert os.path.isfile(t1) and os.path.isfile(t1ce) and os.path.isfile(t2) and os.path.isfile(flair) and os.path.isfile(mask), f'{t1}, {t1ce}, {t2}, {flair}, {mask}'

            # read data
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1))
            t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2))
            flair = sitk.GetArrayFromImage(sitk.ReadImage(flair))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask))

            # process data
            t1, t1ce, t2, flair = min_max_norm(t1), min_max_norm(t1ce), min_max_norm(t2), min_max_norm(flair)

            # save slices
            for i, ss in enumerate(range(np.shape(t1)[0])):
                if np.sum(mask[ss]) > 0:
                    Image.fromarray(t1[ss]).save(os.path.join(save_dir_t1, folder, f'{i}.png'))
                    Image.fromarray(t2[ss]).save(os.path.join(save_dir_t2, folder, f'{i}.png'))
                    Image.fromarray(t1ce[ss]).save(os.path.join(save_dir_t1ce, folder, f'{i}.png'))
                    Image.fromarray(flair[ss]).save(os.path.join(save_dir_flair, folder, f'{i}.png'))
                    Image.fromarray(mask[ss]).save(os.path.join(save_dir_mask, folder, f'{i}.png'))

    # process validation data
    val_dir = '../data/val'
    save_dir_t1 = '../data/testT1'
    save_dir_t2 = '../data/testT2'
    save_dir_t1ce = '../data/testT1CE'
    save_dir_flair = '../data/testFlair'

    os.makedirs(save_dir_t1, exist_ok=True)
    os.makedirs(save_dir_t2, exist_ok=True)
    os.makedirs(save_dir_t1ce, exist_ok=True)
    os.makedirs(save_dir_flair, exist_ok=True)

    folder_ls = os.listdir(val_dir)
    for folder in tqdm(folder_ls):
        if os.path.isdir(os.path.join(val_dir, folder)):
            # make dir
            os.makedirs(os.path.join(save_dir_t1, folder), exist_ok=True)
            os.makedirs(os.path.join(save_dir_t2, folder), exist_ok=True)
            os.makedirs(os.path.join(save_dir_t1ce, folder), exist_ok=True)
            os.makedirs(os.path.join(save_dir_flair, folder), exist_ok=True)

            t1 = os.path.join(val_dir, folder, folder + '_t1.nii')
            t1ce = os.path.join(val_dir, folder, folder + '_t1ce.nii')
            t2 = os.path.join(val_dir, folder, folder + '_t2.nii')
            flair = os.path.join(val_dir, folder, folder + '_flair.nii')

            assert os.path.isfile(t1) and os.path.isfile(t1ce) and os.path.isfile(t2) and os.path.isfile(flair), f'{t1}, {t1ce}, {t2}, {flair}'

            # read data
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1))
            t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2))
            flair = sitk.GetArrayFromImage(sitk.ReadImage(flair))

            # process data
            t1, t1ce, t2, flair = min_max_norm(t1), min_max_norm(t1ce), min_max_norm(t2), min_max_norm(flair)

            # save slices
            for i, ss in enumerate(range(np.shape(t1)[0])):
                if np.sum(t1[ss] != 0) > 3000:
                    Image.fromarray(t1[ss]).save(os.path.join(save_dir_t1, folder, f'{i}.png'))
                    Image.fromarray(t2[ss]).save(os.path.join(save_dir_t2, folder, f'{i}.png'))
                    Image.fromarray(t1ce[ss]).save(os.path.join(save_dir_t1ce, folder, f'{i}.png'))
                    Image.fromarray(flair[ss]).save(os.path.join(save_dir_flair, folder, f'{i}.png'))