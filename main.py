from train import train
from model import CycleGAN, Generator
from dataset import MRTrainDataset, MRTestDataset
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *

if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='data/')
    parser.add_argument('--model_root', help='model root', default='checkpoints/G_A2B_60.pth')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=120)
    parser.add_argument('--mode', help='mode: train/test', choices=['train', 'test'], default='train')
    parser.add_argument('--output_root', help='output root', default='output/')
    parser.add_argument('--ssim', help='use ssim instead of l1', action='store_true')
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    args = parser.parse_args()

    mode = args.mode
    model_root = args.model_root
    data_root = args.data_root
    output_root = args.output_root
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config (can be modified)
    config = {
        'max_epoch': args.epochs,
        'device': device,
        'verbose': True,
        'print_every_step': 10,
        'identity_learning': False,
        'supervised_learning': True,
        'ssim': args.ssim
    }
    if mode == 'train':
        # log file
        if not os.path.isdir('./logs/'):
            os.mkdir('./logs/')
        # checkpoint file
        if not os.path.isdir('./checkpoints/'):
            os.mkdir('./checkpoints/')
        # prepare data
        train_dataset = MRTrainDataset(root=data_root)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # prepare model
        model = CycleGAN().to(device)
        # start training
        train(model, train_dataloader, config)
    elif mode == 'test':
        # output dir
        real_dir = os.path.join(output_root, 'real')
        syn_dir = os.path.join(output_root, 'synthetic')
        real_FLAIR_dir = os.path.join(real_dir, 'FLAIR')
        real_T2_dir = os.path.join(real_dir, 'T2')
        syn_FLAIR_dir = os.path.join(syn_dir, 'FLAIR')
        syn_T2_dir = os.path.join(syn_dir, 'T2')
        os.makedirs(real_FLAIR_dir, exist_ok=True)
        os.makedirs(real_T2_dir, exist_ok=True)
        os.makedirs(syn_FLAIR_dir, exist_ok=True)
        os.makedirs(syn_T2_dir, exist_ok=True)

        # use G_A2B to generate synthetic images
        G_A2B = Generator(2).to(device)
        # load pre-trained model
        G_A2B.load_state_dict(torch.load(model_root))

        # prepare data
        test_dataset = MRTestDataset(root=data_root)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        # start inference
        t2_psnr_avg, t2_ssim_avg = 0.0, 0.0
        flair_psnr_avg, flair_ssim_avg = 0.0, 0.0
        G_A2B.eval()
        with tqdm(total=len(test_dataloader)) as pbar:
            for i, batch in enumerate(test_dataloader):
                real_A, real_B = batch
                real_A, real_B = real_A.to(device), real_B.to(device)
                with torch.no_grad():
                    synthetic_B, _ = G_A2B(real_A)
                real_B_np, synthetic_B_np = torch2numpy(real_B), torch2numpy(synthetic_B)
                # calculate PSNR and SSIM
                t2_psnr = psnr(real_B_np[:, :, 0], synthetic_B_np[:, :, 0])
                t2_psnr_avg += t2_psnr
                t2_ssim = ssim(real_B_np[:, :, 0], synthetic_B_np[:, :, 0])
                t2_ssim_avg += t2_ssim
                flair_psnr = psnr(real_B_np[:, :, 1], synthetic_B_np[:, :, 1])
                flair_psnr_avg += flair_psnr
                flair_ssim = ssim(real_B_np[:, :, 1], synthetic_B_np[:, :, 1])
                flair_ssim_avg += flair_ssim

                save_img(real_B_np[:, :, 1], os.path.join(real_FLAIR_dir, f'FLAIR_{i}.png'))
                save_img(real_B_np[:, :, 0], os.path.join(real_T2_dir, f'T2_{i}.png'))
                save_img(synthetic_B_np[:, :, 1], os.path.join(syn_FLAIR_dir, f'FLAIR_{i}.png'))
                save_img(synthetic_B_np[:, :, 0], os.path.join(syn_T2_dir, f'T2_{i}.png'))

                pbar.update(1)
                pbar.set_description(f'T2: PSNR {t2_psnr:.4f} SSIM {t2_ssim:.4f}, FLAIR: PSNR {flair_psnr:.4f} SSIM {flair_ssim:.4f}')

        t2_psnr_avg /= len(test_dataloader)
        t2_ssim_avg /= len(test_dataloader)
        flair_psnr_avg /= len(test_dataloader)
        flair_ssim_avg /= len(test_dataloader)

        print(f'T2, PSNR: {t2_psnr_avg:.4f}, SSIM: {t2_ssim_avg:.4f}')
        print(f'FLAIR, PSNR: {flair_psnr_avg:.4f}, SSIM: {flair_ssim_avg:.4f}')