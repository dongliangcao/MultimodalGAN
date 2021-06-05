from train import train
from model import CycleGAN
from dataset import MRTrainDataset, MRTestDataset

import os
import argparse

import torch
from torch.utils.data import DataLoader

if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='data root', default='MR/')
    parser.add_argument('--model_root', help='model root', default='checkpoints/')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=60)
    parser.add_argument('--mode', help='mode: train/test', default='train')
    args = parser.parse_args()
    mode = args.mode
    model_root = args.model_root
    max_epoch = args.epochs
    root = args.data_root

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == 'train':
        # log file
        if not os.path.isdir('./log/'):
            os.mkdir('./log/')
        # checkpoint file
        if not os.path.isdir('./checkpoints/'):
            os.mkdir('./checkpoints/')
        # prepare data
        train_dataset = MRTrainDataset(root=root)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)
        # prepare model
        model = CycleGAN().to(device)
        # start training
        train(model, train_dataloader, max_epoch, device)
    elif mode == 'test':
        # TODO
        pass
    else:
        raise ValueError(f'Unknown mode: {mode}')

    

    
