from collections import OrderedDict
import os, csv, time

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from utils import ImagePool

DECAY_EPOCH = 101
DATE_TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime())

def train(model, train_dataloader, max_epoch, device='cuda', verbose=True, print_every_step=1):
    # log losses during training
    training_history = OrderedDict()
    logger = SummaryWriter(f'./logs/{DATE_TIME}/')
    DA_losses = []
    DB_losses = []
    gA_d_losses_synthetic = []
    gB_d_losses_synthetic = []
    gA_losses_reconstructed = []
    gB_losses_reconstructed = []

    GA_losses = []
    GB_losses = []
    reconstruction_losses = []
    D_losses = []
    G_losses = []

    # image pools used to update the discriminators
    # synthetic_pool_size = 25
    # synthetic_pool_A = ImagePool(synthetic_pool_size)
    # synthetic_pool_B = ImagePool(synthetic_pool_size)

    # loss criterion
    loss_D = F.mse_loss
    loss_weights_D = 0.5

    # cyclic loss weight A2B, cyclic loss weight B2A, cyclic loss with attention B2A, loss for discriminator
    loss_G = [F.l1_loss, F.l1_loss, F.l1_loss, F.mse_loss, F.mse_loss]
    loss_weights_G = [10.0, 8.0, 10.0, 1.0, 1.0]

    # identity mapping will be done each time the iteration number is divisable with this number
    identity_mapping_modulus = 10
    loss_I = F.l1_loss
    loss_weights_I = 1.0

    # optimizer
    lr_D = 2e-4
    lr_G = 3e-4
    optimizer_D = optim.Adam([
                {'params': model.D_A.parameters()},
                {'params': model.D_B.parameters(), 'lr': 1e-3}
            ], lr=lr_D, betas=(0.5, 0.999))
    optimizer_G = optim.Adam([
                {'params': model.G_A2B.parameters()},
                {'params': model.G_B2A.parameters(), 'lr': 1e-3}
            ], lr=lr_G, betas=(0.5, 0.999))
    # learning rate decay
    decay_D, decay_G = get_lr_linear_decay_rate(lr_D, lr_G, max_epoch, train_dataloader)
    for epoch in range(1, max_epoch+1):
        step = 1
        for batch in train_dataloader:
            # prepare data
            real_images_A, real_images_B, real_mask = batch
            real_images_A, real_images_B, real_mask = real_images_A.to(device), real_images_B.to(device), real_mask.to(device)
            
            # training iteration
            synthetic_images_B, attn_syn_images_B = model.G_A2B(real_images_A, real_mask)
            synthetic_images_A, attn_syn_images_A = model.G_B2A(real_images_B, real_mask)

            # ======= Discriminator training ==========
            # caculate discriminator loss
            # discriminator should predicts all patches of real images as real (1)
            guess_A = model.D_A(real_images_A)
            ones = torch.ones_like(guess_A) * 0.95 # Use 0.95 to avoid training the discriminators to zero loss
            DA_loss_real = loss_D(guess_A, ones)
            guess_B = model.D_B(real_images_B)
            DB_loss_real = loss_D(guess_B, ones)

            # discriminator should predicts all patches of synthetic images as fake (0)
            guess_A = model.D_A(synthetic_images_A)
            zeros = torch.zeros_like(guess_A)
            DA_loss_synthetic = loss_D(guess_A, zeros)
            guess_B = model.D_B(synthetic_images_B)
            DB_loss_synthetic = loss_D(guess_B, zeros)

            DA_loss = DA_loss_real + DA_loss_synthetic
            DB_loss = DB_loss_real + DB_loss_synthetic
            # update discriminator
            D_loss = loss_weights_D * (DA_loss + DB_loss)
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B, real_images_B*real_mask, ones, ones]  # Compare reconstructed images to real images
            input_data = model.generator(real_images_A, real_images_B, real_mask)
            G_losses = []
            for input, target, loss, loss_weight in zip(input_data, target_data, loss_G, loss_weights_G):
                G_losses.append(loss_weight * loss(input, target))
            reconstruction_loss_A = G_losses[0]
            reconstruction_loss_B = G_losses[1]
            gA_d_loss_synthetic = G_losses[3]
            gB_d_loss_synthetic = G_losses[4]

            # Identity training
            if step % identity_mapping_modulus == 0:
                identity_images_B, attn_idnt_images_B = model.G_A2B(real_images_B, real_mask)
                G_A2B_identity_loss = loss_I(identity_images_B, real_images_B) + loss_I(attn_idnt_images_B, real_images_B*real_mask)
                identity_images_A, attn_idnt_images_A = model.G_B2A(real_images_A, real_mask)
                G_B2A_identity_loss = loss_I(identity_images_A, real_images_A) + loss_I(attn_idnt_images_A, real_images_A*real_mask)
                G_identity_loss = loss_weights_I * (G_A2B_identity_loss + G_B2A_identity_loss)
                G_losses.append(G_identity_loss)
            # update generator
            G_loss = sum(G_losses)
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            # Store training data
            DA_losses.append(DA_loss.item())
            DB_losses.append(DB_loss.item())
            gA_d_losses_synthetic.append(gA_d_loss_synthetic.item())
            gB_d_losses_synthetic.append(gB_d_loss_synthetic.item())
            gA_losses_reconstructed.append(reconstruction_loss_A.item())
            gB_losses_reconstructed.append(reconstruction_loss_B.item())

            GA_loss = gA_d_loss_synthetic.item() + reconstruction_loss_A.item()
            GB_loss = gB_d_loss_synthetic.item() + reconstruction_loss_B.item()
            D_losses.append(D_loss.item())
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss.item())
            reconstruction_loss = reconstruction_loss_A.item() + reconstruction_loss_B.item()
            reconstruction_losses.append(reconstruction_loss)

            logger.add_scalars('Discriminator', {'A': DA_loss.item()}, global_step=step)
            logger.add_scalars('Discriminator', {'B': DB_loss.item()}, global_step=step)
            logger.add_scalars('Discriminator', {'total': D_loss.item()}, global_step=step)

            logger.add_scalars('Generator', {'A': GA_loss}, global_step=step)
            logger.add_scalars('Generator', {'B': GB_loss}, global_step=step)
            logger.add_scalars('Generator', {'total': G_loss.item()}, global_step=step)

            logger.add_scalars('Reconstruction', {'A': reconstruction_loss_A.item()}, global_step=step)
            logger.add_scalars('Reconstruction', {'B': reconstruction_loss_B.item()}, global_step=step)
            logger.add_scalars('Reconstruction', {'total': reconstruction_loss}, global_step=step)
            step += 1
            if verbose and step % print_every_step == 0:
                print('\n')
                print('Epoch----------------', epoch, '/', max_epoch)
                print('Loop index----------------', step, '/', len(train_dataloader))
                print(f'D_loss: {D_loss.item():.4f}')
                print(f'G_loss: {G_loss.item():.4f}')
                print(f'reconstruction_loss: {reconstruction_loss:.4f}')
        
        if epoch > DECAY_EPOCH:
            update_lr(optimizer_D, decay_D)
            update_lr(optimizer_G, decay_G)

        if epoch % 10 == 0:
            save_checkpoint(model.D_A, epoch, 'D_A.pth')
            save_checkpoint(model.D_B, epoch, 'D_B.pth')
            save_checkpoint(model.G_A2B, epoch, 'G_A2B.pth')
            save_checkpoint(model.G_B2A, epoch, 'G_B2A.pth')
    
        training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
        writeLossDataToFile(training_history)
    # close logger
    logger.close()

def get_lr_linear_decay_rate(lr_D, lr_G, max_epoch, data_loader):
    max_nr_images = len(data_loader.dataset)
    updates_per_epoch_D = 2 * max_nr_images
    updates_per_epoch_G = max_nr_images
    updates_per_epoch_G *= (1 + 1 / 10)
    denominator_D = (max_epoch - DECAY_EPOCH) * updates_per_epoch_D
    denominator_G = (max_epoch - DECAY_EPOCH) * updates_per_epoch_G
    
    decay_D = lr_D / denominator_D
    decay_G = lr_G / denominator_G

    return decay_D, decay_G

def update_lr(optimizer, decay):
    for g in optimizer.param_groups:
                g['lr'] -= decay

def save_checkpoint(model, epoch, name):
    path = './checkpoints/'
    if not os.path.isdir(path):
        os.mkdir(path)

    filename = os.path.join(path, f'epoch-{epoch}_'+name)
    torch.save(model.state_dict(), filename)

def writeLossDataToFile(history):
    if not os.path.isdir('./logs/'):
        os.mkdir('./logs/')
    
    keys = sorted(history.keys())
    filename = f'./logs/{DATE_TIME}/loss_output.csv'
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(keys)
        writer.writerows(zip(*[history[key] for key in keys]))