import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def convleakyrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, normalization=True):
    if not normalization:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), 
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, normalization=True):
    if not normalization:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), 
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

def deconvrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, normalization=True):
    if not normalization:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups, bias=bias, dilation=dilation),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups, bias=False, dilation=dilation),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

class CycleGAN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.in_channels = in_channels
        # discriminator 
        self.D_A = Discriminator(in_channels)
        self.D_B = Discriminator(in_channels)
        # generator
        self.G_A2B = Generator(in_channels)
        self.G_B2A = Generator(in_channels)
        # weight initilization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
    def generator(self, real_A, real_B, mask):
        synthetic_B, attn_synthetic_B = self.G_A2B(real_A, mask)
        synthetic_A, attn_synthetic_A = self.G_B2A(real_B, mask)
        # Do not update discriminator weights during generator training
        with torch.no_grad():
            dA_guess_synthetic = self.D_A(synthetic_A)
            dB_guess_synthetic = self.D_B(synthetic_B)
        reconstructed_B, attn_rec_B = self.G_A2B(synthetic_A, mask)
        reconstructed_A, attn_rec_A = self.G_B2A(synthetic_B, mask)
        return reconstructed_A, reconstructed_B, attn_rec_B, dA_guess_synthetic, dB_guess_synthetic
    
class Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(48, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        # layer 2
        self.layer2 = convrelu(48, 72)
        # layer 3
        self.layer3 = convrelu(72, 128)
        # lyaer 4-12: residual layer
        layers = []
        for _ in range(4, 13):
            layers.append(ResidualBlock(128))
        self.residual_blocks = nn.Sequential(*layers)
        # layer 13
        self.layer13 = deconvrelu(128, 72)
        # layer 14
        self.layer14 = nn.Sequential(
            deconvrelu(72, 48),
            nn.Conv2d(48, in_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, img, mask):
        out = self.layer3(self.layer2(self.layer1(img)))
        out = self.residual_blocks(out)
        out = self.layer14(self.layer13(out))
        out_attn = out * mask
        return out, out_attn

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # layer 1 (instance normalization is not used for this layer)
        self.layer1 = convleakyrelu(in_channels, 64, kernel_size=4, stride=2, padding=1, normalization=False)
        # layer 2
        self.layer2 = convleakyrelu(64, 128, kernel_size=4, stride=2, padding=1)
        # layer 3
        self.layer3 = convleakyrelu(128, 256, kernel_size=4, stride=2, padding=1)
        # layer 4
        self.layer4 = convleakyrelu(256, 512, kernel_size=4, stride=2, padding=1)
        # output layer
        self.out = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, padding=2),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.out(self.layer4(self.layer3(self.layer2(self.layer1(img)))))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu1(self.norm1(self.conv1(x)))
        y = self.relu2(self.norm2(self.conv2(y)))
        return x + y