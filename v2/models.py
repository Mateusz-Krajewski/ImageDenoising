import torch
import torch.nn as nn

class UNetDenoiser(nn.Module):
    def __init__(self, dropout_rate=0.05):
        super(UNetDenoiser, self).__init__()
        
        # Encoder (downsampling) - zapisujemy feature maps dla skip connections
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 256x256 -> 128x128
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling) z skip connections
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 32x32 -> 64x64
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # 256 (skip) + 256 (up) = 512 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 64x64 -> 128x128
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128 (skip) + 128 (up) = 256 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 128x128 -> 256x256
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 (skip) + 64 (up) = 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.final = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder z zapisaniem feature maps dla skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder z skip connections (concatenate feature maps)
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)  # Skip connection
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.dec1(x)
        
        # Output
        x = self.final(x)
        x = self.sigmoid(x)
        
        return x
