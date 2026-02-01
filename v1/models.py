"""Model definitions for v1"""
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder - zmniejszony dropout i dodany BatchNorm dla lepszej stabilności
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),  # Dropout tylko w encoderze
            nn.MaxPool2d(2, 2),  # 256x256 -> 128x128
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Brak dropout na końcu encoder - zachowujemy informację
        )
        
        # Decoder - bez dropout (zachowuje szczegóły) i używa ConvTranspose2d zamiast Upsample
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 64x64 -> 128x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Brak dropout w decoderze - zachowujemy szczegóły
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 128x128 -> 256x256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Brak dropout w decoderze
            
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # Normalizacja do [0,1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
