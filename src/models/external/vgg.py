import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            ConvBNReLU(3, 64),
            nn.Dropout(p=0.3),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            
            ConvBNReLU(64, 128),
            nn.Dropout(p=0.4),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            
            ConvBNReLU(128, 256),
            nn.Dropout(p=0.4),
            ConvBNReLU(256, 256),
            nn.Dropout(p=0.4),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            
            ConvBNReLU(256, 512),
            nn.Dropout(p=0.4),
            ConvBNReLU(512, 512),
            nn.Dropout(p=0.4),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            
            ConvBNReLU(512, 512),
            nn.Dropout(p=0.4),
            ConvBNReLU(512, 512),
            nn.Dropout(p=0.4),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        return self.model(x)


def vgg(num_classes: int):
    return VGG(num_classes=num_classes)