import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 512 * 4 * 4)
        self.trans_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = F.leaky_relu(self.bn1(self.trans_conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.trans_conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.trans_conv3(x)), 0.2)
        x = torch.tanh(self.trans_conv4(x))
        return x
