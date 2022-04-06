import torch, torch.nn as nn
import torch.nn.functional as F
from .base import Model
import torchvision.models as models
class TinyConv(Model):
    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(1, 4),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(1, 4),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(1, 4),
            #nn.MaxPool2d((2, 2), stride=(2, 1)),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(1, 4))
            #nn.MaxPool2d((2, 2), stride=(2, 1)))
        self.fc1 = nn.Linear(64 , out_features=256)
        self.fc2 = nn.Linear(256, out_features=64)
        self.fc3 = nn.Linear(64, out_features=11)
        self.dropout=nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.dropout(F.selu(self.fc1(x)))
        x = self.dropout(F.selu(self.fc2(x)))
        x =self.fc3(x)
        return x