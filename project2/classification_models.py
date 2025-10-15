"""
Simple CNN architectures for PathMNIST classification.
Includes MLP baseline and CNN variants for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinear(nn.Module):
    """
    Simple Linear Model: Logistic Regression
    """
    def __init__(self, num_classes=9):
        super(SimpleLinear, self).__init__()

        input_size = 3 * 28 * 28

        self.layer = nn.Linear(input_size, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.layer(x))


class MLPModel(nn.Module):
    """
    Simple MLP model: Flatten input then run through hidden layers.
    """
    
    def __init__(self, num_classes=9):
        super(MLPModel, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 features
        input_size = 3 * 28 * 28
        l1 = 1024
        l2 = 512
        l3 = 256
        
        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(input_size, l1)
        self.layer2 = nn.Linear(l1, l2)
        self.layer3 = nn.Linear(l2, l3)
        self.layer4 = nn.Linear(l3, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flat(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)

class CNNModel(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """

    def down_block(self, ch, kernel_size=3, pool_size=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size, padding=padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[2], kernel_size, padding=padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
        )
    
    def __init__(self, num_classes=9):
        super(CNNModel, self).__init__()
        
        self.cnn = nn.Sequential(
            self.down_block([3, 32, 64]), # 64, 14, 14
            nn.MaxPool2d(2),
            self.down_block([64, 128, 256]), # 256, 7, 7
            nn.MaxPool2d(2),
            self.down_block([256, 256, 512]), # 512, 3, 3
        )
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 512
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.final(self.cnn(x))

def get_model(model_name, num_classes=9):
    """Get model by name."""
    if model_name == 'mlp':
        return MLPModel(num_classes)
    elif model_name == 'cnn':
        return CNNModel(num_classes)
    else:
        #TODO: add your models names here
        raise ValueError("Unknown model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
