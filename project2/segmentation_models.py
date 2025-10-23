"""
Simple U-Net architecture for box segmentation.
Includes MLP baseline, CNN, and U-Net for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPSegmentation(nn.Module):
    """
    Simple MLP model for segmentation.
    Flattens input, applies multiple linear layers, reshapes to output mask.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(MLPSegmentation, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 input features
        input_size = in_channels*28*28
        # Output should be 1x28x28 = 784 features
        self.out_channels = out_channels
        output_size = out_channels*28*28
        
        # TODO: Add your own MLP architecture here
        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(input_size, 2000)
        self.layer2 = nn.Linear(2000, 1500)
        self.layer3 = nn.Linear(1500, 1000)
        self.layer4 = nn.Linear(1000, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flat(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x.reshape(-1, self.out_channels, 28, 28)
    

class SimpleCNN(nn.Module):
    """
    SimpleCNN for segmentation of 28x28 images.
    Optimized for small images and simple segmentation tasks.
    """
    def block(self, ch, kernel_size=3, pool_size=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size, padding=padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[2], kernel_size, padding=padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
        )
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super(SimpleCNN, self).__init__()
        
        # Encoder (contracting path)
        # TODO: Add your own encoder architecture here
        self.pool = nn.MaxPool2d(2)
        self.enc_1 = self.block([in_channels, 64, 64])
        self.enc_2 = self.block([64, 128, 128])
        self.bottleneck = self.block([128, 256, 256])
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_2 = self.block([128, 128, 128])
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_1 = self.block([64, 64, 64])
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        x = self.enc_1(x)
        x = self.enc_2(self.pool(x))
        x = self.bottleneck(self.pool(x))
        x = self.dec_2(self.up2(x))
        x = self.dec_1(self.up1(x))
        out = self.out(x)
        return self.sigmoid(out)

class TinyUNet(nn.Module):
    """
    Tiny U-Net for segmentation of 28x28 images.
    Optimized for small images and simple segmentation tasks.
    """
    def block(self, ch, kernel_size=3, pool_size=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size, padding=padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[2], kernel_size, padding=padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
        )
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super(TinyUNet, self).__init__()
        
        # Encoder (contracting path)
        # TODO: Add your own encoder architecture here
        self.pool = nn.MaxPool2d(2)
        self.enc_1 = self.block([in_channels, 64, 64])
        self.enc_2 = self.block([64, 128, 128])
        self.bottleneck = self.block([128, 256, 256])
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_2 = self.block([256, 128, 128])
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_1 = self.block([128, 64, 64])
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        x_1 = self.enc_1(x)
        x_2 = self.enc_2(self.pool(x_1))
        # x_3 = self.enc_3(self.pool(x_2))
        # x_b = self.bottleneck(x_3)
        x_b = self.bottleneck(self.pool(x_2))
        # xp_3 = self.dec_3(torch.cat([self.up3(x_b), x_3], dim=1))
        xp_2 = self.dec_2(torch.cat([self.up2(x_b), x_2], dim=1))
        xp_1 = self.dec_1(torch.cat([self.up1(xp_2), x_1], dim=1))
        out = self.out(xp_1)
        return self.sigmoid(out)
        

def get_segmentation_model(model_name, in_channels=3, out_channels=1):
    """Get segmentation model by name."""
    if model_name == 'mlp':
        return MLPSegmentation(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'unet':
        return TinyUNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'cnn':
        return SimpleCNN(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError("Unknown segmentation model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Intersection over Union (IoU) metric for segmentation
def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union for binary segmentation masks.
    
    Args:
        pred_mask: Predicted segmentation mask [B, 1, H, W] or [B, H, W]
        true_mask: Ground truth segmentation mask [B, 1, H, W] or [B, H, W]
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score (float)
    """
    # Convert to binary
    if torch.is_tensor(pred_mask):
        pred_binary = (pred_mask > threshold).float()
    else:
        pred_binary = (pred_mask > threshold).astype(float)
    
    if torch.is_tensor(true_mask):
        true_binary = (true_mask > 0.5).float()
    else:
        true_binary = (true_mask > 0.5).astype(float)
    
    # Flatten for easier computation
    if len(pred_binary.shape) > 2:
        pred_binary = pred_binary.view(pred_binary.size(0), -1)
        true_binary = true_binary.view(true_binary.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_binary * true_binary).sum(dim=-1)
    union = pred_binary.sum(dim=-1) + true_binary.sum(dim=-1) - intersection
    
    # Handle case where both masks are empty
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    return iou.mean().item() if torch.is_tensor(iou) else iou.mean()

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Better than BCE for imbalanced segmentation.
    """
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        overlap = (pred * target).sum(dim=(1, 2, 3))
        total = (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)))
        loss = 1 - (2*overlap + self.smooth)/(total + self.smooth)
        return loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for better segmentation performance.
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss() # TODO: Initialize the BCE loss
        self.dice_loss = DiceLoss() # TODO: Initialize the DICE loss
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
