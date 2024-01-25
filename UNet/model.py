#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:51:02 2024

@author: anna.yang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import torch.optim as optim

"""
Input:
    x - Tensor(n, c, h, w)
    
Parameters:
    in_channels
    out_channels
"""

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, block_out_channels):  # [64, 128, 256, 512]
        super().__init__()
        
        self.out_channels = block_out_channels
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        for out_channels in block_out_channels:
            self.encoder.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels
        
    def forward(self, x):
        out_skip_connections = []
        for module in self.encoder:
            x = module(x)
            out_skip_connections.append(x)
            x = self.pool(x)
        
        return x, out_skip_connections


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, block_out_channels): # [512, 256, 128, 64]
        super().__init__()
        
        self.out_channels = block_out_channels
        self.decoder = nn.ModuleList()
        
        for out_channels in block_out_channels:
            self.decoder.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(out_channels*2, out_channels))
            in_channels = out_channels
            
            
    def forward(self, x, in_skip_connections):
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x) #up-conv
            skip_connection = in_skip_connections[idx//2]
            
            #resize skip_connection
            skip_connection = tf.resize(skip_connection, size=x.shape[2:], antialias=True)
            # concat on channels
            concat = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat) # 2 conv

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, block_out_channels):
        super().__init__()
        
        self.encoder = UNetEncoder(in_channels, block_out_channels)
        self.bottle_neck = DoubleConv(block_out_channels[-1], block_out_channels[-1]*2)
        self.decoder = UNetDecoder(block_out_channels[-1]*2, block_out_channels[::-1])
        self.final_layer = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        out_encoder, out_skip_connections = self.encoder(x)
        x = self.bottle_neck(out_encoder)
        x = self.decoder(x, out_skip_connections[::-1])
        
        x = self.final_layer(x)
        
        return x


x = torch.rand((2, 3, 572, 572))
model = UNet(3, 3, block_out_channels)
out = model(x)