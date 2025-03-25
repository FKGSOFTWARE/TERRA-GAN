# src/models/pconv.py

import torch
import torch.nn as nn

class PConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True):
        super(PConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.slide_winsize = self.input_conv.weight.data.shape[2] * self.input_conv.weight.data.shape[3]
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)

        # Initialize mask_conv weights to 1
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # Optional batch normalization
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

    def forward(self, input, mask):
        # Multiply input by mask (broadcasting mask if necessary)
        input_masked = input * mask  # mask shape: [B,1,H,W]; input shape: [B,C,H,W]

        # Apply convolution to masked input
        output = self.input_conv(input_masked)

        # Update mask
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
            output_mask = (output_mask > 0).float()

        # Calculate mask ratio
        mask_sum = self.mask_conv(mask)
        mask_ratio = self.slide_winsize / (mask_sum + 1e-8)
        mask_ratio = mask_ratio * (mask_sum > 0).float()

        # Normalize output
        output = output * mask_ratio

        # Apply batch normalization and activation
        if self.batch_norm:
            output = self.bn(output)
        output = self.activation(output)

        return output, output_mask
