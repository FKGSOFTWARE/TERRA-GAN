# # src/models/generator.py

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from .pconv import PConv2d

class PConvUNet(nn.Module):
    def __init__(self):
        super(PConvUNet, self).__init__()

        # Encoder layers
        self.enc1 = PConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.enc2 = PConv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc3 = PConv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.enc4 = PConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc5 = PConv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc6 = PConv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc7 = PConv2d(512, 512, kernel_size=3, stride=2, padding=1)

        # Decoder layers
        self.dec7 = PConv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec6 = PConv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec5 = PConv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec4 = PConv2d(512 + 256, 256, kernel_size=3, padding=1)
        self.dec3 = PConv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.dec2 = PConv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.dec1 = PConv2d(64, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, mask):
        # Encoder
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)
        e6, m6 = self.enc6(e5, m5)
        e7, m7 = self.enc7(e6, m6)

        # Decoder
        d6, dm6 = self.decode_step(e7, m7, e6, m6, self.dec7)
        d5, dm5 = self.decode_step(d6, dm6, e5, m5, self.dec6)
        d4, dm4 = self.decode_step(d5, dm5, e4, m4, self.dec5)
        d3, dm3 = self.decode_step(d4, dm4, e3, m3, self.dec4)
        d2, dm2 = self.decode_step(d3, dm3, e2, m2, self.dec3)
        d1, dm1 = self.decode_step(d2, dm2, e1, m1, self.dec2)

        # Final decoding without skip connection
        d0_up = interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        dm0_up = interpolate(dm1, scale_factor=2, mode='nearest')
        d0_up = self._pad_to_match(d0_up, x)
        dm0_up = self._pad_to_match(dm0_up, mask)
        m_combined = torch.max(dm0_up, mask)
        d0, _ = self.dec1(d0_up, m_combined)
        output = self.final(d0)
        output = torch.sigmoid(output)

        # Ensure that unmasked regions are copied from the input
        valid_mask = mask
        hole_mask = 1 - mask
        output = output * hole_mask + x * valid_mask

        return output

    def decode_step(self, up_feature, up_mask, skip_feature, skip_mask, decoder_layer):
        up_feature = interpolate(up_feature, scale_factor=2, mode='bilinear', align_corners=False)
        up_mask = interpolate(up_mask, scale_factor=2, mode='nearest')

        up_feature = self._pad_to_match(up_feature, skip_feature)
        up_mask = self._pad_to_match(up_mask, skip_mask)

        merged_feature = torch.cat([up_feature, skip_feature], dim=1)
        merged_mask = torch.max(up_mask, skip_mask)
        out_feature, out_mask = decoder_layer(merged_feature, merged_mask)
        return out_feature, out_mask

    def _pad_to_match(self, x, target):
        """Pads tensor x to match the size of target tensor along spatial dimensions."""
        diffY = target.size(2) - x.size(2)
        diffX = target.size(3) - x.size(3)
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        return x
