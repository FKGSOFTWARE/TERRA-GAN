# src/utils/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get sorted list of filenames to match images and masks
        self.img_filenames = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

        assert len(self.img_filenames) == len(self.mask_filenames), "Number of images and masks do not match."

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # Load grayscale image
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        img = Image.open(img_path).convert('L')

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = Image.open(mask_path).convert('L')

        # Apply transformations
        if self.transform:
            # Ensure both are resized to the same dimensions
            img = self.transform(img)
            mask = self.transform(mask)
            mask = (mask > 0).float()  # Binarize the mask

        # Ensure mask has shape (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return {'image': img, 'mask': mask}
