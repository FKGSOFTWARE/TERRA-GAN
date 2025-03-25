# src/evaluate.py

import torch
from torchvision import transforms
from PIL import Image
from .models.generator import PConvUNet

def evaluate(image_path, mask_path, model_or_checkpoint_path, save_path):
    """
    Evaluate a model on a single image.

    Args:
        image_path: Path to the input image
        mask_path: Path to the mask
        model_or_checkpoint_path: Either a PConvUNet model instance or path to a checkpoint
        save_path: Path to save the inpainted image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Adjust image size to match training
    img_size = (512, 512)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    image = transform(image).unsqueeze(0).to(device)
    mask = transform(mask).unsqueeze(0).to(device)
    mask = (mask > 0).float()  # Binarize the mask
    masked_img = image * mask

    # Handle either model instance or checkpoint path
    if isinstance(model_or_checkpoint_path, PConvUNet):
        generator = model_or_checkpoint_path
    else:
        generator = PConvUNet().to(device)
        # Update checkpoint loading with weights_only=True
        checkpoint = torch.load(model_or_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)

    generator.eval()

    with torch.no_grad():
        output = generator(masked_img, mask)

    # Convert tensors to images for saving
    output_img = output.cpu().squeeze().numpy()
    output_img = (output_img * 255).astype('uint8')
    output_pil = Image.fromarray(output_img, mode='L')

    # Resize to 500x500 if needed
    output_pil = output_pil.resize((500, 500), Image.BILINEAR)
    output_pil.save(save_path)
    print(f"Inpainted image saved to {save_path}")
