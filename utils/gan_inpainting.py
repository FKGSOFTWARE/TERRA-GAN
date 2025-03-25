import logging
from pathlib import Path
from mvp_gan.src.evaluate import evaluate  # Import the GAN evaluation function

def inpaint_with_gan(dem_image_path, mask_path, output_dir, checkpoint_path):
    """
    Inpaint using GAN and save the output.

    Parameters:
    - dem_image_path: Path to the DEM image.
    - mask_path: Path to the binary mask.
    - output_dir: Directory where the inpainted image will be saved.
    - checkpoint_path: Path to the GAN model checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    inpainted_image_path = output_dir / f"{dem_image_path.stem}_inpainted.png"
    evaluate(dem_image_path, mask_path, checkpoint_path, inpainted_image_path)
    logging.info(f"Inpainted image saved to {inpainted_image_path}")
    return inpainted_image_path
