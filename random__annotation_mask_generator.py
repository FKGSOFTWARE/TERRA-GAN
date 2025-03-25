import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import time
import json
import argparse

def line(x0, y0, x1, y1):
    """Draw line between two points."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    rr, cc = [], []
    while True:
        rr.append(x0)
        cc.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return np.array(rr), np.array(cc)

def generate_dem_random_mask(size=500, approach=None):
    """Generate a single random mask that mimics plausible DEM artifacts.

    Args:
        size: Size of mask (square dimensions)
        approach: One of "edge", "patch", "region", or None for random selection

    Returns:
        Binary mask array with False (0) as background and True (1) as features
    """
    # Initialize empty mask (False = black background)
    mask = np.zeros((size, size), dtype=bool)

    # Choose random approach for this mask if not specified
    if approach is None:
        approach = np.random.choice(["edge", "patch", "region"])

    if approach == "edge":
        # Simulate elevation edges/ridges
        base = np.zeros((size, size))

        # Add random gradient features
        num_features = np.random.randint(3, 10)
        for _ in range(num_features):
            # Create random line segments
            points = np.random.randint(0, size, (np.random.randint(3, 8), 2))

            # Connect points with lines
            for j in range(len(points)-1):
                rr, cc = line(points[j][0], points[j][1],
                             points[j+1][0], points[j+1][1])
                # Keep only valid coordinates
                valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
                rr, cc = rr[valid], cc[valid]
                if len(rr) > 0:  # Make sure we have valid points
                    base[rr, cc] = 1

        # Dilate and blur
        base = ndimage.binary_dilation(base, iterations=np.random.randint(2, 5))
        base = ndimage.gaussian_filter(base.astype(float), sigma=np.random.uniform(1, 3))

        # Threshold at random value
        threshold = np.random.uniform(0.4, 0.7)
        mask = base > threshold

    elif approach == "patch":
        # Create patches of "implausible" terrain
        num_patches = np.random.randint(3, 12)

        for _ in range(num_patches):
            cx, cy = np.random.randint(0, size, 2)
            radius = np.random.randint(10, 50)

            # Create a patch
            y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
            dist = np.sqrt(x*x + y*y)

            # Add some noise to the patch boundary
            noise = ndimage.gaussian_filter(
                np.random.normal(0, 1, (size, size)),
                sigma=np.random.uniform(3, 8)
            )

            noisy_patch = (dist <= radius + noise * np.random.uniform(5, 15))
            mask = mask | noisy_patch

    elif approach == "region":
        # Create coherent regions
        num_regions = np.random.randint(1, 4)

        for _ in range(num_regions):
            # Create base region
            cx, cy = np.random.randint(0, size, 2)

            # Random shape parameters
            min_size = np.random.randint(30, 60)
            max_size = np.random.randint(60, 120)

            y, x = np.ogrid[-cy:size-cy, -cx:size-cx]

            # Create elliptical or irregular base
            if np.random.random() > 0.5:
                # Elliptical
                a = np.random.randint(min_size, max_size)
                b = np.random.randint(min_size, max_size)
                region = (x*x)/(a*a) + (y*y)/(b*b) <= 1
            else:
                # Irregular shape using noise
                noise = np.zeros((size, size))

                for i in range(size):
                    for j in range(size):
                        noise[i,j] = np.random.random()

                noise = ndimage.gaussian_filter(noise, sigma=np.random.uniform(10, 30))
                region = ((x*x + y*y) <= max_size**2) & (noise > np.random.uniform(0.4, 0.6))

            mask = mask | region

    # Apply morphological operations for realism
    if np.random.random() > 0.3:
        mask = ndimage.binary_opening(mask, iterations=np.random.randint(1, 2))
    if np.random.random() > 0.3:
        mask = ndimage.binary_closing(mask, iterations=np.random.randint(1, 2))

    # Ensure reasonable density
    density = mask.mean()
    if density < 0.01:
        mask = ndimage.binary_dilation(mask, iterations=np.random.randint(1, 2))
    elif density > 0.3:
        mask = ndimage.binary_erosion(mask, iterations=np.random.randint(1, 3))

    # INVERT THE MASK - this is now the default behavior
    mask = ~mask

    return mask

def generate_and_save_masks(filenames, size=500, output_dir="random_masks", approach=None):
    """Generate random inverted masks with specific filenames and save them to the specified directory.

    Args:
        filenames: List of specific filenames to use
        size: Size of masks (square dimensions)
        output_dir: Directory to save the masks
        approach: The specific approach to use for all masks (edge, patch, region, or None for mixed)

    Returns:
        Dictionary of filenames and their corresponding densities
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    mask_stats = {}
    n = len(filenames)

    if approach is not None:
        # Use the same approach for all masks
        approaches = [approach] * n
        print(f"Using '{approach}' approach for all masks")
    else:
        # Generate masks using all three approaches in roughly equal proportions
        approaches = ["edge"] * (n // 3) + ["patch"] * (n // 3) + ["region"] * (n // 3)
        # Add random approaches for any remaining masks
        approaches += [np.random.choice(["edge", "patch", "region"]) for _ in range(n - len(approaches))]
        np.random.shuffle(approaches)
        print("Using mixed approaches (edge, patch, region)")

    start_time = time.time()

    for i, filename in enumerate(filenames):
        # Print progress update every 10 masks
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            time_per_mask = elapsed / (i + 1)
            remaining = time_per_mask * (n - i - 1)
            print(f"Generated {i+1}/{n} masks. Estimated time remaining: {remaining:.1f} seconds")

        # Generate mask with the current approach
        current_approach = approaches[i]
        mask = generate_dem_random_mask(size=size, approach=current_approach)
        # Note: The mask is already inverted in the generate_dem_random_mask function

        # Calculate density
        density = mask.mean() * 100  # as percentage

        # Save as PNG (white foreground on black background)
        output_path = os.path.join(output_dir, filename)
        plt.imsave(output_path, mask, cmap='binary')

        # Store statistics
        mask_stats[filename] = {
            "density": density,
            "approach": current_approach,
            "white_pixels": int(mask.sum()),
            "total_pixels": size * size,
            "type": "inverted"
        }

    # Calculate overall statistics
    densities = [stats["density"] for stats in mask_stats.values()]

    summary_stats = {
        "mean_density": np.mean(densities),
        "median_density": np.median(densities),
        "min_density": np.min(densities),
        "max_density": np.max(densities),
        "std_density": np.std(densities),
        "num_masks": len(filenames),
        "mask_size": f"{size}x{size}",
        "approach_counts": {
            "edge": approaches.count("edge"),
            "patch": approaches.count("patch"),
            "region": approaches.count("region")
        },
        "specific_approach": approach,  # Will be None if mixed
        "inverted": True
    }

    # Save statistics to JSON file
    stats_filename = f"mask_statistics{'_' + approach if approach else ''}_inverted.json"
    with open(os.path.join(output_dir, stats_filename), "w") as f:
        json.dump({
            "summary": summary_stats,
            "masks": mask_stats
        }, f, indent=2)

    # Print statistics about the generated masks
    print(f"\nGenerated {n} inverted masks with dimensions {size}x{size}")
    print(f"Density statistics (percentage of white pixels):")
    print(f"  Mean: {summary_stats['mean_density']:.2f}%")
    print(f"  Min: {summary_stats['min_density']:.2f}%")
    print(f"  Max: {summary_stats['max_density']:.2f}%")
    print(f"  Median: {summary_stats['median_density']:.2f}%")
    print(f"Masks saved to: {output_dir}")
    print(f"Statistics saved to: {os.path.join(output_dir, stats_filename)}")

    return mask_stats

def display_sample_masks(output_dir, num_samples=6, approach=None):
    """Display a few sample masks from the generated set."""
    # Get list of all mask files
    mask_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and not f.startswith('sample_')]

    # If we have fewer files than requested samples, show all files
    num_samples = min(len(mask_files), num_samples)

    # Randomly select a subset to display
    selected_files = np.random.choice(mask_files, size=num_samples, replace=False)

    # Create a grid of sample images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, filename in enumerate(selected_files):
        img = plt.imread(os.path.join(output_dir, filename))
        axes[i].imshow(img, cmap='binary')
        axes[i].set_title(os.path.basename(filename))
        axes[i].axis('off')

    approach_text = f" ({approach} approach)" if approach else " (mixed approaches)"
    plt.suptitle(f"Sample Inverted DEM Masks{approach_text}", fontsize=16)
    plt.tight_layout()

    # Save with approach in filename if specified
    sample_filename = f"sample_masks_inverted{'_' + approach if approach else ''}.png"
    plt.savefig(os.path.join(output_dir, sample_filename))
    plt.close()
    print(f"Sample visualization saved to {output_dir}/{sample_filename}")

def get_filenames_from_pattern():
    """Generate filenames based on the NS83 pattern."""
    filenames = []
    for x in range(0, 10):
        for z in range(0, 10):
            filenames.append(f"NS83_ns8{x}3{z}_inpainted_colored_Zmlu_mask.png")
    return filenames

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate random inverted DEM masks for evaluation.')
    parser.add_argument('--size', type=int, default=500,
                        help='Size of masks (square dimensions, default: 500)')
    parser.add_argument('--output-dir', type=str, default="random_dem_masks_inverted",
                        help='Directory to save the masks (default: random_dem_masks_inverted)')
    parser.add_argument('--approach', type=str, choices=['edge', 'patch', 'region', 'mixed'],
                        default='mixed',
                        help='Approach to use for generating masks (default: mixed)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of sample masks to display (default: 6)')
    parser.add_argument('--subset', type=int, default=None,
                        help='Generate only a subset of masks (default: all)')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Convert 'mixed' to None for internal processing
    approach = None if args.approach == 'mixed' else args.approach

    # Get filenames
    all_filenames = get_filenames_from_pattern()

    # Use subset if specified
    if args.subset is not None:
        filenames = all_filenames[:args.subset]
        print(f"Using subset of {args.subset} filenames")
    else:
        filenames = all_filenames

    # Create output directory with approach name
    output_dir = args.output_dir
    if approach:
        output_dir = f"{output_dir}_{approach}"

    # Generate the inverted masks with the specified approach
    mask_stats = generate_and_save_masks(
        filenames=filenames,
        size=args.size,
        output_dir=output_dir,
        approach=approach
    )

    # Display sample masks
    display_sample_masks(output_dir, args.samples, approach)

    print("\nDone!")
