import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_grid_dimensions(width, height, max_patches=9):
    """
    Calculates the optimal grid dimensions (p_w, p_h) for dynamic resolution.

    Args:
        width (int): Original image width.
        height (int): Original image height.
        max_patches (int): Maximum number of patches allowed (H).

    Returns:
        tuple: (p_w, p_h) best grid dimensions.
    """
    best_pw, best_ph = 1, 1

    # Iterate over possible values of p_w
    # Since p_h >= 1, p_w cannot exceed max_patches
    for pw in range(1, max_patches + 1):
        # Constraint: p_h = ceil(p_w * h / w)
        ph = math.ceil(pw * height / width)

        # Constraint: p_w * p_h <= H
        if pw * ph <= max_patches:
            # We assume we want to maximize resolution, so we update best found so far
            # Since p_w is increasing and p_h is roughly monotonic increasing,
            # the last valid configuration is usually the one with max patches.
            best_pw, best_ph = pw, ph
        else:
            # Once we exceed max_patches, larger p_w will only lead to larger p_h,
            # so we can stop.
            break

    return best_pw, best_ph

def visualize_patches(width, height, pw, ph, patch_size=336):
    """
    Creates a visualization of the image patch processing.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aspect ratio of the original image
    aspect = width / height

    # Draw original image placeholder
    # Normalize to fit in plot
    # Use 0-1 coords for simplicity

    # Instead of purely abstract, let's represent the grid
    # Resized image dimensions
    resized_w = pw * patch_size
    resized_h = ph * patch_size

    # We plot the grid structure
    # Let's say we plot the rectangle of size (p_w, p_h) but labeled clearly

    ax.set_aspect('equal')
    ax.set_xlim(0, pw)
    ax.set_ylim(0, ph)

    # Draw grid
    for i in range(pw + 1):
        ax.axvline(i, color='black', linestyle='-', linewidth=1)
    for i in range(ph + 1):
        ax.axhline(i, color='black', linestyle='-', linewidth=1)

    # Add text
    ax.text(pw/2, ph/2, f"Target Grid: {pw} x {ph}\nOriginal: {width}x{height}",
            horizontalalignment='center', verticalalignment='center', fontsize=12)

    ax.set_title(f"Dynamic Partition: p_w={pw}, p_h={ph} (Total: {pw*ph})")
    ax.invert_yaxis() # Image system coordinates

    plt.tight_layout()
    plt.savefig('patch_visualization.png')
    print("Saved visualization to patch_visualization.png")

if __name__ == "__main__":
    # Example usage
    img_h, img_w = 1000, 1000 # Square
    H = 9
    pw, ph = calculate_grid_dimensions(img_w, img_h, H)
    print(f"Image {img_w}x{img_h}, H={H} -> ({pw}, {ph})")

    img_h, img_w = 600, 1200 # Wide
    pw, ph = calculate_grid_dimensions(img_w, img_h, H)
    print(f"Image {img_w}x{img_h}, H={H} -> ({pw}, {ph})")

    img_h, img_w = 1200, 600 # Tall
    pw, ph = calculate_grid_dimensions(img_w, img_h, H)
    print(f"Image {img_w}x{img_h}, H={H} -> ({pw}, {ph})")
