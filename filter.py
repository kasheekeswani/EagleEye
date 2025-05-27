import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to create the Gaussian spatial weight (G)
def gaussian_spatial_weight(kernel_size, sigma_s):
    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    G = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))
    return G

# Function to create the range weight (S) based on intensity difference
def range_weight(I, x, y, i, j, sigma_r):
    intensity_diff = I[x + i, y + j] - I[x, y]
    return np.exp(-(intensity_diff**2) / (2 * sigma_r**2))

# EagleEye Filter function
def eagle_eye_filter(I, sigma_s=1.5, sigma_r=10, kernel_size=5):
    k = kernel_size // 2
    filtered_image = np.zeros_like(I, dtype=np.float32)

    # Precompute Gaussian spatial weights (constant)
    G = gaussian_spatial_weight(kernel_size, sigma_s)

    # Iterate over image excluding border
    for x in range(k, I.shape[0] - k):
        for y in range(k, I.shape[1] - k):
            sum_filter = 0.0
            normalization_factor = 0.0

            # Calculate range weights and combine with spatial weights
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    S_value = range_weight(I, x, y, i, j, sigma_r)
                    weight = G[i + k, j + k] * S_value

                    sum_filter += weight * I[x + i, y + j]
                    normalization_factor += weight

            filtered_image[x, y] = sum_filter / normalization_factor if normalization_factor != 0 else I[x, y]

    return filtered_image

def main():
    # Load image (grayscale) and check
    image_path = 'image.png'  # Change filename if needed
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from path '{image_path}'. Please check the file exists.")
        return

    # Convert to float32 for safe math operations
    image = image.astype(np.float32)

    # Apply EagleEye filter
    filtered_image = eagle_eye_filter(image, sigma_s=1.5, sigma_r=10, kernel_size=5)

    # Clip and convert filtered image for display
    filtered_display = np.clip(filtered_image, 0, 255).astype(np.uint8)

    # Display side-by-side using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image.astype(np.uint8), cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(filtered_display, cmap='gray')
    axs[1].set_title('Filtered Image (EagleEye)')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
