import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2  # Importing OpenCV for reading and writing images

def gaussian(x, y, sigma=2.0):
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))

def gaussian_kernel(size, sigma=2.0):
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = gaussian(x, y, sigma)
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

# Read an image
image_path = 'img001.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Reading the image in grayscale

# Generate a 5x5 Gaussian kernel
kernel = gaussian_kernel(5, 1.0)

# Apply Gaussian smoothing
gaussian_blurred = ndimage.convolve(img, kernel)

# Apply Sobel filters to the blurred image
gradient_magnitude, gradient_direction = sobel_filters(gaussian_blurred)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_blurred, cmap='gray')
plt.title('Gaussian Blurred')

plt.subplot(1, 3, 3)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Edge Detection (Sobel)')

plt.tight_layout()
plt.show()
