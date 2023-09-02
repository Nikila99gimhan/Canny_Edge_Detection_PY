import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2


# Gaussian Kernel Generation
def gaussian_kernel(size, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1.0 / (2 * np.pi * sigma ** 2)) *
                     np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / kernel.sum()


# Sobel Filter
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


# Non-Maximum Suppression
def non_maximum_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q, r = img[i, j + 1], img[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q, r = img[i + 1, j - 1], img[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q, r = img[i + 1, j], img[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q, r = img[i - 1, j - 1], img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z


# Double Thresholding
def double_threshold(img, low_ratio=0.05, high_ratio=0.09):
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


# Main Program
image_path = 'img001.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

kernel = gaussian_kernel(5, 1.0)
gaussian_blurred = ndimage.convolve(img, kernel)
gradient_magnitude, gradient_direction = sobel_filters(gaussian_blurred)
nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)
threshold_result = double_threshold(nms_result)

plt.figure(figsize=(20, 8))
plt.subplot(1, 5, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 5, 2), plt.imshow(gaussian_blurred, cmap='gray'), plt.title('Gaussian Blurred')
plt.subplot(1, 5, 3), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
plt.subplot(1, 5, 4), plt.imshow(nms_result, cmap='gray'), plt.title('Non-Maximum Suppressed')
plt.subplot(1, 5, 5), plt.imshow(threshold_result, cmap='gray'), plt.title('Double Thresholded')
plt.tight_layout()
plt.show()
