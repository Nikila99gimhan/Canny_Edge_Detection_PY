import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

# Gaussian Kernel Generation
def gaussian(x, y, sigma=1.0):
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))

def gaussian_kernel(size, sigma=1.0):
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = gaussian(x, y, sigma)
    kernel /= kernel.sum()
    return kernel

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

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = img[i+1, j]
                r = img[i-1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if img[i, j] >= q and img[i, j] >= r:
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0

    return Z

# Double Thresholding
def double_threshold(img, low_threshold, high_threshold):
    high_threshold_value = img.max() * high_threshold
    low_threshold_value = high_threshold_value * low_threshold

    M, N = img.shape
    result = np.zeros((M, N), dtype=np.uint8)

    strong_edge_i, strong_edge_j = np.where(img >= high_threshold_value)
    weak_edge_i, weak_edge_j = np.where((img >= low_threshold_value) & (img < high_threshold_value))

    result[strong_edge_i, strong_edge_j] = 255
    result[weak_edge_i, weak_edge_j] = 100  # You can choose any value for weak edges, e.g., 100

    return result

# Edge Tracking by Hysteresis
def edge_tracking_hysteresis(img, weak_value):
    M, N = img.shape
    visited = np.zeros((M, N), dtype=bool)

    def dfs(i, j):
        visited[i, j] = True
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if 0 <= x < M and 0 <= y < N and not visited[x, y] and img[x, y] == weak_value:
                    dfs(x, y)

    strong_i, strong_j = np.where(img == 255)
    for i, j in zip(strong_i, strong_j):
        if not visited[i, j]:
            dfs(i, j)

    result = np.zeros((M, N), dtype=np.uint8)
    result[visited] = 255

    return result

# Main Program
image_path = 'img001.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

kernel = gaussian_kernel(5, 1.0)
gaussian_blurred = ndimage.convolve(img, kernel)
gradient_magnitude, gradient_direction = sobel_filters(gaussian_blurred)
nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)


# Double Thresholding Parameters
low_threshold = 0.1  # Adjust as needed
high_threshold = 0.3  # Adjust as needed

# Apply Double Thresholding
double_thresholded = double_threshold(nms_result, low_threshold, high_threshold)

# Apply Edge Tracking by Hysteresis
tracked_edges = edge_tracking_hysteresis(double_thresholded, 100)  # Use the same weak edge value as before


plt.figure(figsize=(16, 6))
plt.subplot(1, 6, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 6, 2)
plt.imshow(gaussian_blurred, cmap='gray')
plt.title('Gaussian Blurred')
plt.subplot(1, 6, 3)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude (Sobel)')
plt.subplot(1, 6, 4)
plt.imshow(nms_result, cmap='gray')
plt.title('Non-Maximum Suppression')
plt.subplot(1, 6, 5)
plt.imshow(double_thresholded, cmap='gray')
plt.title('Double Thresholding')
plt.subplot(1, 6, 6)
plt.imshow(tracked_edges, cmap='gray')
plt.title('Edge Tracking by Hysteresis')
plt.tight_layout()
plt.show()
