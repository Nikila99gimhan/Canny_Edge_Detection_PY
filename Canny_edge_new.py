import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the input image
image_path = 'img001.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to apply Gaussian blur to an image
def apply_gaussian_blur(image, k_size=5, sigma=0):
    """Apply Gaussian blur to smooth the image."""
    return cv2.GaussianBlur(image, (k_size, k_size), sigmaX=sigma)

# Function to compute the Sobel gradient of the image
def sobel_operator(image, k_size=3):
    """Calculate gradient magnitude using Sobel operator."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k_size)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k_size)
    gradient = cv2.magnitude(grad_x, grad_y)
    return gradient

# Function to apply Canny edge detection
def canny_edge_detection(image, low, high):
    """Apply Canny edge detection algorithm."""
    img_uint8 = cv2.convertScaleAbs(image)
    return cv2.Canny(img_uint8, low, high)

# Function to apply double thresholding for edge detection
def double_thresholding(image, low, high):
    """Apply double thresholding to categorize pixel intensities."""
    output = np.zeros_like(image)
    weak, strong = 128, 255
    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image <= high) & (image >= low))
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak
    return output

# Function to link weak edges based on the presence of strong edges in their neighborhood
def edge_tracking(image):
    """Link weak edges based on strong edge neighborhood."""
    weak, strong = 128, 255
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] == weak:
                if np.any(image[i-1:i+2, j-1:j+2] == strong):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

# Function to display the processed image and save it in a specified folder
def display_and_save_image(image, title, folder, filename):
    """Display the image using matplotlib and save it to a folder."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)

# Process and display the results
blurred = apply_gaussian_blur(image)
display_and_save_image(blurred, "Blurred Image", "outputs", "blurred.jpg")

sobel_result = sobel_operator(blurred)
display_and_save_image(sobel_result, "Sobel Operator", "outputs", "sobel.jpg")

canny_result = canny_edge_detection(sobel_result, 50, 150)
display_and_save_image(canny_result, "Canny Edge Detection", "outputs", "canny.jpg")

double_thresh_result = double_thresholding(canny_result, 50, 150)
display_and_save_image(double_thresh_result, "Double Thresholding", "outputs", "double_thresh.jpg")

final_edges = edge_tracking(double_thresh_result)
display_and_save_image(final_edges, "Edge Tracking", "outputs", "final_edges.jpg")
