
---

## Edge Detection with Gaussian Blur, Sobel Filter, and Non-Maximum Suppression

This code applies a series of image processing techniques on an input grayscale image to achieve clear and thin edges. The steps involved are:
1. Apply Gaussian blur to smooth the image.
2. Use the Sobel filter for edge detection.
3. Refine the detected edges using Non-Maximum Suppression.

### Dependencies

- Numpy
- SciPy
- OpenCV (cv2)
- Matplotlib

### Code Breakdown

#### Gaussian Kernel Generation

```python
def gaussian(x, y, sigma=1.0):
    ...
def gaussian_kernel(size, sigma=1.0):
    ...
```

These functions generate a Gaussian kernel of a specified size and standard deviation (sigma). The kernel is then used to blur the image.

#### Sobel Filter

```python
def sobel_filters(img):
    ...
```

The Sobel filter is applied to detect edges in the blurred image. This filter calculates the gradient magnitude and direction for each pixel in the image.

#### Non-Maximum Suppression

```python
def non_maximum_suppression(img, D):
    ...
```

After edge detection using the Sobel filter, the edges are usually thick. To thin them, the Non-Maximum Suppression technique is applied. It compares the gradient magnitude of the current pixel with its neighbors along the gradient direction.

### Main Program

The main program performs the following operations:

1. Load the image in grayscale format.
2. Apply the Gaussian blur.
3. Use the Sobel filter for edge detection.
4. Apply Non-Maximum Suppression to refine the edges.
5. Display the results.

```python
image_path = 'path_to_your_image.jpg'
...
plt.tight_layout()
plt.show()
```

### Execution

Make sure to update `'path_to_your_image.jpg'` with the correct path to your image. Once set, execute the code. It will display the original image, the blurred image, the result from the Sobel filter, and the final result after applying Non-Maximum Suppression.

---

---

### Full Code with Logic Explanation

#### Gaussian Kernel Generation

Before detecting edges, it's a common practice to reduce noise in the image. We do this using the Gaussian blur.

```python
def gaussian(x, y, sigma=1.0):
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
```

The above function calculates the value of the Gaussian function for given `x`, `y`, and `sigma`. 

```python
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
```

Here, we generate a 2D Gaussian kernel. The center of the kernel corresponds to the peak of the Gaussian function.

#### Sobel Filter

After blurring, we detect edges using the Sobel filter. This filter calculates the gradient for each pixel in the image. High gradients correspond to potential edges.

```python
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)
```

We have two kernels, `Kx` and `Ky`, for horizontal and vertical edges respectively. The convolution of the image with these kernels results in gradient images (`Ix` and `Iy`). The magnitude (`G`) and direction (`theta`) of the gradient are calculated from these.

#### Non-Maximum Suppression

To thin the detected edges, we apply Non-Maximum Suppression (NMS). This step ensures that the final edges are single-pixel wide.

```python
def non_maximum_suppression(img, D):
    ...
```

The logic behind NMS is to traverse the gradient image and keep only those pixels that have the maximum gradient magnitude in their gradient direction. Others are set to zero.

#### Main Program

Finally, the main logic ties all the above steps together.

```python
image_path = 'path_to_your_image.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

kernel = gaussian_kernel(5, 1.0)
gaussian_blurred = ndimage.convolve(img, kernel)
gradient_magnitude, gradient_direction = sobel_filters(gaussian_blurred)
nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)

plt.figure(figsize=(16, 6))
...
plt.tight_layout()
plt.show()
```

1. The image is loaded in grayscale.
2. Gaussian blur is applied to reduce noise.
3. Edges are detected using the Sobel filter.
4. Edges are refined using NMS.
5. Results from each step, along with the original image, are displayed side-by-side.

---

