
# Advanced Edge Detection with OpenCV

This project implements an advanced edge detection algorithm using Python's OpenCV library. The pipeline includes Gaussian blurring, Sobel filtering, Canny edge detection, double thresholding, and edge tracking. Processed images at each stage are both displayed and saved in the "outputs" folder.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- OpenCV (cv2)
- Numpy
- Matplotlib
- Os

You can install them using pip:

```bash
pip install opencv-python numpy matplotlib
```

### Usage

1. Clone this repository or copy the code.
2. Place your target image in the root directory and name it `img001.png`.
3. Execute the main script.

The processed images at various stages will be saved in the "outputs" directory.

## Pipeline

1. **Gaussian Blur**: The image is smoothened to reduce noise and improve edge detection.
   
2. **Sobel Operator**: This operator is used to compute the gradient magnitude, emphasizing edges.

3. **Canny Edge Detection**: The Sobel-filtered image undergoes Canny edge detection to highlight the primary edges.

4. **Double Thresholding**: Pixels are categorized based on their intensities to differentiate between weak and strong edges.

5. **Edge Tracking**: Weak edges are either promoted to strong edges or suppressed based on the presence of strong edges in their neighborhood.

## Viewing Results

You can find the processed images in the "outputs" directory. Additionally, each image will be displayed using `matplotlib` upon processing.

## Troubleshooting

Ensure that the input image is named `img001.png` and is located in the root directory of the project.

## License

This project is open-source. Feel free to modify, distribute, and utilize as per your needs.

---

