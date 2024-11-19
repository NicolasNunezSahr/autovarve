import numpy as np
import cv2
from scipy import signal
# from skimage import filters
import matplotlib.pyplot as plt
import os


def create_varve_visualization(img, peaks, save_path=None):
    """
    Create a visualization of detected varves with red lines.

    Parameters:
    img: Original image
    peaks: Array of peak positions (varve locations)
    save_path: Optional path to save the visualization

    Returns:
    numpy array: Image with varves marked
    """
    # Create a copy of the image to draw on
    marked_img = img.copy()

    # Draw red lines at each peak location
    for peak in peaks:
        cv2.line(marked_img,
                 (0, peak),
                 (marked_img.shape[1], peak),
                 (0, 0, 255),  # BGR format - red color
                 2)  # Line thickness

    # Save the visualization if a path is provided
    if save_path:
        cv2.imwrite(save_path, marked_img)

    return marked_img


def crop_core_image(img, top=135, bottom=43673, left=2000, right=1025):
    """
    Crop the image according to specified dimensions.

    Parameters:
    img: Input image (numpy array)
    top: Number of rows to remove from top
    bottom: Number of rows to remove from bottom
    left: Number of columns to remove from left
    right: Number of columns to remove from right

    Returns:
    numpy array: Cropped image
    """
    height, width = img.shape[:2]

    # Calculate the bottom and right coordinates
    bottom_coord = height - bottom
    right_coord = width - right

    # Perform the crop
    cropped = img[top:bottom_coord, left:right_coord]

    return cropped


def process_core_image(image_path, debug=False, save_visualization=None):
    """
    Process a sediment core image to detect and count varves.

    Parameters:
    image_path (str): Path to the input image
    debug (bool): If True, shows intermediate processing steps
    save_visualization (str): Optional path to save the varve visualization

    Returns:
    tuple: (number of varves, processed image, peak locations, marked image)
    """
    # Read image
    img = cv2.imread(image_path)

    # Crop image
    img = crop_core_image(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Compute vertical gradient (varves are horizontal layers)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_y = np.abs(gradient_y)

    # Normalize gradient
    gradient_normalized = cv2.normalize(gradient_y, None, 0, 255, cv2.NORM_MINMAX)

    # Calculate mean intensity profile along horizontal axis
    intensity_profile = np.mean(gradient_normalized, axis=1)

    # Find peaks in intensity profile
    peaks, _ = signal.find_peaks(intensity_profile,
                                 distance=10,  # Minimum distance between peaks
                                 prominence=10)  # Minimum prominence of peaks

    # Create visualization with marked varves
    marked_img = create_varve_visualization(img, peaks, save_visualization)

    if debug:
        # Visualize results
        plt.figure(figsize=(15, 12))

        plt.subplot(231)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Original')

        plt.subplot(232)
        plt.imshow(enhanced, cmap='gray')
        plt.title('Contrast Enhanced')

        plt.subplot(233)
        plt.imshow(gradient_normalized, cmap='gray')
        plt.title('Vertical Gradient')

        plt.subplot(234)
        plt.plot(intensity_profile)
        plt.plot(peaks, intensity_profile[peaks], "rx")
        plt.title('Intensity Profile with Peaks')

        plt.subplot(235)
        plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Varves (Count: {len(peaks)})')

        plt.tight_layout()
        plt.show()

    return len(peaks), gradient_normalized, peaks, marked_img


def validate_results(image_path, manual_count=None, save_visualization=None):
    """
    Process an image and validate results against manual count if provided.

    Parameters:
    image_path (str): Path to the input image
    manual_count (int): Optional manual count for validation
    save_visualization (str): Optional path to save the varve visualization
    """
    num_varves, processed_img, peaks, marked_img = process_core_image(
        image_path,
        debug=True,
        save_visualization=save_visualization
    )
    print(f"Detected {num_varves} varves")

    if manual_count is not None:
        error = abs(manual_count - num_varves)
        error_percentage = (error / manual_count) * 100
        print(f"Manual count: {manual_count}")
        print(f"Absolute error: {error}")
        print(f"Error percentage: {error_percentage:.2f}%")

    return num_varves, marked_img


if __name__ == "__main__":
    # Basic usage
    num_varves, gradient_img, peak_locations, marked_img = process_core_image(os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'images', 'D15-4Lspliced_no ruler.png'), debug=True)

    # With validation against manual count
    varve_count, marked_img = validate_results(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data',
                                                    'images', 'D15-4Lspliced_no ruler.png'), manual_count=70,
                                                save_visualization="varves_marked.jpg")