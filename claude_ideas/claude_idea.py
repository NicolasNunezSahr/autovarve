import numpy as np
import cv2
from scipy import signal
# from skimage import filters
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from typing import List


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


def crop_core_image(img, top=0, bottom=43673, left=2000, right=1025):
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


def validate_results(image_path, human_labels_path=None, save_visualization=None):
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
    print(f"Detected {num_varves} varves in the following pixel rows: {peaks}")

    human_labels_df = pd.read_csv(human_labels_path)

    if human_labels_path is not None:
        correct_preds = []
        incorrect_preds = []
        for peak in peaks:
            if is_within_labels_range(human_labels_df, pixel_index=peak):
                correct_preds.append(peak)
            else:
                incorrect_preds.append(peak)

        print(f'Varve accuracy: {len(correct_preds)/(len(correct_preds) + len(incorrect_preds))}.\n'
              f'Correct preds: {correct_preds}')

    return num_varves, marked_img


def is_within_labels_range(human_labels_df, pixel_index, leeway=10):
    """
    Determines whether pixel_index is contained within the dataframe.
    :param human_labels_df:
    :param pixel_index:
    :param leeway:
    :return:
    """
    cond = ((human_labels_df['start_pixel_row'] - leeway < pixel_index) &
             (human_labels_df['end_pixel_row'] + leeway > pixel_index))
    filtered_df = human_labels_df.loc[cond, :]
    for index, row in filtered_df.iterrows():
        if row['start_pixel_row'] - leeway <= pixel_index <= row['end_pixel_row'] + leeway:
            return True
    return False


def compute_precision(human_labels_df: pd.DataFrame, pixel_index_pred_list: List):
    correct_preds = []
    incorrect_preds = []
    for pixel_index_pred in pixel_index_pred_list:
        if is_within_labels_range(human_labels_df, pixel_index=pixel_index_pred):
            correct_preds.append(pixel_index_pred)
        else:
            incorrect_preds.append(pixel_index_pred)
    precision = len(correct_preds) / (len(correct_preds) + len(incorrect_preds))
    print(f'Varve precision: {precision}.\n'
          f'Correct preds: {correct_preds}')

    return precision, correct_preds




if __name__ == "__main__":

    # With validation against human labels
    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'images', 'D15-4Lspliced_no ruler.png')
    human_labels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'labeled_images', 'human_labels.csv')
    varve_count, marked_img = validate_results(image_path, human_labels_path=human_labels_path, save_visualization="varves_marked.jpg")