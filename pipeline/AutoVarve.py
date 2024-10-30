import os
import sys


class AutoVarve(object):
    def __init__(self, config_file, image_directory=None, ):
        if image_directory is None:
            self.image_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'images')
        else:
            self.image_directory = image_directory

        self.config_file = config_file

    def execute(self):
        # First step is to load image files in images directory
        image_tensors = self.load_images()

        # Preprocess images
        image_tensors = self.preprocess_images(image_tensors)

        # Checkpoint here to save results
        self.save_tensors(image_tensors)

        # Transform images
        image_tensors = self.transform_images(image_tensors)

        # Split image column-wise te generate distinct samples from within the core
        image_samples = self.generate_samples(image_tensors)

        # Maybe loop over this section to compute a histogram for each threshold value
        # Count varves
        threshold = 0.05
        varve_counts = self.generate_varve_counts(image_samples=image_samples, threshold=threshold)

        # Save varve_counts
        self.save_counts(varve_counts)

        # Plot histogram of counts
        self.plot_counts_histogram(varve_counts)

    def load_images(self):
        """
        Load images from png format into tensor. Must decide on a loading mode, whether RGB or grayscale.
        :return:
        """
        image_tensors = None
        return image_tensors

    def preprocess_images(self, image_tensors):
        """
        Run preprocessing of images. Various judgment calls need to be made here, detailed below:
            1. Whether to convert values from 0 to 255 into 0 to 1.
            2. Where to crop the image on the left, right, top, and bottom.
            3. Whether to set any pixel values to NA, so that they do not bias results or processing steps carried out
                on image.
        :param image_tensors: [batch, channels, height, width]
        :return:
        """
        return image_tensors

    def save_tensors(self, image_tensors, save_directory=None):
        """
        Save tensors to a save directory
        :param image_tensors:
        :return:
        """
        if save_directory is None:
            save_directory = os.path.join(os.path.dirname(self.image_directory), 'other')

    def save_counts(self, varve_counts, save_directory=None):
        """
        Save tensors to a save directory
        :param image_tensors:
        :return:
        """
        if save_directory is None:
            save_directory = os.path.join(os.path.dirname(self.image_directory), 'other')

    def transform_images(self, image_tensors):
        """
        Compute transformations of cropped images.
        What transformations to carry out on images.
        Options:
            1. average/median/max pooling of pixel groups; stride, padding, kernel size (important knobs). This will
                increase the blurriness of the image, but will also smooth out the image. The larger the kernel size,
                the smoother the pixel changes will be.
            2. Other transformations?
        :param image_tensors: [batch, channels, height, width]
        :return:
        """
        return image_tensors

    def generate_samples(self, image_tensors):
        """
        Say you have a transformed grayscale image tensor of [120, 120]. Each of the 120 columns is potentially a
        different data sample, with its own total number of varves. The reliability of the overall varve count relies
        on the fact that each of these 120 columns should be correlated if the color is consistent throughout the
        horizontal core, which tends to be true. Sometimes the darker part of the varve is diagonal, which means the
        pixel values are correlated with some lag/lead. Do we keep the 120 columns or do we further combine pixel
        values horizontally?
        :param image_tensors:
        :return:
        """
        return image_tensors

    def compute_sample_derivative(self, image_samples):
        """
        Compute color changes column-wise.
        :param image_samples:
        :return:
        """
        return image_samples

    def varve_counts_by_color_threshold(self, sample_changes, threshold):
        """
        Compute the number of times each sample goes above the threshold.
        :param sample_changes:
        :param threshold:
        :return:
        """
        varve_counts = 'Something'
        return varve_counts

    def generate_varve_counts(self, image_samples, threshold):
        """
        For each sample, where each sample is a column, compute the change in the color of the pixels vertically.
        :param image_samples:
        :return:
        """
        sample_changes = self.compute_sample_derivative(image_samples)

        if threshold is None:
            threshold = 0.05
        varve_counts = self.varve_counts_by_color_threshold(sample_changes, threshold=threshold)
        return varve_counts

    def plot_counts_histogram(self, varve_counts):
        """
        Plot histogram of counts.
        :param varve_counts:
        :return:
        """
        pass



