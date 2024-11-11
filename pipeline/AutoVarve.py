import os
import sys
import json
import torch
import torchvision
from torchvision.io import read_image, ImageReadMode, decode_image
import torchvision.transforms as T
import os
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ALLOWED_MODES = ['RGB', 'GRAY']
ALLOWED_KERNEL_FUNCTIONS = ['MEAN', 'MEDIAN', 'MAX']


class AutoVarve(object):
    def __init__(self, config_file, image_directory=None, ):
        if image_directory is None:
            self.image_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'images')
        else:
            self.image_directory = image_directory

        self.config_file = self.load_config_file(config_file)

        # Define mode
        if 'mode' in self.config_file and self.config_file['mode'] in ALLOWED_MODES:
            self.mode = self.config_file['mode']
        else:
            if self.config_file['mode'] not in ALLOWED_MODES:
                raise Exception(f'Config file contains invalid \'mode\'. Allowed modes are {ALLOWED_MODES}, and found '
                                f'{self.config_file["mode"]}')
            self.mode = 'GRAY'  # Default

        # Define verbosity
        if 'verbose' in self.config_file:
            self.verbose = int(self.config_file['verbose'])
        else:
            self.verbose = 0  # Default

        # Preprocessing params
        if 'scale_pixel_value_max' in self.config_file:
            self.scale_pixel_value_max = int(self.config_file['scale_pixel_value_max'])
        else:
            self.scale_pixel_value_max = None  # Do not scale

        self.crop = [0] * 4  # Default to no cropping (order: [left, right, top, bottom])
        if 'crop' in self.config_file:
            for i, direction in enumerate(['left', 'right', 'top', 'bottom']):
                if direction in self.config_file['crop']:
                    self.crop[i] = int(self.config_file['crop'][direction])

        if ('kernel_config' in self.config_file
                and all([side in self.config_file['kernel_config'] for side in ['horizontal', 'vertical']])
                and all([all([key in self.config_file['kernel_config'][side]
                              for side in ['horizontal', 'vertical']])
                         for key in ['size', 'function']])):
            self.horizontal_kernel_size = int(self.config_file['kernel_config']['horizontal']['size'])
            self.vertical_kernel_size = int(self.config_file['kernel_config']['vertical']['size'])

            if all([self.config_file['kernel_config'][side]['function'] in ALLOWED_KERNEL_FUNCTIONS
                    for side in ['vertical', 'horizontal']]):
                self.horizontal_kernel_function = self.config_file['kernel_config']['horizontal']['function']
                self.vertical_kernel_function = self.config_file['kernel_config']['vertical']['function']
            else:
                raise Exception(f'Kernel function in config file must be one of: {ALLOWED_KERNEL_FUNCTIONS}')
        else:
            raise Exception('Kernel config not allowed.')


    @staticmethod
    def load_config_file(config_file):
        """
        Parse config file into dict
        :param config_file:
        :return:
        """
        if not (os.path.isfile(config_file) and os.access(config_file, os.R_OK)):
            raise Exception(
                f"Config file {config_file} does not exist or is not readable. Please provide a valid config file."
            )
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return config_dict

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

        all_image_tensors = torch.zeros((1, 1), dtype=torch.float32)
        for i, image_name in enumerate(os.listdir(self.image_directory)):
            image_path = os.path.join(self.image_directory, image_name)
            single_image_tensor = decode_image(image_path, mode=self.mode)
            if i == 0:
                # Read first image to get dimensions
                c, h, w = single_image_tensor.shape
                # Initialize tensor to store all images
                batch_size = len(os.listdir(self.image_directory))
                all_image_tensors = torch.zeros((batch_size, c, h, w), dtype=torch.float32)
            all_image_tensors[i] = single_image_tensor

        if self.verbose > 0:
            print(f'(1.1)\tUPLOADING: Loaded tensor of shape: {all_image_tensors.shape}')
        return all_image_tensors

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
        if self.scale_pixel_value_max is not None:
            # Convert Tensor to Float and scale
            max_value = image_tensors.max()  # Get current max value, assuming it shows up in the image
            image_tensors = image_tensors.float() * (self.scale_pixel_value_max / max_value)
            if self.verbose:
                print(f'(2.1)\tPREPROCESSING: Scaling images such that max value is {image_tensors.max()}')
        else:
            if self.verbose:
                print(f'(2.1)\tPREPROCESSING: Skip scaling.')
        new_max_value = image_tensors.max()

        # Crop images on the left, right, top and bottom
        image_tensors = image_tensors[:, :, self.crop[2]:-self.crop[3], self.crop[0]:-self.crop[1]]
        if self.verbose:
            print(f'(2.2)\tPREPROCESSING: Cropped image so that current shape is {image_tensors.shape}')

        # TODO: Exclude max pixel value

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
        b, c, w, h = image_tensors.shape

        vertical_group_size = self.vertical_group_size
        # Step 1: Average pixels vertically
        reshaped = image_tensors.reshape(b, h // vertical_group_size, vertical_group_size, width)
        vertically_averaged = reshaped.mean(dim=2)  # Shape: [1, 8220, 2000]


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


if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'example_configs', 'example_config.json')
    av = AutoVarve(config_file=config_file)
    av.execute()



