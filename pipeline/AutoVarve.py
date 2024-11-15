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
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import django

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'autovarve_project.settings')
django.setup()

from autovarve.models import PipeRun, CoreColumn

ALLOWED_MODES = ["RGB", "GRAY"]
ALLOWED_KERNEL_FUNCTIONS = ["MEAN", "MEDIAN", "MAX"]


class AutoVarve(object):
    def __init__(
            self,
            config_file,
            image_directory=None,
            save_to_db=False
    ):
        self.save_to_db = save_to_db
        self.config_file = self.load_config_file(config_file)

        # Define verbosity
        if "verbose" in self.config_file:
            self.verbose = int(self.config_file["verbose"])
        else:
            self.verbose = 0  # Default

        if image_directory is None:
            self.image_directory = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "images"
            )
        else:
            self.image_directory = image_directory

        # Define mode
        if "mode" in self.config_file and self.config_file["mode"] in ALLOWED_MODES:
            self.mode = self.config_file["mode"]
        else:
            if self.config_file["mode"] not in ALLOWED_MODES:
                raise Exception(
                    f"Config file contains invalid 'mode'. Allowed modes are {ALLOWED_MODES}, and found "
                    f'{self.config_file["mode"]}'
                )
            self.mode = "GRAY"  # Default

        # Preprocessing params
        if "scale_pixel_value_max" in self.config_file:
            self.scale_pixel_value_max = int(self.config_file["scale_pixel_value_max"])
        else:
            self.scale_pixel_value_max = None  # Do not scale

        self.crop = [0] * 4  # Default to no cropping (order: [left, right, top, bottom])
        if "crop" in self.config_file:
            for i, direction in enumerate(["left", "right", "top", "bottom"]):
                if direction in self.config_file["crop"]:
                    self.crop[i] = int(self.config_file["crop"][direction])

        if (
                "kernel_config" in self.config_file
                and all(
            [
                side in self.config_file["kernel_config"]
                for side in ["horizontal", "vertical"]
            ]
        )
                and all(
            [
                all(
                    [
                        key in self.config_file["kernel_config"][side]
                        for side in ["horizontal", "vertical"]
                    ]
                )
                for key in ["size", "function"]
            ]
        )
        ):
            self.horizontal_kernel_size = int(
                self.config_file["kernel_config"]["horizontal"]["size"]
            )
            self.vertical_kernel_size = int(
                self.config_file["kernel_config"]["vertical"]["size"]
            )

            if all(
                    [
                        self.config_file["kernel_config"][side]["function"]
                        in ALLOWED_KERNEL_FUNCTIONS
                        for side in ["vertical", "horizontal"]
                    ]
            ):
                self.horizontal_kernel_function = self.config_file["kernel_config"][
                    "horizontal"
                ]["function"]
                self.vertical_kernel_function = self.config_file["kernel_config"][
                    "vertical"
                ]["function"]
            else:
                raise Exception(
                    f"Kernel function in config file must be one of: {ALLOWED_KERNEL_FUNCTIONS}"
                )
        else:
            raise Exception("Kernel config not allowed.")

        if "pixel_change_threshold" in self.config_file:
            self.pixel_change_threshold = self.config_file["pixel_change_threshold"]
        else:
            self.pixel_change_threshold = 0.05  # Default

        # Create Django PipeRun object
        if save_to_db:
            piperun_object = PipeRun.objects.create(mode=self.mode,
                                                    scale_pixel_value_max=self.scale_pixel_value_max,
                                                    crop_left=self.crop[0],
                                                    crop_right=self.crop[1],
                                                    crop_top=self.crop[2],
                                                    crop_bottom=self.crop[3],
                                                    kernel_size_horizontal=self.horizontal_kernel_size,
                                                    kernel_size_vertical=self.vertical_kernel_size,
                                                    kernel_function_horizontal=self.horizontal_kernel_function,
                                                    kernel_function_vertical=self.vertical_kernel_function,
                                                    pixel_change_threshold=self.pixel_change_threshold)
            self.piperun_id = piperun_object.id
        else:
            self.piperun_id = None

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
        threshold = self.pixel_change_threshold
        varve_counts = self.generate_varve_counts(
            image_samples=image_samples, threshold=threshold
        )

        # Save varve_counts
        if self.save_to_db:
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
                all_image_tensors = torch.zeros(
                    (batch_size, c, h, w), dtype=torch.float32
                )
            all_image_tensors[i] = single_image_tensor

        if self.verbose > 0:
            print(
                f"(1.1)\tUPLOADING: Loaded tensor of shape: {all_image_tensors.shape}"
            )
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
            max_value = (
                image_tensors.max()
            )  # Get current max value, assuming it shows up in the image
            image_tensors = image_tensors.float() * (
                    self.scale_pixel_value_max / max_value
            )
            if self.verbose:
                print(
                    f"(2.1)\tPREPROCESSING: Scaling images such that max value is {image_tensors.max()}"
                )
        else:
            if self.verbose:
                print(f"(2.1)\tPREPROCESSING: Skip scaling.")
        new_max_value = image_tensors.max()

        # Crop images on the left, right, top and bottom
        image_tensors = image_tensors[
                        :, :, self.crop[2]: -self.crop[3], self.crop[0]: -self.crop[1]
                        ]
        if self.verbose:
            print(
                f"(2.2)\tPREPROCESSING: Cropped image so that current shape is {image_tensors.shape}"
            )

        # TODO: Exclude max pixel value

        return image_tensors

    def save_tensors(self, image_tensors, save_directory=None):
        """
        Save tensors to a save directory
        :param image_tensors:
        :return:
        """
        if save_directory is None:
            save_directory = os.path.join(
                os.path.dirname(self.image_directory), "other"
            )

    def save_tensors_to_txt(self, image_tensors, save_directory=None, save_filename=None):
        """
        Save tensors to a save directory
        :param image_tensors:
        :return:
        """
        if save_directory is None:
            save_directory = os.path.join(
                os.path.dirname(self.image_directory), "other"
            )
        if save_filename is None:
            save_filename = f'tensor_to_txt_{datetime.now().strftime("%d%m%Y%H%M%S")}.txt'
        output_path = os.path.join(save_directory, save_filename)

        image_tensors = image_tensors.squeeze()

        if len(image_tensors.shape) > 2:
            if self.verbose:
                print('Image tensor contains multiple images or multiple channels. Selecting first image and channel '
                      'for saving to text.')
            image_tensors = image_tensors[0, 0, :, :].squeeze()

        bools = image_tensors.cpu().numpy()

        # Save to file with row numbers
        with open(output_path, "w") as f:
            header = "Row_Group\t" + "\t".join([f"column_{i+1}" for i in range(bools.shape[1])]) + "\n"
            f.write(header)  # Header
            for i, bool_row in enumerate(bools):
                f.write(f"{i}\t" + "\t".join([f"{bool_value}" for bool_value in bool_row]) + "\n")

        print(f"Derivatives above threshold saved to: {output_path}")

    def save_counts(self, varve_counts):
        """
        Save varve counts to SQLite db
        :param varve_counts:
        :return:
        """
        new_corecolumn_created = 0
        all_columns = len(varve_counts)
        for i, varve_count in enumerate(varve_counts):
            column_order = i + 1
            column_width = self.horizontal_kernel_size

            # If we crop the first 10 pixels, the first column starts at 10 + 1 = 11
            pixel_start = self.crop[0] + (i * self.horizontal_kernel_size) + 1

            # If we crop the first 10 pixels, and the width is 10, the first column ends at 10 + 10 = 20
            pixel_end = self.crop[0] + (i + 1) * self.horizontal_kernel_size

            corecolumn_object, created = CoreColumn.objects.get_or_create(pipe_run_id=self.piperun_id,
                                                           column_order=column_order,
                                                           column_width=column_width,
                                                           pixel_start=pixel_start,
                                                           pixel_end=pixel_end,
                                                           varve_count=varve_count)
            if created:
                new_corecolumn_created += 1
        print(f'Created {new_corecolumn_created}/{all_columns} new CoreColumn objects')


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
        b, c, h, w = image_tensors.shape

        # Step 1: Transform pixels vertically
        if self.vertical_kernel_function == "MEAN":
            reshaped = image_tensors.reshape(
                b, c, h // self.vertical_kernel_size, self.vertical_kernel_size, w
            )
            vertically_transformed = reshaped.mean(dim=3)  # Shape: [1, 1, 20000, 1000]
        elif self.vertical_kernel_function == "MEDIAN":
            vertically_transformed = []
            for i in range(
                    h // self.vertical_kernel_size
            ):  # h // self.vertical_kernel_size groups
                start_idx = i * self.vertical_kernel_size
                end_idx = (i + 1) * self.vertical_kernel_size
                group_median = torch.median(
                    image_tensors[:, :, start_idx:end_idx, :], dim=2
                ).values
                vertically_transformed.append(group_median.unsqueeze(2))
            vertically_transformed = torch.cat(vertically_transformed, dim=2)
        elif self.vertical_kernel_function == "MAX":
            reshaped = image_tensors.reshape(b, c, h // self.vertical_kernel_size, w)
            vertically_transformed = reshaped.max(dim=2)  # Shape: [1, 1, 20000, 1000]
        else:
            raise Exception(
                f"Kernel function {self.vertical_kernel_function} not implemented"
            )
        image_tensors = vertically_transformed

        if self.verbose:
            print(
                f"(3.1)\tTRANSFORM: Vertical transformation with kernel size {self.vertical_kernel_size} and function"
                f" {self.vertical_kernel_function} resulted in tensor of shape: {image_tensors.shape}"
            )

        b, c, h2, w = image_tensors.shape
        # Step 2: Transform horizontally
        if self.horizontal_kernel_function == "MEAN":
            reshaped = image_tensors.reshape(
                b, c, h2, w // self.horizontal_kernel_size, self.horizontal_kernel_size
            )
            horizontally_transformed = reshaped.mean(
                dim=4
            )  # Shape: [1, 1, 20000, 1000]
        elif self.horizontal_kernel_function == "MEDIAN":
            horizontally_transformed = []
            for i in range(
                    w // self.horizontal_kernel_size
            ):  # w // self.horizontal_kernel_size groups
                start_idx = i * self.horizontal_kernel_size
                end_idx = (i + 1) * self.horizontal_kernel_size
                group_median = torch.median(
                    image_tensors[:, :, :, start_idx:end_idx], dim=3
                ).values
                horizontally_transformed.append(group_median.unsqueeze(3))
            horizontally_transformed = torch.cat(horizontally_transformed, dim=3)
        elif self.horizontal_kernel_function == "MAX":
            reshaped = image_tensors.reshape(b, c, h2, w // self.horizontal_kernel_size)
            horizontally_transformed = reshaped.max(dim=3)  # Shape: [1, 1, 20000, 1000]
        else:
            raise Exception(
                f"Kernel function {self.horizontal_kernel_function} not implemented"
            )
        image_tensors = horizontally_transformed

        if self.verbose:
            print(
                f"(4.2)\tTRANSFORM: Horizontal transformation with kernel size {self.horizontal_kernel_size} and function "
                f"{self.horizontal_kernel_function} resulted in tensor of shape: {image_tensors.shape}"
            )

        # TODO: implement avg_pool2d

        return image_tensors

    def generate_samples(self, image_tensors):
        """
        Say you have a transformed grayscale image tensor of shape [1, 1, 2744, 5]. Each of the 120 columns is
        potentially a different data sample, with its own total number of varves. The reliability of the overall varve
        count relies on the fact that each of these 120 columns should be correlated if the color is consistent
        throughout the horizontal core, which tends to be true. Sometimes the darker part of the varve is diagonal,
        which means the pixel values are correlated with some lag/lead. Do we keep the 120 columns or do we further
        combine pixel values horizontally?
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
        changes = image_samples[:, :, 1:, :] - image_samples[:, :, :-1, :]
        print(changes)
        if self.verbose:
            print(
                f"\n(4.1)\tComputing derivative resulted in a tensor of shape {changes.shape}"
            )
        return changes

    def varve_counts_by_color_threshold(self, sample_changes, threshold):
        """
        Compute the number of times each sample goes above the threshold.
        :param sample_changes:
        :param threshold:
        :return:
        """
        above_threshold = sample_changes > threshold

        print(above_threshold)
        # Save boolean array
        self.save_tensors_to_txt(above_threshold, save_filename=f'above_threshold_{threshold}.txt')

        # Compute cross-correlation with neighbors

        varve_counts = above_threshold.sum(dim=2, keepdim=True)
        return varve_counts

    def generate_varve_counts(self, image_samples, threshold):
        """
        For each sample, where each sample is a column, compute the change in the color of the pixels vertically.
        :param image_samples:
        :return:
        """
        # First, compute the change in the pixel values
        sample_changes = self.compute_sample_derivative(image_samples)

        sample_change_deciles = self.quantize_to_deciles(sample_changes)
        # self.save_tensors_to_txt(sample_change_deciles, save_filename='decile_changes.txt')

        # Compute cross-correlation of columns
        for column_num in range(sample_change_deciles.shape[3] - 1):
            lags, correlations = self.compute_column_cross_correlation(tensor=sample_change_deciles,
                                                                       col1_idx=column_num,
                                                                       col2_idx=column_num + 1,
                                                                       max_lag=10)
            self.plot_cross_correlation(lags, correlations)

        if threshold is None:
            threshold = 0.05
        varve_counts = self.varve_counts_by_color_threshold(
            sample_changes, threshold=threshold
        ).squeeze()

        if self.verbose:
            print(f"Columnwise counts above {threshold} threshold: {varve_counts}")
        return varve_counts

    def plot_counts_histogram(self, varve_counts):
        """
        Plot histogram of counts.
        :param varve_counts:
        :return:
        """
        pass

    def quantize_to_deciles(self, tensor):
        """
        Convert tensor values to decile buckets (1-10).

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch, channels, height, width]

        Returns:
            torch.Tensor: Tensor of same shape with values replaced by their decile bucket numbers (1-10)
        """
        # Create output tensor of same shape
        quantized = torch.zeros_like(tensor)

        # Get the dimensions
        batch, channels, height, width = tensor.shape

        # Process each batch and channel separately
        for b in range(batch):
            for c in range(channels):
                # Get the 2D slice
                data = tensor[b, c]

                # Calculate decile boundaries (0 to 100 by 10)
                deciles = torch.tensor(
                    np.percentile(data.numpy(), np.arange(0, 101, 10))
                )

                # Assign bucket numbers (1-10) based on which decile range the value falls into
                for i in range(10):
                    if i == 0:
                        mask = data <= deciles[i + 1]
                    elif i == 9:
                        mask = data > deciles[i]
                    else:
                        mask = (data > deciles[i]) & (data <= deciles[i + 1])

                    quantized[b, c][mask] = i + 1

        return quantized

    def compute_column_cross_correlation(self, tensor, col1_idx=0, col2_idx=1, max_lag=None):
        """
        Compute cross-correlation between two columns of a tensor at different lags.

        Args:
            tensor (torch.Tensor): Input tensor of shape [1, 1, height, width]
            col1_idx (int): Index of first column (default: 0)
            col2_idx (int): Index of second column (default: 1)
            max_lag (int): Maximum lag to compute (default: None, uses full length)

        Returns:
            lags (numpy.ndarray): Array of lag values
            correlations (numpy.ndarray): Cross-correlation values for each lag
        """
        # Extract the columns (removing batch and channel dimensions)
        col1 = tensor[0, 0, :, col1_idx].numpy()
        col2 = tensor[0, 0, :, col2_idx].numpy()

        # Standardize the columns (subtract mean and divide by std)
        col1_std = (col1 - np.mean(col1)) / np.std(col1)
        col2_std = (col2 - np.mean(col2)) / np.std(col2)

        # Compute cross-correlation
        correlations = np.correlate(col1_std, col2_std, mode='full')

        # Calculate lags
        n = len(col1)
        lags = np.arange(-(n - 1), n)

        # If max_lag is specified, truncate the results
        if max_lag is not None:
            mask = np.abs(lags) <= max_lag
            lags = lags[mask]
            correlations = correlations[mask]
            print(f'Correlations between col {col1_idx} and {col2_idx}: {correlations}\n')

        return lags, correlations

    def plot_cross_correlation(self, lags, correlations):
        """
        Plot the cross-correlation results.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(lags, correlations)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.grid(True)
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.title('Cross-correlation between columns')
        plt.show()


if __name__ == "__main__":
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "example_configs",
        "example_config.json",
    )
    av = AutoVarve(config_file=config_file, save_to_db=False)
    av.execute()
