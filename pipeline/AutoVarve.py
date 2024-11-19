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
import math
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
MAX_VARVE_THICKNESS = 20


class AutoVarve(object):
    def __init__(
            self,
            config_file,
            image_directory=None,
            save_to_db=False,
            human_labels_csv=None
    ):
        self.image_shape = (1, 65776, 4000)  # Defined inside self.set_image_shape()
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

        self.crop = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}  # Default to no cropping
        if "crop" in self.config_file:
            for i, direction in enumerate(["left", "right", "top", "bottom"]):
                if direction in self.config_file["crop"]:
                    self.crop[direction] = int(self.config_file["crop"][direction])

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

        if "correlation_group_size" in self.config_file:
            self.correlation_group_pixel_height = self.config_file['correlation_group_size']
        else:
            self.correlation_group_pixel_height = 1000  # Default

        if "vertical_or_aggregation_size" in self.config_file:
            self.vertical_or_aggregation_size = self.config_file['vertical_or_aggregation_size']
        else:
            self.vertical_or_aggregation_size = 3

        if "column_fraction_threshold" in self.config_file:
            self.column_fraction_threshold = self.config_file['column_fraction_threshold']
        else:
            self.column_fraction_threshold = 0.35

        # Create Django PipeRun object
        if save_to_db:
            piperun_object = PipeRun.objects.create(mode=self.mode,
                                                    scale_pixel_value_max=self.scale_pixel_value_max,
                                                    crop_left=self.crop['left'],
                                                    crop_right=self.crop['right'],
                                                    crop_top=self.crop['top'],
                                                    crop_bottom=self.crop['bottom'],
                                                    kernel_size_horizontal=self.horizontal_kernel_size,
                                                    kernel_size_vertical=self.vertical_kernel_size,
                                                    kernel_function_horizontal=self.horizontal_kernel_function,
                                                    kernel_function_vertical=self.vertical_kernel_function,
                                                    pixel_change_threshold=self.pixel_change_threshold,
                                                    vertical_or_aggregation_size=self.vertical_or_aggregation_size,
                                                    column_fraction_threshold=self.column_fraction_threshold)
            self.piperun_id = piperun_object.id
        else:
            self.piperun_id = None

        if human_labels_csv is not None:
            self.human_labels_df = pd.read_csv(human_labels_csv)
            if self.verbose:
                print(f'Loaded {self.human_labels_df.shape[0]} human-labeled varves')
            self.transform_human_labels()
        else:
            self.human_labels_df = None

        if "save_directory" in self.config_file:
            self.save_directory = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                self.config_file['save_directory']
            )
        else:
            self.save_directory = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "labeled_images"
            )

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

        # Maybe loop over this section to compute a histogram for each threshold value
        # Count varves
        varve_counts, varve_pixel_heights = self.generate_varve_counts(
            image_samples=image_tensors, threshold=self.pixel_change_threshold
        )

        # Save varve_counts
        if self.save_to_db:
            self.save_counts(varve_counts)

        # Plot histogram of counts
        self.plot_counts_histogram(varve_counts)

        # self.modify_image(varve_pixel_heights)

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
                self.set_image_shape(shape=single_image_tensor.shape)
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

    def modify_image(self, height_coordinates, line_thickness=10):
        """
        Draw horizontal lines on a grayscale image at specified height coordinates.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape [1, 1, H, W]
            height_coordinates (list): List of height coordinates where lines should be drawn
            line_thickness (int): Thickness of the lines in pixels
            save_dir (str, optional): Directory to save the modified image. If None, saves in current directory

        Returns:
            torch.Tensor: Modified image tensor
        """
        image_tensor = self.load_images()
        # Create a copy of the input tensor to avoid modifying the original
        modified_tensor = image_tensor.clone()

        # Get image dimensions
        _, _, height, width = modified_tensor.shape

        # Validate height coordinates
        height_coordinates = [h for h in height_coordinates if 0 <= h < height]

        # Draw horizontal lines
        for h in height_coordinates:
            # Calculate the range for line thickness
            start_h = max(0, h - line_thickness // 2)
            end_h = min(height, h + line_thickness // 2)

            # Set pixel values to white (1.0) for the line
            modified_tensor[0, 0, start_h:end_h, :] = 1.0

        # Ensure save directory exists
        os.makedirs(self.save_directory, exist_ok=True)

        # Convert tensor to PIL Image and save
        # Assuming values are in range [0, 1]
        image_pil = T.functional.to_pil_image(modified_tensor[0])

        # Generate save path
        base_name = 'image'  # Default name if original name not provided
        save_path = os.path.join(self.save_directory, f'{base_name}_modified.png')

        # Save the image
        image_pil.save(save_path)

        return modified_tensor



    def set_image_shape(self, shape):
        self.image_shape = shape

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
                        :, :, self.crop['top']: -self.crop['bottom'], self.crop['left']: -self.crop['right']
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
            pixel_start = self.crop['left'] + (i * self.horizontal_kernel_size) + 1

            # If we crop the first 10 pixels, and the width is 10, the first column ends at 10 + 10 = 20
            pixel_end = self.crop['left'] + (i + 1) * self.horizontal_kernel_size

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
                f"(3.2)\tTRANSFORM: Horizontal transformation with kernel size {self.horizontal_kernel_size} and function "
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
        if self.verbose:
            print(
                f"\n(4.1)\tComputing derivative resulted in a tensor of shape {changes.shape}"
            )
        return changes

    def varve_counts_by_color_threshold(self, sample_changes, threshold, group=None, l2r_idx=None, group_cols=True):
        """
        Compute the number of times each sample goes above the threshold.
        :param sample_changes:
        :param threshold:
        :return:
        """
        above_threshold = sample_changes > threshold

        # print(above_threshold)
        # Save boolean array
        save_basename = f'above_threshold_{threshold}'
        if group is not None:
            save_basename = f'{save_basename}_g{group}'
        if l2r_idx is not None:
            save_basename = f'{save_basename}_i{l2r_idx}'
        save_filename = f'{save_basename}.txt'
        self.save_tensors_to_txt(above_threshold, save_filename=save_filename)
        if group_cols:
            varve_counts, true_indices = self.count_horizontal_lines(tensor=above_threshold, group=group,
                                                                     blur_size=self.vertical_or_aggregation_size)
            varve_counts = torch.tensor(varve_counts)
        else:
            varve_counts = above_threshold.sum(dim=2, keepdim=True)
            true_indices = torch.where(above_threshold[0, 0])
            true_indices = list(zip(true_indices[0].tolist(), true_indices[1].tolist()))
        return varve_counts, true_indices

    def count_horizontal_lines(self, tensor, group=None, blur_size=3):
        # First, blur 3 rows together to increase collision probability
        N = tensor.shape[2]
        width = tensor.shape[3]
        output_size = math.ceil(N / blur_size)

        result = torch.zeros((1, 1, output_size, width), dtype=torch.bool, device=tensor.device)

        # Process complete groups of 3
        complete_groups = N // blur_size
        if complete_groups > 0:
            main_part = tensor[:, :, :complete_groups * blur_size, :]
            reshaped = main_part.reshape(1, 1, -1, blur_size, width)
            result[:, :, :complete_groups, :] = torch.any(reshaped, dim=3)

        remaining = N % blur_size
        if remaining > 0:
            start_idx = complete_groups * blur_size
            remaining_values = tensor[:, :, start_idx:, :]
            result[:, :, -1, :] = torch.any(remaining_values, dim=2)
        if self.verbose:
            print(f'Blurring resulted in tensor of shape {result.shape}')

        result = self.majority_vote_rows(result, threshold=math.ceil(result.shape[3] * self.column_fraction_threshold),
                                         group=group)
        true_indices = [i for i, value in enumerate(result.squeeze().tolist()) if value is True]

        if self.verbose:
            print(f'Majority vote led to a tensor of shape {result.shape} with true indices: '
                  f'\n{true_indices}')

        varve_count = result.sum(dim=2, keepdim=True)

        return varve_count, true_indices

    def majority_vote_rows(self, tensor, threshold=8, group=None):
        """
        For each row in the tensor, set it to True if the number of True values
        in that row exceeds the threshold.

        Args:
            tensor (torch.Tensor): Input tensor of shape [1, 1, 33, 15]
            threshold (int, optional): Minimum number of True values needed. Defaults to 8.

        Returns:
            torch.Tensor: Output tensor of shape [1, 1, 33, 1] where each value represents
                         the majority vote for that row
        """
        # Input validation
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")

        if len(tensor.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got shape {tensor.shape}")

        if tensor.dtype != torch.bool:
            raise TypeError("Input tensor must contain boolean values")

        # Count True values along the last dimension (columns)
        true_counts = torch.sum(tensor, dim=3)  # Shape: [1, 1, 33]

        # Compare with threshold
        result = (true_counts >= threshold)  # Shape: [1, 1, 33]

        # Add the final dimension to match desired output shape
        result = result.unsqueeze(-1)  # Shape: [1, 1, 33, 1]

        return result

    def generate_varve_counts(self, image_samples, threshold=None):
        """
        For each sample, where each sample is a column, compute the change in the color of the pixels vertically.
        :param image_samples:
        :return:
        """
        # First, compute the change in the pixel values
        sample_changes = self.compute_sample_derivative(image_samples)

        if threshold is None:
            threshold = 0.05

        # sample_change_deciles = self.quantize_to_deciles(sample_changes)
        # self.save_tensors_to_txt(sample_change_deciles, save_filename='decile_changes.txt')

        # Compute cross-correlation of columns
        # The maximum curvature of a varve line tends to be around 0.5, or 45 degrees; the thickness of a varve line
        # tends to be between 2 and 20 pixels. Hence, with raw pixels, we would want the max lag to be 20 + 1, so that
        # the bottom pixel of the first column can be correlated with the top of the other, thereby enforcing
        # continuity of the varve line. If the horizontal kernel size grows, then we want to shrink the max negative
        # lag proportionally to enforce the continuity constraint. For now we ignore the horizontal kernel size.
        left_column_max_upshift = math.ceil((MAX_VARVE_THICKNESS / self.vertical_kernel_size) + 1)

        # On the positive end i.e. shifting left column downwards to match the right column, is rare. However, this
        # could be because of our specific data sample. We set the max_downshift to the same, even though it would be
        # wise to set this to a smaller value due to our prior that negative-sloped varves are rare.
        left_column_max_downshift = math.ceil((MAX_VARVE_THICKNESS / self.vertical_kernel_size) + 1)

        # The edge columns, i.e. column_num values that are close to 0 or close to the max, tend to have more
        #  curvature due to the way sediment cores are extracted. Therefore, we could utilize this prior knowledge to
        #  limit our max positive/negative lag.
        #  Specifically, we could use a decreasing linear relationship between max negative lag and column_num:
        #  y_0 = (MAX_VARVE_THICKNESS / self.vertical_kernel_size) + 1 [i.e. max lag when column_num = 0]
        #  y_end = (MAX_VARVE_THICKNESS / 4 * self.vertical_kernel_size) + 1 [i.e. minimum possible max negative lag]
        #  left_column_max_upshift = y_0 - column_num * ( (y_0 - y_end) / sample_changes.shape[3] )
        group_size = math.ceil(self.correlation_group_pixel_height / self.vertical_kernel_size)  # Compute correlations at intervals of 1000 pixels
        num_groups = int((sample_changes.shape[2] // group_size) + 1)
        cumulative_varve_counts_per_col = [0] * sample_changes.shape[3]
        group_cols = True  # Whether to group columns or to have independent counts. Must be set to True
        varve_count = 0
        varve_pixel_heights = []
        for i in range(num_groups):
            start_idx = i * group_size
            if i + 1 == num_groups:  # last group
                end_idx = int(sample_changes.shape[2] - 1)  # last possible pixel
            else:
                end_idx = (i + 1) * group_size
            group_size = end_idx - start_idx  # Last group is smaller

            # Extract group
            sample_change_horizontal_group = sample_changes[:, :, start_idx:end_idx, :]
            print(f'Starting new group from index {start_idx} to {end_idx}, obtaining a tensor of shape {sample_change_horizontal_group.shape}')
            optimal_lag_list = []
            for column_num in range(sample_change_horizontal_group.shape[3] - 1):
                lags, correlations = self.compute_column_cross_correlation(tensor=sample_change_horizontal_group,
                                                                           col1_idx=column_num,
                                                                           col2_idx=column_num + 1,
                                                                           left_column_max_upshift=left_column_max_upshift,
                                                                           left_column_max_downshift=left_column_max_downshift)
                # self.plot_cross_correlation(lags=lags, correlations=correlations, col1_idx=column_num,
                #                             col2_idx=column_num+1)
                optimal_lag_idx = np.argmax(correlations)
                optimal_lag = lags[optimal_lag_idx]
                optimal_lag_list.append(optimal_lag)
                max_correlation = correlations[optimal_lag_idx]
                if self.verbose:
                    print(f'Group {i}: Columns {column_num} <-> {column_num + 1}: Max correlation of '
                          f'{max_correlation} achieved at a lag of {optimal_lag}')
            # Shift columns according to optimal lag
            shifted_sample_change_group = torch.zeros_like(sample_change_horizontal_group)  # Set all to 0 pixel change
            # Last column stays the same
            cumulative_lag = 0
            shifted_sample_change_group[:, :, :, -1] = sample_change_horizontal_group[:, :, :, -1]
            for l2r_idx in range(len(optimal_lag_list)):  # shape[3] - 2 elements
                r2l_idx = (len(optimal_lag_list) - 1) - l2r_idx
                l_star = optimal_lag_list[r2l_idx]
                cumulative_lag += l_star
                # print(f'r to l idx: {r2l_idx}; l to r idx: {l2r_idx} optimal lag: {l_star}; cumulative_lag: {cumulative_lag}')
                if cumulative_lag > 0:  # Need to shift down, so we lose data at the end
                    shifted_sample_change_group[:, :, cumulative_lag:, r2l_idx] = sample_change_horizontal_group[:, :, :group_size-cumulative_lag, r2l_idx]
                    # print(f'Start idx: {start_idx}; end_idx = {end_idx}; '
                    #       f'change values: {sample_change_horizontal_group[:, :, :group_size-cumulative_lag, r2l_idx].shape}')
                else:  # Need to shift up
                    up_shift = np.abs(cumulative_lag)
                    shifted_sample_change_group[:, :, :group_size - up_shift, r2l_idx] = sample_change_horizontal_group[:, :, up_shift:, r2l_idx]
                    # print(f'Start idx: {start_idx}; end_idx = {end_idx}; '
                    #       f'change values: {sample_change_horizontal_group[:, :, up_shift:, r2l_idx].shape}')

            varve_counts_in_group, true_indices = self.varve_counts_by_color_threshold(
                shifted_sample_change_group, threshold=threshold, group=i, group_cols=True
            )
            varve_counts_in_group = varve_counts_in_group.squeeze()
            if self.verbose:
                print(f'Varve counts in group {i}: {varve_counts_in_group}')
            if group_cols:
                varve_count += float(varve_counts_in_group.item())
                varve_pixel_heights.extend(self.get_varve_pixel_heights(group_num=i, indices=true_indices))
            else:
                for ii in range(len(cumulative_varve_counts_per_col)):
                    cumulative_varve_counts_per_col[ii] += varve_counts_in_group[ii]

        if self.verbose:
            print(f"Columnwise counts above {threshold} threshold: {varve_count}")
        if group_cols:
            varve_count = [varve_count]
        return varve_count, varve_pixel_heights

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

    def compute_column_raw_cross_correlation(self, tensor, col1_idx=0, col2_idx=1, max_lag=None):
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

    def compute_column_cross_correlation(self, tensor, col1_idx=0, col2_idx=1, left_column_max_upshift=None,
                                         left_column_max_downshift=None):
        """
        Compute cross-correlation between first and second columns of a tensor at various lags.

        Args:
            tensor: Tensor of shape [1, 1, height, width] representing grayscale images
            left_column_max_upshift: Maximum lag to compute in negative direction.
                    If None, will compute for all possible negative lags
            left_column_max_downshift: Maximum lag to compute in positive direction.
                    If None, will compute for all possible positive lags

        Returns:
            lags: Array of lag values
            correlations: Array of correlation values for each lag
        """
        # Extract first two columns from the tensor
        # Squeeze out batch and channel dimensions
        col1 = tensor[0, 0, :, col1_idx]  # Shape: [100]
        col2 = tensor[0, 0, :, col2_idx]  # Shape: [100]

        # If max up/down shifts are not specified, compute for all possible lags
        if left_column_max_upshift is None:
            left_column_max_upshift = len(col1) - 1
        if left_column_max_downshift is None:
            left_column_max_downshift = len(col1) - 1

        # Initialize arrays to store results
        lags = range(-left_column_max_upshift, left_column_max_downshift + 1)
        correlations = []

        for lag in lags:
            if lag < 0:
                # Shift col1 up (or col2 down)
                col1_shifted = col1[-lag:]
                col2_shifted = col2[:len(col1_shifted)]
            elif lag > 0:
                # Shift col1 down (or col2 up)
                col1_shifted = col1[:-lag]
                col2_shifted = col2[lag:]
            else:
                # No shift
                col1_shifted = col1
                col2_shifted = col2

            # Compute Pearson correlation
            # TODO: alternatively use signal-processing cross-correlation
            if len(col1_shifted) > 1:  # Need at least 2 points for correlation
                # Convert to float and subtract mean
                x = col1_shifted.float() - col1_shifted.float().mean()
                y = col2_shifted.float() - col2_shifted.float().mean()

                # Compute correlation coefficient
                r = (x * y).sum() / (torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum()))
                correlations.append(r.item())
            else:
                correlations.append(0.0)

        return np.array(lags), np.array(correlations)

    def plot_cross_correlation(self, lags, correlations, col1_idx=None, col2_idx=None):
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
        if all([col is not None for col in [col1_idx, col2_idx]]):
            plt.title(f'Cross-correlation between columns {col1_idx} and {col2_idx}')
        else:
            plt.title('Cross-correlation between columns')
        plt.show()

    def transform_human_labels(self):
        for index, row in self.human_labels_df.iterrows():
            group_num, index_in_group = self.pixel_location_to_group_index(h=row['start_pixel_row'], w=row['pixel_col'])
            print(f'Human labeled varve found in Group {group_num} in index {index_in_group}')

    def pixel_location_to_group_index(self, h, w):
        if h < self.crop['top'] or h > self.image_shape[1] - self.crop['bottom']:
            new_h = None  # Cropped out
            group_num = None
            index_in_group = None
        else:
            new_h = math.floor((h - self.crop['top']) / self.vertical_kernel_size)
            group_num = math.floor(new_h / (self.correlation_group_pixel_height / self.vertical_kernel_size))
            index_in_group = math.floor((new_h % (self.correlation_group_pixel_height / self.vertical_kernel_size)) / self.vertical_or_aggregation_size)

        # if w < self.crop['left'] or w > self.image_shape[2] - self.crop['right']:
        #     new_w = None  # Cropped out
        # else:
        #     new_w = math.floor((w - self.crop['left']) / self.horizontal_kernel_size)

        return group_num, index_in_group

    def get_varve_pixel_heights(self, group_num, indices):
        pixel_heights = []
        for true_index in indices:
            pixel_height = (group_num * self.correlation_group_pixel_height +
                   self.vertical_kernel_size * self.vertical_or_aggregation_size * true_index) + self.crop['top']
            pixel_heights.append(pixel_height)
        return pixel_heights


if __name__ == "__main__":
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "example_configs",
        "example_config.json",
    )

    human_labels_csv = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "labeled_images",
        "varve_locations.csv"
    )

    av = AutoVarve(config_file=config_file, save_to_db=True, human_labels_csv=human_labels_csv)
    av.execute()
