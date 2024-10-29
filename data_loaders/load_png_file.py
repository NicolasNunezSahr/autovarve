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


def process_and_plot_image_distribution(tensor):
    """
    Process an image tensor using average pooling.

    Args:
        tensor: PyTorch tensor of shape [channels, height, width]
    """
    # Ensure input is the correct shape
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension if not present

    # Apply average pooling
    # Padding='same' to ensure output size is ceil(input_size/kernel_size)
    pooled = F.avg_pool2d(tensor, kernel_size=4, stride=4, padding=0)

    # Flatten the tensor for distribution analysis
    flattened_values = pooled.flatten().cpu().numpy()

    return pooled, flattened_values


def average_middle_rows(tensor, edge_pixels=1000):
    """
    Average the middle section of each row in the image, excluding edge pixels.

    Args:
        tensor: PyTorch tensor of shape [3, height, width] with values 0-1
        edge_pixels: Number of pixels to exclude from each edge (default: 1000)

    Returns:
        Tensor of shape [3, height, 1] containing row averages
    """

    # Calculate the middle section for each row
    middle_section = tensor[:, :, edge_pixels:-edge_pixels]

    # Average along the row dimension (dim=2)
    row_averages = middle_section.mean(dim=2, keepdim=True)

    return row_averages


def plot_row_averages_distribution(row_averages, save_path='row_averages_distribution.png', dpi=300,
                                   channel_names=['Red', 'Green', 'Blue'],
                                   colors = ['red', 'green', 'blue']):
    """
    Plot the distribution of row averages for each channel.

    Args:
        row_averages: Tensor of shape [3, height, 1] with values between 0-1
        save_path: Path where to save the plot
        dpi: Resolution of the saved image
    """
    plt.figure(figsize=(12, 7))

    for i, (color, name) in enumerate(zip(colors, channel_names)):
        # Get values for this channel
        channel_values = row_averages[i].flatten().cpu().numpy()

        # Plot distribution
        sns.kdeplot(data=channel_values, color=color, fill=True, alpha=0.3,
                    label=f'{name} Channel')

        # Print statistics
        print(f"{name} Channel Row Averages - Min: {channel_values.min():.3f}, "
              f"Max: {channel_values.max():.3f}, "
              f"Mean: {channel_values.mean():.3f}")

    plt.title('Distribution of Row Averages by RGB Channel (Middle 2000 pixels)')
    plt.xlabel('Average Pixel Value (Normalized)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the plot
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    # Display the plot
    plt.show(block=True)


def average_row_groups(tensor, group_size=8):
    """
    Average groups of rows in a grayscale tensor.

    Args:
        tensor: PyTorch tensor of shape [1, height, 1]
        group_size: Number of rows to average together (default: 8)

    Returns:
        Tensor of shape [1, height//group_size, 1] containing group averages
    """
    # Get original dimensions
    channels, height, width = tensor.shape

    # Reshape to group the rows
    # New shape: [1, height//group_size, group_size, 1]
    reshaped = tensor.reshape(channels, height // group_size, group_size, width)

    # Average along the group dimension (dim=2)
    group_averages = reshaped.mean(dim=2)

    return group_averages


def plot_group_averages_distribution(group_averages, save_path='group_averages_distribution.png', dpi=300):
    """
    Plot the distribution of group averages.

    Args:
        group_averages: Tensor of shape [1, height//8, 1]
        save_path: Path where to save the plot
        dpi: Resolution of the saved image
    """
    plt.figure(figsize=(10, 6))

    # Get values and create distribution plot
    values = group_averages.flatten().cpu().numpy()

    sns.kdeplot(data=values, fill=True, color='gray')

    # Print statistics
    print(f"Group Averages - Min: {values.min():.3f}, "
          f"Max: {values.max():.3f}, "
          f"Mean: {values.mean():.3f}")

    plt.title('Distribution of 8-Row Group Averages')
    plt.xlabel('Average Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    # Display the plot
    plt.show(block=True)

    return values


def save_averages_to_file(group_averages, output_path='group_averages.txt'):
    """
    Save group averages to a text file.

    Args:
        group_averages: Tensor of shape [1, height//8, 1]
        output_path: Path where to save the text file
    """
    # Convert to numpy and flatten
    values = group_averages.squeeze().cpu().numpy()

    # Save to file with row numbers
    with open(output_path, 'w') as f:
        f.write("Row_Group\tAverage\n")  # Header
        for i, value in enumerate(values):
            f.write(f"{i}\t{value:.6f}\n")

    print(f"Averages saved to: {output_path}")

    # Also save as numpy array for easier loading later if needed
    np_output_path = output_path.rsplit('.', 1)[0] + '.npy'
    np.save(np_output_path, values)
    print(f"Numpy array saved to: {np_output_path}")


def process_image_data(tensor, vertical_group_size=8, horizontal_group_size=200):
    """
    Process the image tensor through multiple averaging steps.

    Args:
        tensor: PyTorch tensor of shape [1, 65760, 2000]
        vertical_group_size: Number of rows to average together vertically
        horizontal_group_size: Number of pixels to average horizontally

    Returns:
        Tensor of shape [1, 99, 10] containing the changes between consecutive rows
    """
    batch, height, width = tensor.shape

    # Step 1: Average 8 pixels vertically
    reshaped = tensor.reshape(batch, height // vertical_group_size, vertical_group_size, width)
    vertically_averaged = reshaped.mean(dim=2)  # Shape: [1, 8220, 2000]

    # Step 2: Take median of groups of 200 pixels horizontally
    horizontally_medianed = []
    for i in range(10):
        start_idx = i * horizontal_group_size
        end_idx = (i + 1) * horizontal_group_size
        # Get median of each group
        group_median = torch.median(vertically_averaged[:, :, start_idx:end_idx], dim=2).values
        horizontally_medianed.append(group_median.unsqueeze(2))

    horizontally_medianed = torch.cat(horizontally_medianed, dim=2)  # Shape: [1, 8220, 10]

    # Step 3: Get first 100 rows
    first_100 = horizontally_medianed[:, :100, :]  # Shape: [1, 100, 10]

    # Step 4: Calculate changes between consecutive rows
    changes = first_100[:, 1:, :] - first_100[:, :-1, :]  # Shape: [1, 99, 10]

    return changes


def plot_last_three_columns(tensor, save_path='row_changes.png', dpi=300, color='blue'):
    """
    Create a line plot showing value changes between consecutive rows for the last 3 columns,
    using a gradient of colors.

    Args:
        tensor: Processed tensor of shape [1, 99, 10] containing row-to-row changes
        save_path: Path to save the plot
        dpi: Resolution of the saved image
        color: Base color for the gradient ('blue', 'red', or 'green')
    """
    # Convert to numpy and squeeze batch dimension
    data = tensor.squeeze().cpu().numpy()  # Shape: [99, 10]

    # Select only the last 3 columns
    data = data[:, 7:]  # Shape: [99, 3]

    # Create row numbers for x-axis
    row_numbers = np.arange(1, 100)  # 99 changes between 100 rows

    # Create color gradient for 3 colors
    if color == 'blue':
        color_values = [(0.6, 0.6, 1), (0, 0, 0.5)]  # Light blue to dark blue
    elif color == 'red':
        color_values = [(1, 0.6, 0.6), (0.5, 0, 0)]  # Light red to dark red
    elif color == 'green':
        color_values = [(0.6, 1, 0.6), (0, 0.5, 0)]  # Light green to dark green

    cmap = LinearSegmentedColormap.from_list('custom', color_values)
    colors = [cmap(i / 2) for i in range(3)]  # 3 colors from light to dark

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot each of the last 3 columns with gradient colors
    for i in range(3):
        plt.plot(row_numbers, data[:, i],
                 label=f'Column {i + 8}',  # Adjusted labels to show actual column numbers (8-10)
                 linewidth=2,
                 marker='o',
                 markersize=4,
                 color=colors[i])

    plt.title('Value Changes Between Consecutive Rows (Columns 8-10)')
    plt.xlabel('Gap Between Rows (e.g., 1 means change from row 1 to 2)')
    plt.ylabel('Change in Value')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add horizontal line at y=0 to show positive/negative changes
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.06, color='red', linestyle='--', alpha=0.7, label='Threshold (0.06)')

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    # Display the plot
    plt.show(block=True)

    # Print some statistics about the changes
    print("\nChange statistics for last 3 columns:")
    for i in range(3):
        col_data = data[:, i]
        print(f"\nColumn {i + 8}:")  # Adjusted to show actual column numbers
        print(f"  Mean change: {col_data.mean():.6f}")
        print(f"  Std of changes:  {col_data.std():.6f}")
        print(f"  Min change:  {col_data.min():.6f}")
        print(f"  Max change:  {col_data.max():.6f}")


if __name__ == '__main__':
    autovarve_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visualizations_path = os.path.join(autovarve_directory, 'visualizations')
    image_path = os.path.join(autovarve_directory, 'data', 'D15-4Lspliced_no ruler.png')
    mode = None
    if mode is None:
        img_tensor = decode_image(image_path)
    else:
        img_tensor = decode_image(image_path, mode=mode)

    print(f'Loaded image with shape: {img_tensor.shape}')

    # Convert Tensor to Float
    img_tensor = img_tensor.float() / 255.0

    # pooled, flattened_values = process_and_plot_image_distribution(tensor=img_tensor)
    #
    # # Create distribution plot
    # plt.figure(figsize=(10, 6))
    #
    # if mode == 'RGB':
    #     # Plot distribution for each channel
    #     colors = ['red', 'green', 'blue']
    #     channel_names = ['Red', 'Green', 'Blue']
    #
    #     for i, (color, name) in enumerate(zip(colors, channel_names)):
    #         # Get values for this channel
    #         channel_values = pooled[0, i].flatten().cpu().numpy()
    #
    #         # Plot distribution
    #         sns.kdeplot(data=channel_values, color=color, fill=True, alpha=0.3,
    #                     label=f'{name} Channel')
    #
    #         # Print statistics
    #         print(f"{name} Channel - Min: {channel_values.min():.3f}, "
    #               f"Max: {channel_values.max():.3f}, "
    #               f"Mean: {channel_values.mean():.3f}")
    #
    #     plt.title('Distribution of RGB Channel Values (Normalized 0-1)')
    #     plt.xlabel('Pixel Value (Normalized)')
    #     plt.ylabel('Density')
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #
    #     save_path = os.path.join(visualizations_path, 'rgb_distribution.png')
    #
    # elif mode is None:
    #     sns.kdeplot(data=flattened_values, fill=True)
    #     plt.title('Distribution of Averaged Pixel Values')
    #     plt.xlabel('Pixel Value')
    #     plt.ylabel('Density')
    #     plt.grid(True, alpha=0.3)
    #
    #     save_path = os.path.join(visualizations_path, 'grayscale_distribution.png')
    # else:
    #     raise Exception('Nope')
    #
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #
    # print(f"\nPlot saved to: {save_path}")
    #
    # plt.show()

    edge_pixels = 1000
    middle_section = img_tensor[:, :, edge_pixels:-edge_pixels]

    changes_tensor = process_image_data(middle_section)
    save_path = os.path.join(autovarve_directory, 'visualizations', 'last_3_peaks_median.png')
    plot_last_three_columns(changes_tensor, save_path=save_path)

    # row_averages = average_middle_rows(img_tensor, edge_pixels=1000)
    # print(f'Row averages: {row_averages.shape}')
    #
    # group_averages = average_row_groups(row_averages, group_size=8)
    # print(f'Group averages: {group_averages.shape}')
    #
    # save_averages_to_file(group_averages, os.path.join(autovarve_directory, 'data', 'group_averages.txt'))
    #
    # plot_group_averages_distribution(group_averages, save_path=os.path.join(visualizations_path, 'group_averages.png'))

    # save_path = os.path.join(visualizations_path, 'gray_row_averages_distribution.png')
    # plot_row_averages_distribution(row_averages, save_path, channel_names=['Grayscale'], colors=['gray'])