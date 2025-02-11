# Objective is to output a CSV file with 4 columns: start_pixel_row, end_pixel_row, pixel_col, varve_num
# The start_pixel_row column indicates the start of the thin-grained, dark-colored layer of the varve
# The end_pixel_row column indicates the end of the thin-grained, dark-colored layer of the varve
# The pixel_col column indicates the column where the human labeler was looking for the thin-grained, dark-colored layer
# The varve num is the order of the varve, starting from the top of the input image (the most recent)

import torch
from AutoVarve import AutoVarve
import os
import sys
import pandas as pd


def identify_color_occurrences(tensor, r_min=235, g_max=86, b_min=247):
    """
    Identifies occurrences where R >= r_min, G <= g_max, and B >= b_min simultaneously in a [3, N] tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [3, N]
        r_val (int): Red channel value to match
        g_val (int): Green channel value to match
        b_val (int): Blue channel value to match

    Returns:
        int: Number of positions where all values match simultaneously
    """
    # Create masks for each channel
    red_mask = r_min <= tensor[0]
    green_mask = tensor[1] <= g_max
    blue_mask = b_min <= tensor[2]

    # Combine masks with logical AND
    combined_mask = red_mask & green_mask & blue_mask

    return combined_mask


def output_csv(indices_tensor, output_filepath, pixel_col=2489):
    """
    Create CSV with
    :param indices_tensor:
    :return:
    """
    thingrained_layer_count = 0
    start_of_row_index = None
    last_index = None
    list_of_lists = []
    row_list = [None] * 4
    for i, val in enumerate(indices_tensor):
        index_int = val.item()
        if last_index is None:  # First group
            start_of_row_index = index_int
            thingrained_layer_count += 1
        elif index_int == last_index + 1:  # Same group
            pass
        else:  # New group
            row_list[0] = start_of_row_index
            row_list[1] = last_index
            row_list[2] = pixel_col
            row_list[3] = thingrained_layer_count

            list_of_lists.append(row_list)
            row_list = [None] * 4

            start_of_row_index = index_int
            thingrained_layer_count += 1

        if i + 1 == indices_tensor.size:  # Last group
            row_list[0] = start_of_row_index
            row_list[1] = index_int
            row_list[2] = pixel_col
            row_list[3] = thingrained_layer_count

            list_of_lists.append(row_list)
            row_list = [None] * 4
        last_index = index_int

    human_labels_df = pd.DataFrame(list_of_lists, columns=['start_pixel_row', 'end_pixel_row', 'pixel_col', 'varve_num'])

    human_labels_df.to_csv(output_filepath, index=False)

    return human_labels_df


if __name__ == "__main__":
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "example_configs",
        "extract_human_labels.json",
    )
    image_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "red_line_images")
    av = AutoVarve(config_file=config_file, save_to_db=False, image_directory=image_directory)
    image_tensors = av.load_images()

    print(f'Image tensors have shape: {image_tensors.shape}')

    pixel_column = 700
    max_row = 50600
    r = 220  # min
    g = 200  # max
    b = 220  # min

    column = image_tensors[0, :, :max_row, pixel_column]

    combined_mask = identify_color_occurrences(column, r_min=r, g_max=g, b_min=b)

    counts = torch.sum(combined_mask)
    indices = torch.nonzero(combined_mask).squeeze()

    human_labels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'labeled_images', 'human_labels.csv')
    human_labels_df = output_csv(indices_tensor=indices, output_filepath=human_labels_path)

    print(f'Found {human_labels_df.shape[0]} human labels for fine-grained, dark-colored varve lines at positions:'
          f'\n{human_labels_df.loc[:, "start_pixel_row"].to_list()}')







