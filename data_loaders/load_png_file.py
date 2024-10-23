import torch
import torchvision
from torchvision.io import read_image, ImageReadMode, decode_image
import torchvision.transforms as T
import os



if __name__ == '__main__':
    autovarve_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(autovarve_directory, 'data', 'D15-4Lspliced_no ruler.png')
    img_tensor = decode_image(image_path, mode='RGB')

    print(f'Loaded image with shape: {img_tensor.shape}')
    print(f'Top left pixel: {img_tensor[:, 0, 0]}')

