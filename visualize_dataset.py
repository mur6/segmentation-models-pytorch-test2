import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from src.dataset import ImageSegmentationDataset


def main():
    base_path = Path("/Users/taichi.muraki/workspace/Python/ring-finger-semseg")
    train_dataset = ImageSegmentationDataset(
        root_dir=base_path / "data/outputs/training/",
        transforms=None,
    )
    # visualize first data, first image and first mask, using matplotlib
    img, mask = train_dataset[3]
    print(f"img.shape: {img.shape} dtype: {img.dtype}")
    print(f"img min: {img.min()} max: {img.max()}")
    print(f"mask.shape: {mask.shape} dtype: {mask.dtype}")
    print(f"mask min: {mask.min()} max: {mask.max()}")
    # print(img.shape)
    # print(mask.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0) / 255.0)
    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze())
    plt.show()


if __name__ == "__main__":
    #     from torch.utils.data import DataLoader

    # from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
    main()
