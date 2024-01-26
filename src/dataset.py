import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image


class ImageSegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None, train=True):
        self.root_dir = root_dir
        self.train = train
        self.img_path = self.root_dir / "images"
        self.mask_path = self.root_dir / "masks"
        self.transforms = transforms
        # read images
        image_path_iter = self.img_path.glob("*.jpg")
        self.images = sorted(image_path_iter, key=lambda p: p.name)
        # read annotations
        mask_path_iter = self.mask_path.glob("*.png")
        # for root, dirs, files in os.walk(self.ann_dir):
        self.masks = sorted(mask_path_iter, key=lambda p: p.name)
        print(f"images count: {len(self.images)}")
        print(f"masks count: {len(self.masks)}")
        assert len(self.images) == len(
            self.masks
        ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image_path = self.file_paths[index]
        # image = Image.open(image_path).convert("RGB")
        # mask_path = image_path.replace("images", "masks").replace(".jpg", ".png")
        # mask = Image.open(mask_path).convert("L")

        # image:
        image = cv2.imread(str(self.images[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        # image = Image.open(self.images[index]).convert("RGB")
        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        # mask:
        # mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)
        mask = Image.open(self.masks[index]).convert("L")
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
        mask = (mask != 0).float()
        # mask = mask.unsqueeze(0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask


if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader

    # from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
    base_path = Path("/Users/taichi.muraki/workspace/Python/ring-finger-semseg")
    train_dataset = ImageSegmentationDataset(
        root_dir=base_path / "data/outputs/training/",
        transforms=None,
    )
