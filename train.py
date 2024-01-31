from pathlib import Path
import os

import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

from src.model import PetModel
from src.dataset import ImageSegmentationDataset


# lets look at some samples
def show_samples(train_dataset, valid_dataset, test_dataset):
    sample = train_dataset[0]
    plt.subplot(1, 2, 1)
    plt.imshow(
        sample["image"].transpose(1, 2, 0)
    )  # for visualization we have to transpose back to HWC
    plt.subplot(1, 2, 2)
    plt.imshow(
        sample["mask"].squeeze()
    )  # for visualization we have to remove 3rd dimension of mask
    plt.show()

    sample = valid_dataset[0]
    plt.subplot(1, 2, 1)
    plt.imshow(
        sample["image"].transpose(1, 2, 0)
    )  # for visualization we have to transpose back to HWC
    plt.subplot(1, 2, 2)
    plt.imshow(
        sample["mask"].squeeze()
    )  # for visualization we have to remove 3rd dimension of mask
    plt.show()

    sample = test_dataset[0]
    plt.subplot(1, 2, 1)
    plt.imshow(
        sample["image"].transpose(1, 2, 0)
    )  # for visualization we have to transpose back to HWC
    plt.subplot(1, 2, 2)
    plt.imshow(
        sample["mask"].squeeze()
    )  # for visualization we have to remove 3rd dimension of mask
    plt.show()


def train_main():
    wandb_logger = WandbLogger(project="segmentation_models_pytorch", log_model="all")
    learning_rate = 0.0001
    architecture = "FPN"
    encoder_name = "resnet34"
    epochs = 10
    batch_size = 16
    wandb_logger.experiment.config.update(
        {
            "learning_rate": learning_rate,
            "architecture": architecture,
            "encoder_name": encoder_name,
            "epochs": epochs,
        }
    )
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="segmentation_models_pytorch",
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": learning_rate,
    #         "architecture": architecture,
    #         "encoder_name": encoder_name,
    #         "epochs": epochs,
    #     },
    # )

    # base_path = Path("/Users/taichi.muraki/workspace/Python/ring-finger-semseg/data/")
    base_path = Path("../ring-finger-semseg/")
    train_dataset = ImageSegmentationDataset(
        root_dir=base_path / "outputs/training/",
        transforms=None,
    )
    valid_dataset = ImageSegmentationDataset(
        root_dir=base_path / "outputs/validation/",
        transforms=None,
        train=False,
    )
    # # init train, val, test sets
    # train_dataset = SimpleOxfordPetDataset(root, "train")
    # valid_dataset = SimpleOxfordPetDataset(root, "valid")
    # test_dataset = SimpleOxfordPetDataset(root, "test")

    # # It is a good practice to check datasets don`t intersects with each other
    # assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    # assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    # assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    n_cpu = os.cpu_count()
    print(f"Number of CPUs: {n_cpu}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu
    # )

    # show_samples(train_dataset, valid_dataset, test_dataset)
    model = PetModel(
        architecture,
        encoder_name,
        in_channels=3,
        out_classes=1,
        learning_rate=learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(monitor="valid_accuracy", mode="max")
    # Training
    trainer = pl.Trainer(
        gpus=1, max_epochs=epochs, logger=wandb_logger, callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    # wandb.finish()


def do_wandb_sweep():
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "valid_accuracy"},
        "parameters": {
            # "learning_rate": {"max": 0.002, "min": 0.0001},
            "architecture": {
                "values": [
                    "Unet",
                    "UnetPlusPlus",
                    # MAnet,
                    "Linknet",
                    "FPN",
                    "PSPNet",
                    "DeepLabV3",
                    "DeepLabV3Plus",
                    "PAN",
                ]
            },
            # "batch_size": {"values": [16, 32]},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="segmentation_models_pytorch-sweep_search"
    )
    wandb.agent(sweep_id, function=train_main, count=40)


# call main
if "__main__" == __name__:
    do_wandb_sweep()
