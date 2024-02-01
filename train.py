from pathlib import Path
import os

import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader

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
    config_defaults = {
        "learning_rate": 0.0001,
        "architecture": "Unet",
        "encoder_name": "resnet34",
        "batch_size": 16,
    }
    with wandb.init(
        config=config_defaults, project="segmentation_models_pytorch2"
    ) as run:
        config = wandb.config
        wandb_logger = WandbLogger(log_model="all")
        epochs = 10
        batch_size = config.batch_size
        # -----------------------------------
        # wandb_logger.experiment.config.update(
        #     {
        #         "learning_rate": learning_rate,
        #         "architecture": architecture,
        #         "encoder_name": encoder_name,
        #         "epochs": epochs,
        #     }
        # )
        # -----------------------------------
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
            config,
            in_channels=3,
            out_classes=1,
        )
        # Log gradients, parameters and model topology
        wandb_logger.watch(model, log="all")
        checkpoint_callback = ModelCheckpoint(monitor="valid_accuracy", mode="max")
        # Training
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )


def do_wandb_sweep():
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "valid_accuracy"},
        "parameters": {
            "learning_rate": {"values": [0.001, 0.0001]},
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
            "encoder_name": {"values": ["resnet34", "resnet101"]},
            "batch_size": {"values": [16, 32]},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="segmentation_models_pytorch-sweep_search"
    )
    wandb.agent(sweep_id, function=train_main, count=30)


def main():
    # do_wandb_sweep()
    train_main()


if "__main__" == __name__:
    main()
