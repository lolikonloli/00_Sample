import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import VisionDataset
from torch.optim import Optimizer, lr_scheduler


class Runner():

    def __init__(
        self,
        config: dict,
        train_dataloader: VisionDataset or None,
        val_dataloader: VisionDataset or None,
        test_dataloader: VisionDataset or None,
        model : Module,
        optimizer: Optimizer,
        scheduler:lr_scheduler
    ) -> None:
        pass

    def train():
        pass

    def val():
        pass

    def test():
        pass