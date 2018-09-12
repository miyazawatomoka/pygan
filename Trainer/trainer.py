import torch
import torchvision
from torchvision import transforms
from config import Config
from torch.utils.data.dataloader import DataLoader


class Trainer:
    def __init__(self):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Scale(Config.image_size),
            torchvision.transforms.CenterCrop(Config.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

        self.dataset = torchvision.datasets.ImageFolder(Config.data_path, transform=self.transforms)
        self.dataloader = DataLoader (self.dataset,
                                      batch_size = Config.batch_size,
                                      shuffle=True,
                                      num_workers = Config.num_workers,
                                      drop_last = True)
    def train(self):
        pass