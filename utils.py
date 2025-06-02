import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from PIL import Image
import os
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, CelebA
from datasets.lsun import LSUN
import types
from diffusers.models.resnet import Upsample2D, Downsample2D
import torch_pruning as tp


class UnlabeledImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, exts=["*.jpg", "*.png", "*.jpeg", "*.webp"]):
        self.root = root
        self.files = []
        self.transform = transform
        for ext in exts:
            self.files.extend(glob(os.path.join(root, '**/*.{}'.format(ext)), recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def set_dropout(model, p):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p


def get_dataset(name_or_path, transform=None, config=None):
    if name_or_path.lower()=='cifar10':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif name_or_path.lower()=='cifar100':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif "lsun" in name_or_path.lower():
        train_folder = "{}_train".format(config.data.category)
        if transform is None:
            if config.data.random_flip:
                transform = T.Compose(
                    [
                        T.Resize(config.data.image_size),
                        T.CenterCrop(config.data.image_size),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),
                        T.Normalize(mean=0.5, std=0.5),
                    ]
                )
            else:
                transform = T.Compose(
                    [
                        T.Resize(config.data.image_size),
                        T.CenterCrop(config.data.image_size),
                        T.ToTensor(),
                        T.Normalize(mean=0.5, std=0.5),
                    ]
                )
        dataset = LSUN(
                root=os.path.join("../../dataset/", "lsun"),
                classes=[train_folder],
                transform=transform,)

    elif name_or_path.lower() == "celeba":
        if transform is None:
            if config.data.random_flip:
                transform = T.Compose(
                    [
                        T.Resize(config.data.image_size),
                        T.CenterCrop(config.data.image_size),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),
                        T.Normalize(mean=0.5, std=0.5),
                    ]
                )
            else:
                transform = T.Compose(
                    [
                        T.Resize(config.data.image_size),
                        T.CenterCrop(config.data.image_size),
                        T.ToTensor(),
                        T.Normalize(mean=0.5, std=0.5),
                    ]
                )
        dataset = CelebA(root='./data/', split='train', download=True, transform=transform)

    elif os.path.isdir(name_or_path):
        if transform is None:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(256),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = UnlabeledImageFolder(name_or_path, transform=transform)

    return dataset


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
