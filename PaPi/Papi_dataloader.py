import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from dataset_loader import read_data_cifar_100, read_data_tiny_imagenet_200, read_data_stanford_cars, \
    read_data_caltech_101, read_data_food_101, load_datasets
from PaPi.utils import Cutout
from PaPi.autoaugment import CIFAR10Policy, ImageNetPolicy


def convert_to_rgb(image):
    return image.convert("RGB")


class CIFAR100Partialize(Dataset):
    def __init__(self, X, Y, num_classes):
        self.X = X
        self.Y = Y
        self.given_partial_label_matrix = torch.zeros(len(Y), num_classes)
        torch.manual_seed(1)
        np.random.seed(1)
        for i in range(len(self.X)):
            r = random.randint(0, num_classes - 1)
            if self.Y[i] == r:
                self.given_partial_label_matrix[i][r] = 1.0
            else:
                self.given_partial_label_matrix[i][:] = 1.0
                self.given_partial_label_matrix[i][r] = 0.0

        self.transform1 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform2 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        each_image1 = self.transform1(x)
        each_image2 = self.transform2(x)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = torch.Tensor(self.Y)[index]
        # each_true_label = self.Y[index]

        return each_image1, each_image2, each_label, each_true_label.float(), index


class DatasetPartialize(Dataset):
    def __init__(self, X, Y, num_classes):
        self.X = X
        self.Y = Y
        self.given_partial_label_matrix = torch.zeros(len(Y), num_classes)
        torch.manual_seed(1)
        np.random.seed(1)
        for i in range(len(self.X)):
            r = random.randint(0, num_classes - 1)
            if self.Y[i] == r:
                self.given_partial_label_matrix[i][r] = 1.0
            else:
                self.given_partial_label_matrix[i][:] = 1.0
                self.given_partial_label_matrix[i][r] = 0.0

        self.transform1 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform2 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        each_image1 = self.transform1(x)
        each_image2 = self.transform2(x)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = torch.Tensor(self.Y)[index]

        return each_image1, each_image2, each_label, each_true_label.float(), index


def get_data_handler(dataset):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        datahandler = CIFAR100Partialize(train_data, train_label, num_classes=100)
    else:
        if dataset == 'tiny-imagenet-200':
            train_data, train_label, test_data, test_label, num_classes = read_data_tiny_imagenet_200()
        elif dataset == 'stanford_cars':
            train_data, train_label, test_data, test_label, num_classes = read_data_stanford_cars()
        elif dataset == 'caltech-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_caltech_101()
        elif dataset == 'food-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_food_101()

        datahandler = DatasetPartialize(train_data, train_label, num_classes=num_classes)
    return datahandler


def load_data(args):
    test_loader, _ = load_datasets(
        dataset=args.dataset,
        model_type='clip',
        pattern="val",
        input_size=224,
        batch_size=args.batch_size * 4,
        num_workers=args.num_workers
    )

    partial_training_dataset = get_data_handler(args.dataset)
    partialY_matrix = partial_training_dataset.given_partial_label_matrix

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    return partial_training_dataloader, partialY_matrix, test_loader
