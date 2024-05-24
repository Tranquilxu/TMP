import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image
import random
from dataset_loader import read_data_cifar_100, read_data_tiny_imagenet_200, read_data_stanford_cars, \
    read_data_caltech_101, read_data_food_101
from torchvision.transforms import Normalize, Compose, Resize, ToTensor, RandomCrop, RandomHorizontalFlip
from ocra.utils import TransformTwice


def convert_to_rgb(image):
    return image.convert("RGB")


def ocra_train_transform(image_size):
    return Compose([
        convert_to_rgb,
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class CIFAR100_handler_train(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.transform = transform
        # print("self.YT", self.Y)
        with open('./datasets/CIFAR100/train_label_tf.txt', "r") as file:
            lines = file.readlines()
            self.YT = [int(line) for line in lines]
        with open('./datasets/CIFAR100/train_label_r.txt', "r") as file:
            lines = file.readlines()
            self.Y = [int(line) for line in lines]
        self.YT = torch.Tensor(self.YT)

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


class DatasetHandlerTrain(Dataset):
    def __init__(self, X, Y, dataset_name, num_classes, transform):
        self.X = X
        self.dataset_name = dataset_name
        self.class_num = num_classes
        self.transform = transform
        with open(f'./datasets/{dataset_name}/train_label_tf.txt', "r") as file:
            lines = file.readlines()
            self.YT = [int(line) for line in lines]
        with open(f'./datasets/{dataset_name}/train_label_r.txt', "r") as file:
            lines = file.readlines()
            self.Y = [int(line) for line in lines]
        self.YT = torch.Tensor(self.YT)

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


def divide_labeled_or_not(dataset, input_size):
    data_handler = get_data_handler(dataset, pattern='train', input_size=input_size)

    if dataset == 'food-101':
        np.random.seed(1)
        a = random.randint(0, data_handler.class_num - 1)
        b = random.randint(0, data_handler.class_num - 1)
        for i in range(len(data_handler.Y)):
            if data_handler.Y[i] == a or data_handler.Y[i] == b:
                data_handler.YT[i] = 0
    indices_yt_0 = torch.nonzero(torch.eq(data_handler.YT, 0)).squeeze().tolist()
    indices_yt_1 = torch.nonzero(torch.eq(data_handler.YT, 1)).squeeze().tolist()
    unlabeled_dataset = Subset(data_handler, indices_yt_0)
    labeled_dataset = Subset(data_handler, indices_yt_1)
    return labeled_dataset, unlabeled_dataset


def get_data_handler(dataset, pattern, input_size):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        if pattern == "train":
            datahandler = CIFAR100_handler_train(train_data, train_label,
                                                 transform=TransformTwice(ocra_train_transform(image_size=input_size)))

    else:
        if dataset == 'tiny-imagenet-200':
            train_data, train_label, test_data, test_label, num_classes = read_data_tiny_imagenet_200()
        elif dataset == 'stanford_cars':
            train_data, train_label, test_data, test_label, num_classes = read_data_stanford_cars()
        elif dataset == 'caltech-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_caltech_101()
        elif dataset == 'food-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_food_101()

        if pattern == "train":
            datahandler = DatasetHandlerTrain(train_data, train_label, dataset_name=dataset, num_classes=num_classes,
                                              transform=TransformTwice(ocra_train_transform(image_size=input_size)))

    return datahandler
