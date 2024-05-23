from typing import Dict, Tuple
from torch.utils.data import DataLoader
from dataset_loader import get_transform, read_data_cifar_100, read_data_eruosat, read_data_tiny_imagenet_200, \
    read_data_stanford_cars, read_data_caltech_101, read_data_food_101, CIFAR100_handler_test, DatasetHandlerTest


def get_data_handler(dataset, pattern, input_size):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        if pattern == "train":
            datahandler = CIFAR100_handler_test(train_data, train_label, input_size)
        elif pattern == "val":
            datahandler = CIFAR100_handler_test(test_data, test_label, input_size)
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
            datahandler = DatasetHandlerTest(train_data, train_label, transform=get_transform(input_size))
        elif pattern == "val":
            datahandler = DatasetHandlerTest(test_data, test_label, get_transform(input_size))

    return datahandlers


def load_datasets(
        dataset: str,
        model_type: str,
        pattern: str,
        input_size: int,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = "./datasets/" + dataset

    tag_file = dataset_root + f"/{dataset}.txt"


    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]


    taglist = taglist_or  
    datahandler = get_data_handler(dataset, pattern, input_size)
    loader = DataLoader(dataset=datahandler, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    info = {
        "taglist": taglist
    }

    return loader, info
