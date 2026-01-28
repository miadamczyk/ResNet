from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_cifar_loaders(
    dataset_name='CIFAR100',
    batch_size=128,
    num_workers=2,
    train_augment=True,
    val_split=0.1,
):
    assert dataset_name in ['CIFAR10', 'CIFAR100']

    if dataset_name == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
        DatasetClass = datasets.CIFAR10
    else:  # CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2762)
        DatasetClass = datasets.CIFAR100

    if train_augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_dataset = DatasetClass(
        root='./data',
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = DatasetClass(
        root='./data',
        train=False,
        download=True,
        transform=test_transform,
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
