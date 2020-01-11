#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms


def load_dataset(
        num_train_batch, num_test_batch, num_extra_batch=0, 
        num_worker=8, dataset='mnist'):
    """Load image dataset which provided by PyTorch.
    Prepare the loaded dataset into the train_loader, (extra_loader) and test_loader.
    :param num_train_batch: An integer for assigning amount of training batch.
    :param num_test_batch: An integer for assigning amount of testing batch.
    :param num_extra_batch: An integer for assigning amount of extra batch (if exists).
    :param num_worker: A integer for assigning num_worker.
    :param dataset: A string in set of {mnist, fashion_mnist, kmnist, emnist, cifar10, cifar100}.
    :return img_size: A list of number of pixel in each dimension of a image.
            Using np.prod(img_size) to get the input_neuron for MLPs.
    :returns train_loader, extra_loader, test_loader:
    """
    assert isinstance(num_train_batch, int)
    assert isinstance(num_test_batch, int)
    assert isinstance(num_extra_batch, int)
    assert isinstance(num_worker, int)
    assert isinstance(dataset, str)

    data_locat = ''
    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(
            root=data_locat, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.MNIST(
            root=data_locat, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        img_size = [28, 28, 1]
    elif dataset == 'fashion_mnist':
        train_set = torchvision.datasets.FashionMNIST(
            root=data_locat, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_locat, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        img_size = [28, 28, 1]
    elif dataset == 'kmnist':
        train_set = torchvision.datasets.KMNIST(
            root=data_locat, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_locat, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        img_size = [28, 28, 1]
    elif dataset == 'emnist':
        train_set = torchvision.datasets.EMNIST(
            root=data_locat, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_locat, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        img_size = [28, 28, 1]
    elif dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_locat, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.CIFAR10(
            root=data_locat, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        img_size = [32, 32, 3]
    elif dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=data_locat, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.CIFAR100(
            root=data_locat, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        img_size = [32, 32, 3]
    elif dataset == 'svhn':
        # The extra-section or extra_set is exist in this dataset.
        train_set = torchvision.datasets.SVHN(
            root=data_locat, split='train', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=num_train_batch, shuffle=True, num_workers=num_worker)
        test_set = torchvision.datasets.SVHN(
            root=data_locat, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_batch, shuffle=False, num_workers=num_worker)
        extra_set = torchvision.datasets.SVHN(
            root=data_locat, split='extra', download=True, transform=transform)
        extra_loader = torch.utils.data.DataLoader(
            extra_set, batch_size=num_extra_batch, shuffle=False, num_workers=num_worker)
        img_size = [32, 32, 3]
        return train_loader, test_loader, extra_loader, img_size
    else:
        raise NotImplementedError(
            f'dataset must be in range [mnist, fashion_mnist, kmnist, '
            + f'emnist, cifar10, cifar100, svhn] only, your input: {dataset}')
    return train_loader, test_loader, img_size

