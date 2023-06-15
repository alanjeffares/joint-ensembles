import torchvision
import torchvision.transforms as T
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import json
import os
from PIL import Image
from torch.utils.data import Dataset


# get path to data folder from config
with open('src/configs/paths.json') as json_file:
    paths = json.load(json_file)

# get data info from config
with open('src/configs/data.json') as json_file:
    data_config = json.load(json_file)

# get current ImageNet base learner from config
with open('src/configs/imagenet/experiment.json') as json_file:
    exp_config = json.load(json_file)
    BASE_LEARNER = exp_config['base_learner']

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_CIFAR_dataloaders(version: str, batch_size: int, seed: int):
    """CIFAR dataloaders with train, val, test sets.
    
    Args:
        version: (str) CIFAR version, either '10' or '100'
        batch_size: (int) batch size
        seed: (int) random seed

    Returns:
        train_loader: (DataLoader) train set
        val_loader: (DataLoader) validation set
        test_loader: (DataLoader) test set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    normalize = T.Normalize(
            mean=np.array([125.3, 123.0, 113.9]) / 255.0,
            std=np.array([63.0, 62.1, 66.7]) / 255.0,
        )
    training_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            normalize,
        ]
    )
    testing_transform = T.Compose([T.ToTensor(),
                                normalize,
                                ])
    
    CIFAR = getattr(torchvision.datasets, f'CIFAR{version}')
    train_ds = CIFAR(paths['data'], train=True, download=True, transform=training_transform)
    val_ds = CIFAR(paths['data'], train=True, download=True, transform=testing_transform)
    test_ds = CIFAR(paths['data'], train=False, download=True, transform=testing_transform)
    
    # Generate validation set - https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    val_size = 0.2
    num_train = len(train_ds)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    train_size = len(train_idx)
    val_size = len(valid_idx)
    test_size = len(test_ds)
    print(f'Split sizes: train: {train_size}, val: {val_size}, test: {test_size}')

    train_loader = DataLoader(train_ds,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              worker_init_fn=seed_worker,
                              num_workers=0,
                              generator=g)
    val_loader = DataLoader(val_ds,
                            sampler=val_sampler,
                            batch_size=batch_size,
                            worker_init_fn=seed_worker,
                            generator=g)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             worker_init_fn=seed_worker,
                             num_workers=0,
                             generator=g)
    train_loader.split_size = train_size
    val_loader.split_size = val_size
    test_loader.split_size = test_size

    return train_loader, val_loader, test_loader

def get_SVHN_dataloaders(batch_size: int, seed: int):
    """SVHN dataloaders with train, val, test sets.

    Args:
        batch_size: (int) batch size
        seed: (int) random seed

    Returns:
        train_loader: (DataLoader) train set
        val_loader: (DataLoader) validation set
        test_loader: (DataLoader) test set
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    normalize = T.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )

    training_transform = T.Compose([
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         normalize
        ])
    testing_transform = T.Compose([T.ToTensor(),
                                   normalize
                                   ])

    train_ds = torchvision.datasets.SVHN(paths['data'], split='train', download=True,
                                  transform=training_transform)

    val_ds = torchvision.datasets.SVHN(paths['data'], split='test', download=True,
                              transform=testing_transform)

    test_ds = torchvision.datasets.SVHN(paths['data'], split='test', download=True,
                                  transform=testing_transform)

    # Generate validation set - https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    val_size = 0.5  # of test set
    num_test = len(test_ds)
    indices = list(range(num_test))
    split = int(np.floor(val_size * num_test))

    val_idx, test_idx = indices[split:], indices[:split]
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_size = len(train_ds)
    val_size = len(val_idx)
    test_size = len(test_idx)
    print(f'Split sizes: train: {train_size}, val: {val_size}, test: {test_size}')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              worker_init_fn=seed_worker,
                              num_workers=0,
                              generator=g)
    val_loader = DataLoader(val_ds,
                            sampler=val_sampler,
                            batch_size=batch_size,
                            worker_init_fn=seed_worker,
                            num_workers=0,
                            generator=g)
    test_loader = DataLoader(test_ds,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             shuffle=False,
                             worker_init_fn=seed_worker,
                             num_workers=0,
                             generator=g)
    
    train_loader.split_size = train_size
    val_loader.split_size = val_size
    test_loader.split_size = test_size

    return train_loader, val_loader, test_loader


def get_ImageNet_dataloaders(batch_size: int, seed: int):
    """ImageNet dataloaders with train, val sets.

    Args:
        batch_size: (int) batch size
        seed: (int) random seed

    Returns:
        train_loader: (DataLoader) train set
        val_loader: (DataLoader) validation set
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if BASE_LEARNER == 'MobileViT_IN':
        cropsize = 256
    else:
        cropsize = 224

    train_transform = T.Compose([
         T.RandomResizedCrop(cropsize),
         T.RandomHorizontalFlip(),
         T.ColorJitter(.0,.0,.0),
         T.ToTensor(),
         T.Normalize(mean, std)
         ])

    val_transform = T.Compose([
         T.Resize(256),
         T.CenterCrop(cropsize),
         T.ToTensor(),
         T.Normalize(mean, std)
         ])
    train_ds = ImageNetKaggle(paths['data'], "train", train_transform)
    val_ds = ImageNetKaggle(paths['data'], "val", val_transform)
    train_size = len(train_ds)
    val_size = len(val_ds)
    print(f'Split sizes: train: {train_size}, val: {val_size}')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,  # may need to reduce this depending on your GPU
        num_workers=data_config['ImageNet']['num_workers'],  # may need to reduce this depending on your num of CPUs and RAM
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=g
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,  # may need to reduce this depending on your GPU
        num_workers=data_config['ImageNet']['num_workers'],  # may need to reduce this depending on your num of CPUs and RAM
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        generator=g
    )
    train_loader.split_size = train_size
    val_loader.split_size = val_size
    return train_loader, val_loader


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]