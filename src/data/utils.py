import data.dataloaders as dl

def get_dataloaders(dataset: str, batch_size: int, seed: int):
    """Get dataloaders for a given dataset"""
    if dataset == 'CIFAR-10':
        return dl.get_CIFAR_dataloaders(version='10', batch_size=batch_size, seed=seed)
    elif dataset == 'CIFAR-100':
        return dl.get_CIFAR_dataloaders(version='100', batch_size=batch_size, seed=seed)
    elif dataset == 'SVHN':
        return dl.get_SVHN_dataloaders(batch_size=batch_size, seed=seed)
    elif dataset == 'ImageNet':
        return dl.get_ImageNet_dataloaders(batch_size=batch_size, seed=seed)
    else:
        raise ValueError('Dataset not recognised')