import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

def load_data(batch_size):
    # Transformations for training data
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-8.0, 8.0), fill=(1,)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Transformations for testing data
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    SEED = 1

    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    print("CUDA Available?", cuda_available)

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if cuda_available:
        torch.cuda.manual_seed(SEED)

    # DataLoader arguments
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda_available else dict(shuffle=True, batch_size=64)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    return train_loader, test_loader 