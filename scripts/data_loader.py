# scripts/data_loader.py
import os
import torch
from torchvision import datasets, transforms

def load_mnist(data_dir="data/", batch_size=128):
    """
    Télécharge et charge les données MNIST.

    Args:
        data_dir (str): Répertoire où stocker les données.
        batch_size (int): Taille des batchs pour les DataLoader.

    Returns:
        tuple: DataLoader pour les ensembles d'entraînement et de test.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Vérifier si le dossier existe
    os.makedirs(data_dir, exist_ok=True)
    
    # Charger les ensembles d'entraînement et de test
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Créer des DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
