"""federated-learning: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from collections import OrderedDict
import random


class MLP(nn.Module):
    """
    Mehrschichtiges Perzeptron für tabellarische Diabetes-Daten.

    Args:
        input_dim: Anzahl der Eingangsfunktionen (Features).
        hidden_dims: Liste von Hidden-Layer-Größen.
        output_dim: Anzahl der Ausgabeklassen (2 für binäre Klassifikation).
    """
    def __init__(self, input_dim: int, hidden_dims=[64, 32], output_dim: int = 2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    




def load_data(
    data_path: str,
    batch_size: int = 32,
    num_clients: int = 10,
    #partition_id: int = 0,
    test_split: float = 0.2,
):
    """
    Lädt den Diabetes CSV-Datensatz, trennt Features und Label, erstellt TensorDatasets,
    splittet in Train/Test und partitioniert das Training auf mehrere Clients.
    Es gibt nur die gewünschte Partition und das Test-Set zurück.

    Args:
        data_path: Pfad zur CSV-Datei mit dem Diabetes-Datensatz.
        batch_size: Batch-Größe für DataLoader.
        num_clients: Anzahl der Clients für Federated Learning.
        test_split: Anteil an Daten, der als Test-Set verwendet wird (zwischen 0 und 1).

        Not implemnted anymore: partition_id: Index der gewünschten Client-Partition (0-basiert).
        TODO(ginasixt): Implementere dass das Dataset nur einmal geladen wird und man dann mit partition_id die Partition auswählt kp durch speichern oder so.

    Returns:
        train_loaders: Liste von DataLoader-Objekten für jeden Client.
        test_loader: DataLoader für das gemeinsame Test-Set.
    """
    # Setze den Zufalls-Seed für Reproduzierbarkeit damit alle Clients die gleiche Partition bekommen, wenn sie load_data aufrufen 
    # keine saubere Lösung aber erstmal ok :P
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # 1. CSV laden
    df = pd.read_csv(data_path)

    # 2. Features und Target trennen
    if 'Diabetes_binary' not in df.columns:
        raise ValueError("Spalte 'Diabetes_binary' nicht in CSV gefunden.")
    X = df.drop('Diabetes_binary', axis=1).values
    y = df['Diabetes_binary'].values

    # 3. In Torch-Tensor konvertieren
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 4. Dataset erstellen
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # 5. Train/Test Split
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    # Maybe usage for later
    # base_size = train_size // num_clients
    # start = partition_id * base_size
    # end = start + base_size if partition_id < num_clients - 1 else train_size
    # indices = list(range(start, end))
    # subset = Subset(train_dataset, indices)
    # train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # return train_loader, test_loader

    # Benutzen wir später nochmal wenn wir die Partitionierung zb sauber speichern wollen.
    #TODO(ginasixt): Implementere dass das Dataset nur einmal geladen wird und man dann mit partition_id die Partition auswählt kp durch speichern oder so.

    # 6. Partitionierung für Clients
    # Gleich große Partitionen, letzter Client bekommt Rest
    client_loaders = []
    base_size = train_size // num_clients
    for i in range(num_clients):
        start = i * base_size
        end = start + base_size if i < num_clients - 1 else train_size
        indices = list(range(start, end))
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    # 7. Test-Loader und Testset speichern
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader


# Trains the model, like we did it in our previous MLP Project for more explanation look there.
def train(net, trainloader, epochs, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_train_loss = running_loss / len(trainloader)
    return avg_train_loss


# Evaluates the model on the test data, like we did it in our previous MLP Project for more explanation look there.
def test(net, testloader, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            total_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy


# get the model weights as numpy arrays
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# set the model weights from the numpy arrays
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
