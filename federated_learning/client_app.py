"""federated-learning: A Flower / PyTorch app."""

import torch
import flwr as fl

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_learning.task import MLP, get_weights, load_data, set_weights, test, train # alt


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config=None):
        # Gibt die aktuellen Modellgewichte zurück
        return get_weights(self.model)

    def set_parameters(self, parameters):
        # Setzt die Modellgewichte
        set_weights(self.model, parameters)

    def fit(self, parameters, config):
        # Setzt die Modellgewichte und trainiert das Modell
        self.set_parameters(parameters)
        epochs = int(config.get("epochs", 1))
        train_loss = train(self.model, self.trainloader, epochs, self.device)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Setzt die Modellgewichte und evaluiert das Modell
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def client_fn(context: Context):
    # Lade Daten für diesen Client
    data_path = "/Users/ginasixt/federated-learning/diabetes_binary_health_indicators_BRFSS2015.csv"
    num_clients = context.node_config["num-partitions"]
    partition_id = context.node_config["partition-id"]
    batch_size = 32
    local_epochs = context.run_config["local-epochs"]

    # Lade alle Partitionen und das Testset
    client_loaders, test_loader = load_data(
        data_path=data_path,
        batch_size=batch_size,
        num_clients=num_clients,
        test_split=0.2,
    )

    # Input-Dimension aus den Daten bestimmen
    input_dim = client_loaders[partition_id].dataset[0][0].shape[0]
    model = MLP(input_dim=input_dim)

    # Erzeuge Client für die Partition
    return FlowerClient(model, client_loaders[partition_id], test_loader, torch.device("cpu")).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
