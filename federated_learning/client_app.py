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
        # PyTorch device (CPU or GPU), we get the epochs from the config
        self.device = device

    # FOR EVERY ROUND THE FOLLOWING METHODS ARE CALLED:
    # 1) fit is called to train the model locally for each client
        # 1) the new weights are set
        # 2) the model is trained on the local data for a number of epochs
        # 3) the new weights are returned to the server
    def fit(self, parameters, config):
        # Sets the model weights and trains the model on the local data
        self.set_parameters(parameters)

        # the config arument is send from the server to the client
        # the server can set the number of epochs, learning rate, etc. (look at server_app.py)
        # in the next step we then train our local model.
        epochs = int(config.get("epochs", 1)) 

        # we train the model on the local data on the specified device (CPU or GPU) and the set number of epochs (on client)
        # train() is implemented in task.py, it trains the model on the local data and returns the average training loss.
        train_loss = train(self.model, self.trainloader, epochs, self.device)
        return self.get_parameters(), len(self.trainloader.dataset), {"train_loss": train_loss}

    # 1.1) sets the model weights while training (fit())
    def set_parameters(self, parameters):
        # Setzt die Modellgewichte
        set_weights(self.model, parameters)


    # 1.2) After the training is complete, we return the current model weights to the server
    def get_parameters(self, config=None):
        # Gibt die aktuellen Modellgewichte zurück
        return get_weights(self.model)

   
    # 2) this method is called to evaluate the local clients model
        # 1) the new weights are set
        # 2) the model is evaluated on the local test data
        # 3) Loss and accurany are returned
        # TODO(ginasixt): we can also set other evaluation metrics here, like F1 score, precision, recall, AUC ...
    def evaluate(self, parameters, config):
        # Same as in fit, but we evaluate the model on the local test data
        self.set_parameters(parameters)
        # TODO(ginasixt): we are using the test data, but we can later also create a validation set.
        loss, accuracy = test(self.model, self.testloader, self.device)
        print("[FlowerClient] Evaluation complete, accuracy:", accuracy)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# this function created for every client a modell.
def client_fn(context: Context):
    # Lade Daten für diesen Client
    data_path = "/Users/ginasixt/federated-learning/diabetes_binary_health_indicators_BRFSS2015.csv"
    num_clients = context.node_config["num-partitions"]
    partition_id = context.node_config["partition-id"]

    batch_size = 32
    local_epochs = context.run_config["local-epochs"]

    # train_loader, test_loader = load_data(
    #     data_path=data_path,
    #     batch_size=batch_size,
    #     num_clients=num_clients,
    #     partition_id=partition_id,
    #     test_split=0.2,
    # )

    # input_dim = train_loader.dataset[0][0].shape[0]
    # model = MLP(input_dim=input_dim)
    # return FlowerClient(model, train_loader, test_loader, torch.device("cpu")).to_client()

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
