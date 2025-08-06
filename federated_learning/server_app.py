"""federated-learning: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_learning.task import MLP, get_weights, load_data


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Lade Daten, um die Input-Dimension zu bestimmen
    data_path = "/Users/ginasixt/federated-learning/diabetes_binary_health_indicators_BRFSS2015.csv"
    client_loaders, _ = load_data(data_path=data_path, num_clients=1)
    input_dim = client_loaders[0].dataset[0][0].shape[0]

    # Initialize model parameters
    ndarrays = get_weights(MLP(input_dim=input_dim))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
