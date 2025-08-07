"""federated-learning: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_learning.task import MLP, get_weights, load_data


def server_fn(context: Context):
    print("server_fn called")
    
    # Read from config 
    # the run_config ist set in pyproject.toml, if they are not set there, Flower default values are used.
    num_rounds = context.run_config["num-server-rounds"] # the number of rounds the server will run
    fraction_fit = context.run_config["fraction-fit"] # the fraction of clients that are selected for training in each round

    # Load data, so we can check the input dimension (feature size) of the model
    data_path = "/Users/ginasixt/federated-learning/diabetes_binary_health_indicators_BRFSS2015.csv"
    client_loaders, _ = load_data(
        data_path=data_path,
        num_clients=1,
    )
    input_dim = client_loaders[0].dataset[0][0].shape[0]

    # Creates a new MLP model with the input dimension and gets the initial weights.
    ndarrays = get_weights(MLP(input_dim=input_dim))
    # Convert the weights to Flower parameters, so we can send them to the clients to adjust them.
    parameters = ndarrays_to_parameters(ndarrays)

    def weighted_average(metrics):
        # metrics = List[Dict[str, float]] von allen Clients
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}

    # Define strategy, we chose FedAvg here (default Flower stategy), but we can also use other strategies.
    # A strategy defines how the server aggregates the model updates from the clients.
    # So we can set configurations like epochs, learning rate, etc. here.
    # we then call those config in the client app
    # via config.get("epochs", 1). The one means that if no epochs are set, 
    # the default value is 1.
    strategy = FedAvg(
        # the fraction of clients that are selected for training in each round
        fraction_fit=fraction_fit,

        # all clients are selected for evaluation
        fraction_evaluate=1.0,

        min_fit_clients=2,  # Minimum number of clients to fit the model

        min_evaluate_clients=2,  # Minimum number of clients to evaluate the model

        # a minimum of 2 clients are required to start the training
        min_available_clients=2,

        # the initial model weights
        initial_parameters=parameters,

        evaluate_metrics_aggregation_fn= weighted_average, # Aggregation function for evaluation metrics

        # TODO(ginasixt): we can also set other evaluation metrics here, like F1 score, precision, recall, AUC ...
        # fit_config={"epochs": 5, "lr": 0.01},  # Set other custom parameters here
    )

    # Create server config, we set the number of rounds here
    config = ServerConfig(num_rounds=num_rounds)

    # return the server app components, which includes the defined strategy and config
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
# ServerApp is the main entry point for the Flower server
# It orchestrates the federated learning process, manages clients, and controls training and evaluation.
# When we flwr run, flower reads from the pyproject.toml file and loads the configurations. 
# 
# server_fn prepares everything we need to run the server
    # creates the model, defines the strategy, and sets the server config (numberof rounds)
app = ServerApp(server_fn=server_fn)
