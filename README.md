# Diabetes Prediction with Federated Learning using Flower & PyTorch
This project uses a Multilayer Perceptron (MLP) built with PyTorch to predict diabetes based on health indicators in a federated learning setting. 
The model is trained on the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) dataset and orchestrated using [Flower](https://flower.ai/).

By splitting the dataset across clients, we simulate a real-world federated learning scenario where data remains decentralized.\
Each client:
- Holds a subset of the full dataset
- Trains the MLP model locally using PyTorch \
  -> Want to learn more about my work and insights with MLPs? \
  Check out my other Repo:  [Multilayer Perceptron](https://github.com/ginasixt/Multilayer-Perceptron.git).
- Sends only the updated model weights back to the central server
- Keeps all raw data private and local

In real-world federated learning (also called "cross-silo"), this data is already distributed across organizations, there's no need to split the data artificially.

---

## Project Structure

```bash
diabetes-flower
├── diabetes_flower
│ ├── init.py
│ ├── client_app.py # Defines the ClientApp
│ ├── server_app.py # Defines the ServerApp
│ └── task.py # Model, data loading and training logic
├── diabetes_binary_health_indicators_BRFSS2015.csv # Dataset
├── pyproject.toml # Project metadata like dependencies and configs
└── README.md # This file
```

## Installation

Install the dependencies using the `pyproject.toml`:

```bash
pip install -e .
```
> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `federated-learning` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## My Learnings & How FL Works
Federated Learning (FL) is a machine learning approach where a central model is trained without ever accessing the clients' raw data. The data stays on each client, and only model updates are communicated.
Here’s how a typical FL round works:

### Federated Flow Overview
I use the Flower App API (ServerApp/ClientApp), so there is no start_server()/start_simulation() call. \
flwr run-CLI loads the app objects and calls the functions (server_fn, client_fn). \

Translated with DeepL.com (free version)

1) Start of round (server) \
Strategy (in our case FedAvg) decides which subset of clients will train in this round.
3) Send Global Model \
Passes the parameter to the selected Clients
4) Local Training Begins \
fit(...) loads the parameters (set_weights), trains (train(...), e.g., 1–5 epochs), and returns only weights/metrics.
5) Aggregation \
Strategy aggregate_fit(...) performs FedAvg (weighted average).
6) Repeat Until Converged \
This process continues over multiple rounds .

Flower handles all communication, orchestration, and aggregation. \
It provides:
- Functions: start_server, start_simulation, FedAvg, custom Strategy, etc.
- Protocols: for secure communication between clients and server
- Framework: controlling rounds, aggregation, fault tolerance, and more

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)

