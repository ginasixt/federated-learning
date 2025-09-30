# federated_learning/server_app.py
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context

# WICHTIG: ServerAppComponents importieren (neue API)
try:
    # je nach Version liegt es hier:
    from flwr.server.app import ServerAppComponents
except ImportError:
    # oder hier (Fallback, falls sich der Pfad minimal unterscheidet)
    from flwr.server import ServerAppComponents  # type: ignore


# flwr run lädt über pyproject.toml serverapp = "…server_app:app".
# Flower ruft server_fn(context) auf.
#  context.run_config (die TOML-Werte) werden gelesen
# bauen FedAvg(..., on_fit_config_fn=..., evaluate_metrics_aggregation_fn=...).
# udn geben ServerAppComponents(config, strategy) zurück.
# Ab dann orchestriert Flower die Runden (Sampling, Fit, Evaluate).
def server_fn(context: Context):
    rc = context.run_config

    def on_fit_config_fn(rnd: int):
        return {
            "epochs": 1,
            "lr": 1e-2 if rnd < 3 else 5e-3,
            "batch-size": int(rc.get("batch-size", 128)),
        }

    def evaluate_metrics_aggregation_fn(metrics):
        total = sum(n for n, _ in metrics)
        acc = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / max(total, 1)
        auc = sum(n * m.get("auc", 0.0) for n, m in metrics) / max(total, 1)
        precision = sum(n * m.get("precision", 0.0) for n, m in metrics) / max(total, 1)
        recall = sum(n * m.get("recall", 0.0) for n, m in metrics) / max(total, 1)
        f1 = sum(n * m.get("f1", 0.0) for n, m in metrics) / max(total, 1)
        specificity = sum(n * m.get("specificity", 0.0) for n, m in metrics) / max(total, 1)
        return {
            "accuracy": acc,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
        }

    def fit_metrics_aggregation_fn(metrics):
        total = sum(n for n, _ in metrics)
        acc = sum(n * m.get("fit_accuracy", 0.0) for n, m in metrics) / max(total, 1)
        loss = sum(n * m.get("fit_loss", 0.0) for n, m in metrics) / max(total, 1)
        auc = sum(n * m.get("fit_auc", 0.0) for n, m in metrics) / max(total, 1)
        return {
            "fit_accuracy": acc,
            "fit_loss": loss,
            "fit_auc": auc,
        }

    strategy = FedAvg(
        fraction_fit=float(rc.get("fraction-fit", 0.5)),
        fraction_evaluate=float(rc.get("fraction-evaluate", 1.0)),
        min_fit_clients=int(rc.get("min-fit-clients", 5)),
        min_evaluate_clients=int(rc.get("min-evaluate-clients", 2)),
        on_fit_config_fn=on_fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    cfg = ServerConfig(num_rounds=int(rc.get("num-server-rounds", 5)))


    return ServerAppComponents(config=cfg, strategy=strategy)

# Create ServerApp
# ServerApp is the main entry point for the Flower server
# It orchestrates the federated learning process, manages clients, and controls training and evaluation.
# When we flwr run, flower reads from the pyproject.toml file and loads the configurations. 
# 
# server_fn prepares everything we need to run the server
# # creates the model, defines the strategy, and sets the server config (numberof rounds)

app = ServerApp(server_fn=server_fn)


