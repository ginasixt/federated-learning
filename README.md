# Diabetes Prediction with Federated Learning using Flower & PyTorch
This project uses a Multilayer Perceptron (MLP) built with PyTorch to predict diabetes based on health indicators in a federated learning setting. 
The model is trained on the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) dataset and orchestrated using [Flower](https://flower.ai/).

The dataset has **253,680 samples** from CDC survey, **21 health indicators** like BMI, age, blood pressure, cholesterol, etc. and is highly **Imbalanced**: 86.1% healthy, 13.9% at-risk.

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

## Table of Contents

- [Installation](#-installation)
- [Project Architecture](#-project-architecture)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Multi-Threshold Optimization](#-multi-threshold-optimization)
- [Screening Policy](#-screening-policy)
- [Results & Visualizations](#-results--visualizations)
- [Configuration](#-configuration)
- [References](#-references)


---

## Installation


## Quick Start

### Prepare Data

```bash
# Download dataset from Kaggle and create train/val/test splits
python federated_learning/tools/prepare_data.py \
    --csv diabetes_binary_health_indicators_BRFSS2015.csv \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json
```

**Output:**
- `data/diabetes.parquet`: Normalized features + row IDs
- `data/norm_stats.json`: Mean/std for normalization + split indices

**Why Normalization?**
- Features have different scales (e.g., Age: 1-13, BMI: 12-98)
- Neural networks converge faster with normalized inputs
- Prevents features with large values from dominating

**Split Distribution:**
- Train: 70% (177,576 samples) -> Distributed across 10 clients (simulates 10 hospitals)
- Validation: 10% (25,368 samples) -> for training hyperparameter like threshold selection without test set contamination
- Test: 20% (50,736 samples) -> Global Dataset (centralized Evaluation)

### Create Client Partitions

```bash
# Non-IID split using Dirichlet distribution (α=0.3)
python federated_learning/tools/make_splits.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --out splits_dirichlet_10_a03.json \
    --num-partitions 10 \
    --mode dirichlet \
    --alpha 0.3
```

**Output:** `splits_dirichlet_10_a03.json`
```json
{
  "train": {
    "0": [12, 45, 78, ...],  // Client 0's training sample IDs
    "1": [23, 56, 89, ...],
    ...
  },
  "val": {
    "0": [105, 234, ...],    // Proportional validation split
    "1": [67, 189, ...],
    ...
  }
}
```

**Dirichlet α Parameter:**
- `α=0.1`: Highly non-IID (e.g., Hospital A: 90% elderly, Hospital B: 90% young)
- `α=0.3`: Moderate heterogeneity (realistic)
- `α=1.0`: Mild non-IID

### Run Federated Training

```bash
# Train for 20 rounds with 10 clients
flwr run
```

**What happens then:**
1. **Server** initializes global model (random weights)
2. **Each Round:**
   - Server samples 8/10 clients for training
   - Clients download global model
   - Clients train locally for 1 epoch
   - Clients upload weight updates
   - Server aggregates using FedAvg
   - All 10 clients evaluate on local validation data (10 thresholds each)
   - Server selects best threshold based on aggregated metrics
3. **After 20 Rounds:**
   - Screening policy selects best model
   - Checkpoint saved to `result/alpha03/multi_thr/model_round_X.pt`
   - Metrics saved to `result/alpha03/multi_thr/run_1.json`

**Training Output (Console):**
```
 Round 1/20
  ├─ Training: 8 clients selected
  ├─ Aggregating weights...
  ├─ Evaluating: 10 thresholds tested
  └─ Best Threshold: 0.45 (Recall=0.78, Spec=0.72, F1=0.42)

 Round 5/20
  ├─ Training: 8 clients selected
  ├─ Aggregating weights...
  ├─ Evaluating: 10 thresholds tested
  └─ Best Threshold: 0.50 (Recall=0.79, Spec=0.70, F1=0.43)
  NEW BEST ROUND - Saving checkpoint...

...

 Final Summary:
   Best Round: 5
   AUC: 0.8223
   Recall: 0.7847
   Specificity: 0.7029
   Threshold: 0.55 (from validation)
```

### Evaluate on Test Set

```bash
python federated_learning/tools/final_test_evaluation_with_val_threshold.py \
    --result-json result/alpha03/multi_thr/run_1.json \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output final_evaluation
```

**Output Files:**
- `final_evaluation/test_report.json`: All metrics (AUC, confusion matrix, etc.)
- `final_evaluation/test_roc_curve.png`: Publication-ready ROC plot
- `final_evaluation/test_confusion_matrix.png`: Annotated confusion matrix
- `final_evaluation/test_roc_data.npz`: Raw ROC data (for reproducibility)

### Generate Plots

```bash
python plot_results.py \
    --root result \
    --out plots_out_03
```

**Output**: 10+ plots analyzing performance across thresholds and α values

---

## Project Architecture

### Directory Structure

```
federated-diabetes-screening/
│
├── federated_learning/          # Core package
│   ├── __init__.py
│   ├── client_app.py               # Client training logic
│   ├── server_app.py               # Aggregation strategy
│   ├── task.py                     # Data loading utilities
│   ├── screening_policy.py         # Best round selection
│   │
│   ├── tools/                   # Data preparation scripts
│   │   ├── prepare_data.py         # Train/val/test split + normalization
│   │   ├── make_splits.py          # Dirichlet partitioning
│   │   └── final_test_evaluation_with_val_threshold.py # for evaluation of the best round with the test set
│   │
│   └── plotting/                # Visualization tools
│       ├── compare_thresholds_report.py
│       ├── more_screening_plots.py
│       └── plot_centralized_roc.py
│
├── data/                        # Prepared datasets
│   ├── diabetes.parquet            # Normalized features
│   ├── norm_stats.json             # Mean/std + split indices
│   └── splits_dirichlet_10_a03.json
│
├── final_evaluation/            # Test set results
│   ├── test_report.json
│   ├── test_roc_curve.png
│   └── test_confusion_matrix.png
│
├── pyproject.toml                  # Project config + dependencies
└── README.md                       # This file
```

---


## Model Architecture

### Multi-Layer Perceptron (MLP)

```python
MLP(
  (net): Sequential(
    (0): Linear(in_features=21, out_features=256)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=128)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=2)  # Binary classification
  )
)
```

**Parameters**: 59,650 (trainable)

**Design Choices:**

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Input Dim** | 21 | Number of health indicators |
| **Hidden Layers** | [256, 128] | Sufficient capacity for tabular data |
| **Activation** | ReLU | Fast, prevents vanishing gradients |
| **Output** | 2 logits | Softmax for class probabilities |
| **Initialization** | Kaiming Normal | Optimal for ReLU networks |
| **Dropout** | None | Implicit regularization via FedAvg |


## Training Process

Our federated learning system employs several techniques to handle the challenges of of medical AI: 
- severe class imbalance (86% healthy vs 14% diabetic)
- heterogeneous client data
- and the need for high recall in clinical screening.

---

### Class Weighting (Weighted Cross-Entropy)
- Standard training would produce a model that predicts "healthy" for everyone, achieving 86% accuracy but missing all diabetic cases
- We apply weighted Cross-Entropy loss using global prevalence statistics. The positive class (diabetic) receives significantly higher weight than the negative class (healthy), forcing the model to prioritize recall over overall accuracy. 
- The class weights are computed from the full training set (not client-local - in real FL enviroment each client should send the aggregated count to the server, the sever computes global prevalence and broadcasts class weights, so that the raw patient data never leaves the client)
- The boost factor is configurable (default 2.0) via TOML.

### Federated Averaging (FedAvg)
- Each round, the server samples 80% of clients (8 out of 10) for training while all clients participate in evaluation. Client updates are aggregated using data-size-weighted averaging, giving larger hospitals proportionally more influence.
- Implemented by the Flower strategy, wrapped in our custom class `federated_learning.server_app.FedAvgWithScreening`.
- Why: FedAvg is simple, robust, and effective for cross-silo FL. Weighting by local sample counts balances influence across heterogeneous clients and speeds convergence.

### FedProx for Non-IID Stability
- Non-IID data distributions cause "client drift", clients with different patient demographics can push model updates in conflicting directions, leading to oscillating convergence.
- We add a proximal regularization term that penalizes large deviations from the global model during local training. This keeps client updates within a reasonable neighborhood of the global optimum while preserving local adaptation capability.
- The proximal coefficient (μ = 0.001) provides mild regularization without over-constraining local learning, improving stability by ~15-20% in our non-IID experiments.

### Adaptive Learning Rate Schedule  

**Strategy**: Two-phase learning rate schedule optimized for short federated runs:
- **Phase 1 (Rounds 1-2)**: High learning rate (0.01) for rapid exploration of parameter space
- **Phase 2 (Rounds 3-20)**: Reduced learning rate (0.005) for careful refinement around promising solutions

Early rounds benefit from aggressive updates to quickly escape poor local minima, while later rounds require stability to converge on optimal parameters without overshooting.

### Gradient Clipping for Stability
- Non-IID client data can produce rare gradient spikes that destabilize training. We clip gradients to a maximum L2 norm (5.0) to prevent exploding gradients while preserving gradient direction.
- Essential for training stability when combined with class weighting, which amplifies gradients for minority class examples. Prevents training failures from outlier batches without slowing normal convergence.

### Weight Decay Regularization
L2 penalty (λ = 1e-4) applied via optimizer weight decay, equivalent to adding a regularization term that shrinks parameters toward zero.
- Prevents overfitting to small client datasets
- Encourages simpler model hypotheses that transfer better across clients
- Reduces parameter variance in federated settings where clients see different data distributions

### Additional Training Optimizations

**SGD with Momentum**: We use SGD with 0.9 momentum rather than adaptive optimizers (Adam, AdaGrad), as momentum-based methods are more robust to the noise and heterogeneity inherent in federated learning.

**Single Local Epoch**: Each client trains for exactly one epoch per round - a federated learning best practice that prevents overfitting to local data while maintaining update diversity across clients.

**Kaiming Initialization**: Hidden layers use Kaiming Normal initialization optimized for ReLU activations, ensuring stable gradient flow from the start of training.

**Multi-threshold validation** and screening policy
Each client evaluates a grid of thresholds on its validation split. The server aggregates confusion-matrix counts per threshold and selects the best threshold using a clinically motivated composite score and tracks history.
Screening requires prioritizing recall while maintaining acceptable specificity and stability. Threshold selection on validation (not test) prevents leakage and simulates realistic deployment.
At the end of the run we evaluate the best round (best metrics according to screening policy) with the test data set.

We select the best threshold by:

**Hard Constraints:**
- **Recall ≥ 0.75**: Safety requirement (must catch 75%+ of diabetic patients)

**Soft Preferences:**
- **Specificity ≥ 0.70**: Avoid overwhelming follow-up capacity
- **F1 Score**: Overall balance
- **Stability**: Prefer thresholds that work across clients

**Fallback Strategy:**
If no threshold meets min_recall:
1. Try relaxed recall (0.70) + spec (0.65)
2. Use Youden's Index: `J = recall + spec - 1`
3. Log warning for manual review

---

## Screening Policy
bla bla bla

### Pareto Optimality

**Definition**: Round A dominates Round B if:
- `recall(A) ≥ recall(B)` **AND**
- `spec(A) ≥ spec(B)` **AND**
- At least one inequality is strict

**Example:**
```
Round 5:  Recall=0.785, Spec=0.729  ← Pareto-optimal
Round 10: Recall=0.790, Spec=0.706  ← Pareto-optimal (higher recall, lower spec)
Round 8:  Recall=0.780, Spec=0.720  ← Dominated by Round 5
```

### Stability Score
Late rounds might overfit to validation data, we adress this too with the stability score



## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.


## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)

