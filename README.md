# Neural Network — From Single Neuron to Two Hidden Layers

![Neural Network Diagram](image.png)

---

## 1. Single Neuron (`single-neuron.py`)

A single neuron takes multiple inputs, multiplies each by a weight, sums them up, and adds a bias.

```python
inputs  = [1.0, 2.0, 3.0]
weights = [0.2, 0.4, 0.6]
bias    = 2.0

output = np.dot(inputs, weights) + bias
```

**Formula:** `output = (1.0×0.2) + (2.0×0.4) + (3.0×0.6) + 2.0 = 4.8`

| Input | Weight | Input × Weight |
|-------|--------|----------------|
| 1.0   | 0.2    | 0.2            |
| 2.0   | 0.4    | 0.8            |
| 3.0   | 0.6    | 1.8            |
| —     | bias   | + 2.0          |
| **Output** | | **4.8**   |

---

## 2. Three Neurons / One Layer (`three-neurons.py`)

A layer of 3 neurons, each with its own set of weights and a bias, all sharing the same 4 inputs.

```python
inputs  = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2,  0.8,  -0.5,  1.0],   # neuron 1
           [0.5, -0.91,  0.26, -0.5],   # neuron 2
           [-0.26, -0.27, 0.17, 0.87]]  # neuron 3
biases  = [2.0, 3.0, 0.5]

layer_outputs = np.dot(weights, inputs) + biases
```

Each row in `weights` belongs to one neuron. `np.dot(weights, inputs)` computes all 3 neuron outputs at once, then the corresponding bias is added.

| Neuron | Weights                    | Bias | Output |
|--------|----------------------------|------|--------|
| 1      | [0.2, 0.8, -0.5, 1.0]     | 2.0  | 4.8    |
| 2      | [0.5, -0.91, 0.26, -0.5]  | 3.0  | 1.21   |
| 3      | [-0.26, -0.27, 0.17, 0.87]| 0.5  | 2.385  |

---

## 3. Two Hidden Layers (`two-hidden-layers.py`)

The diagram shows a fully connected (dense) neural network with:
- **Input layer** — 4 neurons, one per feature
- **Hidden Layer 1** — 3 neurons
- **Hidden Layer 2** — 3 neurons

Every neuron in each layer is connected to every neuron in the next layer (orange lines).

```python
inputs = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]]
```

This is a **batch of 3 samples**, each with **4 features**:

| Sample | Feature 1 | Feature 2 | Feature 3 | Feature 4 |
|--------|-----------|-----------|-----------|-----------|
| 1      | 1.0       | 2.0       | 3.0       | 2.5       |
| 2      | 2.0       | 5.0       | -1.0      | 2.0       |
| 3      | -1.5      | 2.7       | 3.3       | -0.8      |

The 4 values in each sample map to the 4 input neurons. The 3 rows represent 3 samples processed as a batch — the diagram reflects the structure for a single sample.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Input** | Raw feature values fed into the network |
| **Weight** | Learnable multiplier for each input connection |
| **Bias** | Learnable offset added after the weighted sum |
| **`np.dot`** | Efficiently computes weighted sums across all neurons |
| **Batch** | Multiple samples processed together in one forward pass |
