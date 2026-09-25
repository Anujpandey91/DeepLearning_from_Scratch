# Module 01: CNN Foundations

This module builds a small convolutional neural network from scratch with
NumPy. The implementation is intentionally explicit so that convolution,
pooling, dense layers, backpropagation, and parameter updates remain visible
while studying the mathematics behind CNNs.

## Concepts Covered

- Single-channel, multi-channel, and multi-filter convolution
- ReLU and numerically stable softmax activations
- Max pooling and average pooling
- Flattening and dense layers
- Categorical cross-entropy
- Backpropagation through convolution, pooling, ReLU, and dense layers
- Gradient descent updates and sample-by-sample SGD
- Numerical gradient checking with finite differences
- End-to-end MNIST training and feature-map inspection

## Structure

```text
module_01_cnn_foundations/
├── concepts/          # Reserved for focused mathematical explanations
├── experiments/       # Forward-pass, gradient-check, and training experiments
├── implementations/   # NumPy CNN building blocks and the CNN class
├── tests/              # Focused unit tests for the building blocks
└── visualizations/    # Reserved for reusable visualization code
```

## Shape Conventions

Images use channels-last layout:

```text
image:   (H, W, C)
filters: (KH, KW, C, N)
output:  (H_out, W_out, N)
```

Dense, softmax, and loss operations use rows for features or classes and
columns for examples:

```text
X: (n_features, m)
W: (n_units, n_features)
b: (n_units, 1)
Y/A: (n_classes, m)
```

## Mathematical Components

For a multi-filter convolution, each output value is computed as:

```text
Z[i, j, n] = sum(X_region * W[:, :, :, n]) + b[n]
```

The dense layer uses `Z = W X + b`. For `m` examples, its backward pass is:

```text
dW = (1/m) dZ X^T
db = (1/m) sum(dZ)
dX = W^T dZ
```

Softmax normalizes logits by class, and categorical cross-entropy is:

```text
J = -(1/m) sum(Y log(A))
dZ = (1/m)(A - Y)
```

The implementation clips probabilities in the forward loss only to keep
`log(0)` finite. The softmax-plus-cross-entropy derivative is kept in its
standard simplified form.

## End-to-End Architecture

```text
Input (H, W, C)
	↓
Conv: 3×3 kernel, 2 filters, stride 1, no padding
	↓
ReLU
	↓
MaxPool: 2×2, stride 2
	↓
Flatten
	↓
Dense: 10 units
	↓
Softmax
```

For MNIST, the main shapes are:

```text
(28, 28, 1) → (26, 26, 2) → (13, 13, 2) → (338,) → (10, 1)
```

`CNN.fit(X, Y)` initializes parameters from the first image and performs
sample-by-sample SGD: each example runs through forward propagation, loss,
backpropagation, and an immediate gradient-descent update. `predict` returns
class indices and `evaluate` returns mean classification accuracy.

## Testing and Gradient Checking

The tests cover numerical outputs, shapes, invalid inputs, broadcasting, and
pooling behavior for the individual components. The gradient-check experiment
compares analytical convolution gradients against central finite differences
for input values, filters, and biases across single- and multi-channel cases,
with padding and stride variations.

Run the tests from the repository root with:

```bash
pytest -q modules/module_01_cnn_foundations/tests
```

Run the gradient check as a module so its relative imports resolve:

```bash
python -m modules.module_01_cnn_foundations.experiments.gradient_check_experiment
```

## MNIST Experiment

`experiments/cnn_training_experiment.ipynb` loads MNIST, normalizes the images
to `[0, 1]`, adds the channel dimension, and one-hot encodes labels. The
documented experiment uses 2,000 training images and 500 test images for 10
epochs. With only two convolution filters, it reached approximately 88.6%
accuracy on the 500-image test subset. It also inspects misclassified images,
learned first-layer filters, and the corresponding feature maps.

This is an educational baseline, not a state-of-the-art MNIST benchmark. It
uses a small fixed architecture, one example per update, and handwritten
NumPy operations rather than TensorFlow, PyTorch, Keras, or sklearn.

## Limitations and Next Steps

The current CNN supports one fixed architecture and ten output classes. The
optimizer module currently contains vanilla `GradientDescent`; Momentum,
RMSprop, and Adam are not present in this checkout. The next natural steps are
to add more concept notes, broaden component tests, and introduce configurable
architectures only after the underlying mechanics are fully understood.
