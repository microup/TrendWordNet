# TrendWordNet

## Overview

TrendWordNet is a multilayer perceptron (MLP) for binary classification of words as trendy or non-trendy, built entirely in **pure Go** with **no external libraries or dependencies**. It analyzes words in Russian (Cyrillic) and English (Latin) alphabets by converting them into 60-dimensional feature vectors based on normalized letter frequencies and word length. The model achieves a validation accuracy of 97% and leverages only the Go standard library for maximum portability and simplicity.

The network is trained on a dataset of labeled words, using a weighted binary cross-entropy loss to handle class imbalance. It employs ReLU activations for hidden layers, a sigmoid output, Adam optimization, dropout, and L2 regularization for robust performance.

## Characteristics

### Architecture

- **Type**: Multilayer Perceptron (MLP).
- **Input Layer**: 60 neurons (feature vector size).
- **Hidden Layer 1**: 128 neurons, ReLU activation, 10% dropout.
- **Hidden Layer 2**: 64 neurons, ReLU activation, 10% dropout.
- **Output Layer**: 1 neuron, sigmoid activation.
- **Weight Initialization**:
  - He initialization for hidden layers (W1, W2).
  - Xavier initialization for output layer (W3).
- **Bias Initialization**: Normal distribution (σ=0.1).

### Training Hyperparameters

- **Optimizer**: Adam (β1=0.9, β2=0.999, ε=1e-8).
- **Learning Rate**: 0.001.
- **Loss Function**: Weighted binary cross-entropy (to address class imbalance).
- **L2 Regularization**: λ=1.
- **Dropout**: 10% on hidden layers.
- **Maximum Epochs**: 5000, with early stopping (patience=100 for training loss, patience=100 for validation accuracy).
- **Batch Size**: 1 (stochastic gradient descent).
- **Validation Accuracy**: 97%.

### Data

- **Training Set**: ~2000 words (≈35.91% trendy).
- **Validation Set**: ~400 words (≈35.75% trendy).
- **Input Format**: CSV files with two columns (word, label: 0 or 1).
- **Feature Vector**: 60 features (normalized letter frequencies for Cyrillic/Latin alphabet + normalized word length).

## Installation

TrendWordNet is written in **pure Go** and requires **no external libraries or dependencies**. It uses only the Go standard library, ensuring seamless execution across platforms.

### Requirements

- **Go**: Version 1.23 or higher.

### Steps

1. Clone or download the repository:

   ```bash
   git clone https://github.com/microup/TrendWordNet.git
   cd TrendWordNet
   ```

2. Place the training dataset (`data/data.csv`), validation dataset (`data/test.csv`), and trained model file (`data/best_network.gob`) in the `data/` directory.
3. No additional dependencies or library installations are needed—pure Go is sufficient.

## Usage

The repository includes two main programs:

- `main.go`: Trains the network and saves it to `data/network.gob` and `data/best_network.gob`.
- `trends/trends.go`: Loads the trained model and predicts trendiness for a list of words.

### Training the Model

Run `main.go` to train the network:

```bash
go run main.go
```

This will:

- Load training and validation datasets from `data/data.csv` and `data/test.csv`.
- Train the network with the specified hyperparameters.
- Save the trained model to `data/network.gob` and the best model to `data/best_network.gob`.

Example output:

```
Loaded 2000 examples
Loaded 400 validation examples
Training set balance: 35.91% trendy, 64.09% non-trendy
Validation set: 35.75% trendy
Starting training...
Epoch 1 of 5000, Average loss: 0.693147, Accuracy (training): 64.20%, Accuracy (validation): 65.00%
...
Training completed. Best validation accuracy: 97.00%
Network saved: data/network.gob
```

### Predicting Trendiness
Run `trends/trends.go` to predict the trendiness of words:

```bash
go run trends/trends.go
```

This will load the trained model and classify a predefined list of words.

Example output:

```
Network successfully loaded from file
Word 'кот': non-trendy (probability: 32.15%)
Word 'программа': trendy (probability: 78.92%)
Word 'база': trendy (probability: 65.43%)
...
```

### Custom Prediction

To predict trendiness for a custom word, use the `IsTrendy` method in your Go code:

```go
package main

import (
	"fmt"
	"neural_network/network"
)

func main() {
	network, err := network.LoadNetwork("data/best_network.gob")
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		return
	}

	word := "soft"

	isTrendy := network.IsTrendy(word)

  _, _, output := network.Forward(network.WordToVector(network.Normalize(word)))

  fmt.Printf("Word '%s': %s (probability: %.2f%%)\n",
		word,
		map[bool]string{true: "trendy", false: "non-trendy"}[isTrendy],
		output[0]*100)
}
```

## Advantages

1. **High Accuracy**: Achieves 97% validation accuracy for reliable trendiness classification.
2. **Pure Go Implementation**: No external dependencies, using only the Go standard library for maximum portability and ease of deployment.
3. **Lightweight and Fast**: Optimized for performance, ideal for resource-constrained environments.
4. **Bilingual Support**: Processes both Russian (Cyrillic) and English (Latin) alphabets, suitable for multilingual datasets.
5. **Robust Training**: Employs Adam optimizer, dropout, L2 regularization, and weighted loss to handle class imbalance and prevent overfitting.
6. **Simple Features**: Uses handcrafted 60-dimensional vectors based on letter frequencies and word length, eliminating the need for external embeddings.
7. **Reproducible**: Full source code for training, inference, and preprocessing ensures easy replication and customization.

## Limitations

- **Alphabet Restriction**: Processes only Cyrillic (а-я, ё) and Latin (a-z) letters; non-alphabetic characters (e.g., digits) are ignored.
- **Non-Informative Words**: Words with only non-alphabetic characters are classified as non-trendy.
- **Data Dependency**: Performance relies on the quality and diversity of the training dataset.
- **No Advanced Embeddings**: Uses handcrafted features instead of pretrained word embeddings, which may limit generalization for complex linguistic patterns.

## Dependencies

- **None**: TrendWordNet uses **pure Go** with the standard library (`math`, `math/rand/v2`, `strings`, `encoding/csv`, `encoding/gob`).

## License
[MIT License](LICENSE)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature enhancements, or documentation improvements.

## Acknowledgments
This project was inspired by the need to classify trendy words in social media and text analysis applications. Thanks to the Go community for providing a robust standard library, enabling a dependency-free implementation.

----

## The results of the neural network can be viewed on my telegram channel. Every day at around 9 am, a summary of trending words is issued from the received article titles, which are used to build a trending set of words, and the telegram channel is located at: [https://t.me/microgonews](https://t.me/microgonews)
