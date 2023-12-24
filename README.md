# Next Product to Buy

This project focuses on predicting the next product a customer is likely to buy based on their purchase history using a neural network.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
   - [Trainer](#trainer)
   - [Inference](#inference)
4. [Configuration](#configuration)
   - [config.conf](#configconf)

## Introduction

Welcome to the Next Product to Buy project! This project leverages neural networks to predict the next product a customer is likely to purchase based on their historical buying patterns. By analyzing sequences of purchased products, the model learns patterns and dependencies, offering valuable insights into potential future purchases.

### Key Features

- **Neural Network Model:** Utilizes a sequential model with an embedding layer, LSTM layer, and dense layer to capture intricate patterns in customer purchase sequences.
- **Training and Inference:** Separate scripts (`trainer.py` and `inference.py`) for model training and making predictions, allowing for flexibility and scalability.
- **Configuration:** Easily customizable through the `config.conf` file, enabling adjustments to paths, model parameters, and training settings.
- **Metrics and Logging:** Utilizes Weights & Biases (W&B) for tracking and logging metrics during model training.

This project serves as a powerful tool for businesses looking to enhance their understanding of customer behaviors and improve recommendation systems. Whether you are exploring machine learning or seeking predictive analytics for your e-commerce platform, Next Product to Buy provides a foundation for building intelligent recommendation systems.

Feel free to explore the project structure, try out the trainer and inference scripts, and customize the configuration to suit your specific use case. We welcome contributions, feedback, and collaboration to further enhance the capabilities of this predictive modeling project.

Happy predicting!

## Project Structure

The project is organized into the following structure:

- `src/`
  - `trainer.py`: Script for training the neural network model.
  - `inference.py`: Script for making predictions using the trained model.
  - `config.conf`: Configuration file for specifying paths, model parameters, and training settings.

## Usage

### Trainer

To train the model, run the `trainer.py` script. The script uses the configuration specified in `config.conf`.

```bash
python trainer.py
```

### Inference

To make predictions using the trained model, run the `inference.py` script. The script loads the trained model and tokenizer specified in the configuration.

```bash
python inference.py
```

### Configuration

#### config.conf

The `config.conf` file contains various settings for the model, training, and paths. Customize these settings as needed.

```ini
[paths]
data = /path/to/your/purchase_history.csv

[model]
embedding_output_dim = 10
lstm_units = 200
dense_units = 1
optimizer = adam
loss_function = mean_squared_error

[training]
epochs = 200
batch_size = 1

[wandb]
project = next_product_to_buy
```

Adjust the paths, model parameters, and training settings according to your requirements.