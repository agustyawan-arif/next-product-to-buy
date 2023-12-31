# Next Product to Buy

This project focuses on predicting the next product a customer is likely to buy based on their purchase history using a neural network.

![Simulation](assets/simulation_preview.gif)

## Table of Contents

1. [Introduction](#introduction)
2. [Training Results](#training-results)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
   - [Trainer](#trainer)
   - [Inference](#inference)
5. [Configuration](#configuration)
   - [config.conf](#configconf)

## Introduction

Welcome to the Next Product to Buy project! This project leverages neural networks to predict the next product a customer is likely to purchase based on their historical buying patterns. By analyzing sequences of purchased products, the model learns patterns and dependencies, offering valuable insights into potential future purchases.

### Key Features

- **Neural Network Model:** Utilizes a sequential model with an embedding layer, LSTM layer, and dense layer to capture intricate patterns in customer purchase sequences.
- **Training and Inference:** Separate scripts (`trainer.py` and `inference.py`) for model training and making predictions, allowing for flexibility and scalability.
- **Configuration:** Easily customizable through the `config.conf` file, enabling adjustments to paths, model parameters, and training settings.
- **Metrics and Logging:** Utilizes Weights & Biases (W&B) for tracking and logging metrics during model training.

### Limitations

- **Supported Products:** The model currently supports predictions for nine specific products:
  1. Samsung Galaxy S21
  2. HP Wireless Mouse
  3. Dell XPS 13
  4. JBL Flip 5
  5. Nintendo Switch
  6. Sony Noise-Cancelling Headphones
  7. Acer Predator Helios
  8. Playstation 5
  9. Xiaomi Mi 11

Please note that the model is trained on data specific to these products, and predictions for other products may not yield accurate results.

This project serves as a powerful tool for businesses looking to enhance their understanding of customer behaviors and improve recommendation systems. Whether you are exploring machine learning or seeking predictive analytics for your e-commerce platform, Next Product to Buy provides a foundation for building intelligent recommendation systems.

Feel free to explore the project structure, try out the trainer and inference scripts, and customize the configuration to suit your specific use case. We welcome contributions, feedback, and collaboration to further enhance the capabilities of this predictive modeling project.

Happy predicting!

## Training Results

The model was trained with the results are summarized below:

- **Created:** Dec 24 '23 09:50
- **Runtime:** 36s
- **End Time:** Dec 24 '23 09:50
- **Updated:** Dec 24 '23 09:50

![Training Loss](assets/training_plot.png)

### Training Metrics

| Created          | Runtime | End Time          | Updated           | Accuracy | Epoch | F1-Score | Loss   | Precision | Recall |
| ---------------- | ------- | ----------------- | ----------------- | -------- | ----- | -------- | ------ | --------- | ------ |
| Dec 24 '23 09:50 | 36s     | Dec 24 '23 09:50 | Dec 24 '23 09:50 | 0.8      | 199   | 0.7991   | 0.04277 | 0.8087    | 0.8    |

The plot above illustrates the training loss over epochs. The table presents key metrics achieved during the training process.

## Project Structure

The project is organized into the following structure:

- `src/`
  - `trainer.py`: Script for training the neural network model.
  - `inference.py`: Module for making predictions using the trained model.
  - `config.conf`: Configuration file for specifying paths, model parameters, and training settings.
- `wrapper.py`: Script for making predictions using the trained model.

## Usage

### Trainer

To train the model, run the `trainer.py` script. The script uses the configuration specified in `config.conf`.

```bash
python trainer.py
```

### Inference

To make predictions using the trained model, run the `wrapper.py` script. The script loads the trained model and tokenizer specified in the configuration.

```bash
python wrapper.py
```

Or import the module

```python
from src.inference import Inference
# Create an instance of Inference
inference = Inference()

# Load the tokenizer and model
inference.load_tokenizer()
inference.load_model()

# Example of predicting the next item for a new items sequence
new_sequence = ['jbl flip 5', 'playstation 5', 'samsung galaxy s21', 'hp wireless mouse', 'acer predator helios']
predicted_product = inference.predict_next_item(new_sequence)

print(f"Predicted next item: {predicted_product}")
```
### API & Simulation

For API and Simulation could be read at [Wiki's repo](https://github.com/agustyawan-arif/next-product-to-buy/wiki/7.-How-to-Use#run-api)

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
