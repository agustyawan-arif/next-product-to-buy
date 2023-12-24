import json
import configparser
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
import wandb
import os, sys

path_this_file = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_this_file)

class Trainer:
    def __init__(self, config_path='config.conf'):
        """
        Initializes the class instance.

        Args:
            config_path (str): The path to the configuration file. Default is 'config.conf'.

        Returns:
            None
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.data = None
        self.tokenizer = None
        self.model = None

    def load_data(self):
        """
        Load the data from a CSV file and process it.

        This function reads the data from a CSV file specified in the configuration and sorts it by the 'purchase_id' column. It then groups the data by 'customer_id' and for each group, it sorts the data by 'purchase_id' and extracts the 'product_id' values as a list. The resulting data is stored in the 'data' attribute of the current object.

        Parameters:
            self (object): The current object.

        Returns:
            None
        """
        csv_file_path = self.config.get('paths', 'data')
        self.data = pd.read_csv(csv_file_path).sort_values(by='purchase_id')
        self.data = self.data.groupby('customer_id').apply(
            lambda group: group.sort_values(by='purchase_id')['product_id'].tolist()).tolist()

    def preprocess_data(self):
        """
        Preprocesses the data by performing the following steps:
        1. Flattens the nested data structure.
        2. Initializes a Tokenizer.
        3. Fits the Tokenizer on the flattened data.
        4. Converts each sequence in the data to a sequence of tokenized integers.
        5. Separates the input sequences (X) and the target values (y).
        6. Pads the input sequences to ensure they have the same length.
        
        Returns:
            X_padded (numpy.ndarray): The padded input sequences.
            y (numpy.ndarray): The target values.
        """
        flat_data = [item for sublist in self.data for item in sublist]
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(flat_data)
        sequences = [self.tokenizer.texts_to_sequences(seq) for seq in self.data]
        X = np.array([seq[:-1] for seq in sequences])
        y = np.array([seq[-1][0] for seq in sequences])
        X_padded = pad_sequences(X)
        return X_padded, y

    def build_model(self):
        """
        Builds the model for the neural network.

        This function initializes the model and adds the necessary layers to it based on the configuration
        parameters specified in the `config` file. The model consists of an embedding layer, an LSTM layer,
        and a dense layer. The optimizer and loss function for the model are also specified in the `config`
        file.

        Parameters:
            None

        Returns:
            None
        """
        embedding_dim = self.config.getint('model', 'embedding_output_dim')
        lstm_units = self.config.getint('model', 'lstm_units')
        dense_units = self.config.getint('model', 'dense_units')

        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=embedding_dim,
                                 input_length=X_padded.shape[1]))
        self.model.add(LSTM(lstm_units))
        self.model.add(Dense(dense_units, activation='linear'))
        optimizer = self.config.get('model', 'optimizer')
        loss_function = self.config.get('model', 'loss_function')
        self.model.compile(optimizer=optimizer, loss=loss_function)

    def train_model(self):
        """
        Train the model.

        Returns:
            history: The training history.
        """
        wandb.init(project=self.config.get('wandb', 'project'))
        wandb_callback = wandb.keras.WandbCallback()
        epochs = self.config.getint('training', 'epochs')
        batch_size = self.config.getint('training', 'batch_size')
        history = self.model.fit(X_padded, y, epochs=epochs, batch_size=batch_size, callbacks=[wandb_callback])
        # wandb.finish()
        return history

    def evaluate_model(self):
        """
        Evaluates the model by making predictions on the data and calculating various metrics.
        
        Parameters:
        - None
        
        Returns:
        - None
        
        This function evaluates the model by making predictions on the data stored in the 'data' attribute. It iterates over each data point, preprocesses the input sequence, and uses the model to make a prediction. The predicted product is then added to the 'pred' list and the actual product is added to the 'ref' list. 
        
        After iterating over all the data points, the function creates a label mapping dictionary that maps each unique product to a numerical index. The 'pred_list' and 'ref_list' are converted to numerical representations using this label mapping. 
        
        Finally, the function calculates various evaluation metrics such as accuracy, precision, recall, and F1 score using the 'num_pred_list' and 'num_ref_list'. These metrics are printed to the console and logged using Weights & Biases (wandb) for easy tracking and visualization.
        """
        pred, ref = [], []
        for cd in self.data:
            try:
                new_sequence = cd[:5]
                new_sequence_int = self.tokenizer.texts_to_sequences([new_sequence])
                new_sequence_padded = pad_sequences(new_sequence_int, maxlen=X_padded.shape[1])

                prediction = self.model.predict(new_sequence_padded)
                predicted_product = self.tokenizer.index_word[int(prediction[0][0])]

                ref.append(cd[-1])
                pred.append(predicted_product)
            except:
                pass

        pred_list = pred
        ref_list = ref

        label_mapping = {product: idx for idx, product in enumerate(set(pred_list + ref_list))}
        num_pred_list = [label_mapping[product] for product in pred_list]
        num_ref_list = [label_mapping[product] for product in ref_list]

        accuracy = accuracy_score(num_ref_list, num_pred_list)
        precision = precision_score(num_ref_list, num_pred_list, average='weighted')
        recall = recall_score(num_ref_list, num_pred_list, average='weighted')
        f1 = f1_score(num_ref_list, num_pred_list, average='weighted')

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        wandb.log({
            "final_loss": history.history['loss'][-1],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1
        })
        wandb.finish()

    def save_model_and_tokenizer(self, model_path='models/model.keras',
                                 tokenizer_config_path='models/tokenizer_config.json'):
        """
        Save the model and tokenizer to the specified paths.

        Parameters:
            model_path (str): The path to save the model. Defaults to 'models/model.keras'.
            tokenizer_config_path (str): The path to save the tokenizer configuration. Defaults to 'models/tokenizer_config.json'.

        Returns:
            None
        """
        if not os.path.exists(os.path.join(path_this_file, "models")):
            os.makedirs(os.path.join(path_this_file, "models"))
        self.model.save(model_path)
        tokenizer_config = self.tokenizer.to_json()
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_data()
    X_padded, y = trainer.preprocess_data()
    trainer.build_model()
    history = trainer.train_model()
    trainer.evaluate_model()
    trainer.save_model_and_tokenizer()