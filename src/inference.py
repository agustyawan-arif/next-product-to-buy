import json
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class Inference:
    def __init__(self, tokenizer_path='models/tokenizer_config.json', model_path='models/model.keras'):
        """
        Initializes the class with optional paths to the tokenizer and model files.

        Parameters:
            tokenizer_path (str): The path to the tokenizer configuration file. Defaults to 'models/tokenizer_config.json'.
            model_path (str): The path to the model file. Defaults to 'models/model.keras'.

        Returns:
            None
        """
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.loaded_tokenizer = None
        self.loaded_model = None
        self.map_items()

    def load_tokenizer(self):
        """
        Load the tokenizer from the specified tokenizer_path.
        
        Returns:
            None
        """
        with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
            loaded_tokenizer_config = json.load(f)
        self.loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_config)

    def load_model(self):
        """
        Load the model from the specified model path.

        This function loads the model from the given model path and assigns it to the `loaded_model` attribute of the class.

        Parameters:
            self (object): The object instance.
        
        Returns:
            None
        """
        self.loaded_model = load_model(self.model_path)
    
    def map_items(self):
        """
        Initializes the mapping dictionaries for product names and IDs.

        This function creates two dictionaries: `product_name_to_id` and `id_product_to_name`.
        The `product_name_to_id` dictionary maps product names to their corresponding IDs,
        while the `id_product_to_name` dictionary maps IDs to their corresponding product names.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.product_name_to_id = {
            "samsung galaxy s21": "product1",
            "hp wireless mouse": "product2",
            "dell xps 13": "product3",
            "jbl flip 5": "product4",
            "nintendo switch": "product5",
            "sony noise-cancelling headphones": "product6",
            "acer predator helios": "product7",
            "playstation 5": "product8",
            "xiaomi mi 11": "product9"
        }

        self.id_product_to_name = {v: k for k, v in self.product_name_to_id.items()}

    def predict_next_item(self, new_sequence):
        """
        Predicts the next item in a sequence.

        Args:
            new_sequence (list): A list of strings representing the new sequence of items.

        Returns:
            str: The predicted next item in the sequence.
        """
        new_sequence = [self.product_name_to_id[item] for item in new_sequence]
        new_sequence_int = self.loaded_tokenizer.texts_to_sequences([new_sequence])
        new_sequence_padded = pad_sequences(new_sequence_int, maxlen=5)

        prediction = self.loaded_model.predict(new_sequence_padded)
        predicted_product = self.loaded_tokenizer.index_word[int(prediction[0][0])]

        return self.id_product_to_name[predicted_product]

if __name__ == "__main__":
    # Create an instance of Inference
    inference = Inference()

    # Load the tokenizer and model
    inference.load_tokenizer()
    inference.load_model()

    # Example of predicting the next item for a new items sequence
    new_sequence = ['jbl flip 5', 'playstation 5', 'samsung galaxy s21', 'hp wireless mouse', 'acer predator helios']
    predicted_product = inference.predict_next_item(new_sequence)

    print(f"Predicted next item: {predicted_product}")
