from src.inference import Inference

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
