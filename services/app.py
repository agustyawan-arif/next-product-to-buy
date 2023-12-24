import os, sys
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

path_this = os.path.abspath(os.path.dirname(__file__))
path_engine = os.path.join(path_this, "..")
sys.path.append(path_this)
sys.path.append(path_engine)

from src.inference import Inference

app = FastAPI()

# Create an instance of Inference
inference = Inference()
inference.load_tokenizer()
inference.load_model()

class PredictionRequest(BaseModel):
    new_sequence: list

@app.post("/predict")
def predict_next_item(request: PredictionRequest):
    try:
        # Example of predicting the next item for a new items sequence
        new_sequence = request.new_sequence
        predicted_product = inference.predict_next_item(new_sequence)

        # Return the predicted product name
        return {"predicted_product": predicted_product}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app with UVicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

