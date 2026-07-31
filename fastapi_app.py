# fastapi_app.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import json

app = FastAPI()

# Load model and config on startup
with open('models/inference_config.json', 'r') as f:
    config = json.load(f)
model = load_model(config['model_path'])

def preprocess(image_bytes):
    img = Image.open(BytesIO(image_bytes)).resize(tuple(config['input_size']))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_tensor = preprocess(contents)
    prob = model.predict(input_tensor, verbose=0)[0][0]
    threshold = config['threshold']
    pred = int(prob >= threshold)
    label = config['class_names'][pred]
    confidence = float(prob if pred == 1 else 1 - prob)
    return {"prediction": label, "confidence": confidence, "raw_score": float(prob)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)