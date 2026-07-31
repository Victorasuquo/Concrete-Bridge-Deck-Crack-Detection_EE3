# app.py
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_model_and_config(model_dir='models'):
    with open(f'{model_dir}/inference_config.json', 'r') as f:
        config = json.load(f)
    model = load_model(config['model_path'])
    return model, config

def predict_image_path(path, model, config):
    img = load_img(path, target_size=tuple(config['input_size']))
    arr = img_to_array(img) / 255.0
    prob = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0][0]
    threshold = config['threshold']
    pred = int(prob >= threshold)
    label = config['class_names'][pred]
    confidence = prob if pred == 1 else 1 - prob
    return label, confidence

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python app.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    model, config = load_model_and_config()
    label, conf = predict_image_path(path, model, config)
    print(f"Image: {path}")
    print(f"Prediction: {label} (confidence: {conf:.2%})")