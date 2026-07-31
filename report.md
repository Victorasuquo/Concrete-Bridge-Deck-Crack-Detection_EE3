# Project Report

**Concrete Bridge Deck Crack Detection — Group EE3**

## Dataset

We used the [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) dataset from Kaggle, containing 40,000 images (20,000 cracked, 20,000 non-cracked) of concrete surfaces at 227 × 227 px.

## Application

Our Streamlit application accepts an uploaded image of a concrete surface and classifies it as Cracked or Not Cracked using a fine-tuned MobileNetV2 model. The model achieved 99.65% accuracy and 0.9999 ROC AUC on the held-out test set. Users upload an image and receive the prediction with a confidence score.

## Challenges

Our main challenge was training compute — we relied on Kaggle's free-tier GPU, which limited session time and hyperparameter exploration. Deploying to Streamlit Community Cloud required pinning Python 3.12, as TensorFlow does not yet support Python 3.14. We also had to use `tensorflow-cpu` to stay within the platform's 1 GB memory limit.

## Possible Improvements

With more compute, we would explore larger backbones and add Grad-CAM visualisations to the app so inspectors can see which regions the model flagged. Collecting local bridge deck images for domain-specific validation would also strengthen real-world applicability.
