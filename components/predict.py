import pickle
import pandas as pd
import sys

from utils.exception import securityException
from utils.logger import logger

try:
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
        logger.info("Model loaded successfully")
except Exception as e:
    logger.exception("Failed to load model")
    raise securityException("Failed to load model", sys)



#  Add human-readable label mapping
label_map = {0: 'low', 1: 'medium', 2: 'high'}

#  Use model.classes_ and map to readable labels if needed
numeric_classes = model.classes_
class_labels = [label_map.get(c, str(c)) for c in numeric_classes]

def predict_output(user_input: dict):
    try:
        df = pd.DataFrame([user_input])
        predicted_class = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        confidence = max(probabilities)

        #  Convert predicted class to readable label
        predicted_label = label_map.get(predicted_class, str(predicted_class))

        # Convert probabilities to label:probability format
        class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))

        logger.info(f"Prediction: {predicted_label}, Confidence: {confidence}")

        return {
            "predicted_category": predicted_label,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probs
        }
    except Exception as e:
        logger.exception("Prediction failed")
        raise securityException("Prediction process failed", sys)
