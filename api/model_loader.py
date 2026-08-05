import os
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.onnx")

session = ort.InferenceSession(MODEL_PATH)

def predict(X: np.ndarray) -> tuple[float, int]:
    probabilities = np.array(session.run(["output"], {"input": X})[0])
    probability   = float(probabilities[0][0])
    prediction    = int(probability >= 0.5)

    return probability, prediction