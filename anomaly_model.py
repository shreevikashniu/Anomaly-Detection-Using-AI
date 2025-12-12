import pandas as pd
import numpy as np
from pyod.models.iforest import IForest

class AnomalyDetector:
    def __init__(self):
        self.model = IForest(contamination=0.05)  # 5% anomalies

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self.data

    def train_model(self):
        self.model.fit(self.data)
        return self.model

    def predict(self, input_data):
        preds = self.model.predict(input_data)
        # 0 = normal, 1 = anomaly
        return preds
