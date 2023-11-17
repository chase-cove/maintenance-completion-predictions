import pandas as pd
from tensorflow.keras.models import load_model

import os

from utils import transform_data

# load model
loaded_model = load_model(
    os.path.join(os.path.dirname(__file__), "../../maintenance_task_model1.keras")
)

# load data
prediction_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data/prediction_data.csv")
)

X, _ = transform_data(prediction_data)


for _ in range(1, 10):
    firstResult, secondResult = loaded_model.predict(X)
    print(firstResult[0], secondResult[0])
