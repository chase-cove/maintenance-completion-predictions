import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import os

from utils import transform_data


# Read csv data in a pandas data frame
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data/task_training_data.csv")
)


# create columns with the correct data types
# split features from label
X, y = transform_data(data)

# split some data to test with
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Scikit-learn (sklearn) is an open-source machine learning library for Python
# standardize data
# Removing the mean (centering) — subtract mean
# Scaling to unit variance (normalization) — divide by std -> std of 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# sequential neural network
# main keras model
# linear, one layer is added at a time
model = Sequential()

# Rectified Linear Unit activation function
# hidden layers, introduces non-linearity to the model
# each neuron in a dense layer is connected to
# every neuron in the preceding layer
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))

# part of the output layer
# transforms final raw output into a value between 0 and 1
model.add(Dense(1, activation="sigmoid"))

# configure the model for training
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# monitors a metric and stops training if metric is no longer improving
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# train neural network with data
# gradient descent — iteratively adjust weights in the direction that decreases losses
# process 32 items before adjusting weights
# train on all data up to 20 times
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


model.save("maintenance_task_model.keras")
