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

# mask = data['completedOn'] == pd.isna() or data['completedOn'] == ''
# data.drop(data[~mask].index, inplace=True)
# df = data[(data['completedOn'] != 'null')]
data = data.dropna(subset=['completedOn'])
# data['hasCompletedOn'] = data["completedOn"].apply(lambda x:  1 if  len(x) > 0 or not pd.isna(x) else 0)
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# print(data.describe())


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
# good starting point for straight-forward input
# linear, one layer is added at a time
model = Sequential()

# Rectified Linear Unit activation function
# hidden layers, introduces non-linearity to the model
# each neuron in a dense layer is connected to
# every neuron in the preceding layer
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))

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
# train on all data up to 10 times
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


testing_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data/prediction_data.csv")
)
test, _ = transform_data(testing_data)
predictions1 = model.predict(test)
predictions2 = model.predict(test)
predictions3 = model.predict(test)


def get_average_predictions(x, y, z):
    total = x[0] + y[0] + z[0]
    return total / 3


averaged_predictions = [
    get_average_predictions(x, y, z)
    for x, y, z in zip(predictions1, predictions2, predictions3)
]

print("final!@#", averaged_predictions)

if averaged_predictions[0] < 0.40 and averaged_predictions[1] > 0.80:
    model.save("maintenance_task_model7.keras")
