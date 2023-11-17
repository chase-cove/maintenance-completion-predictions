import pandas as pd
from datetime import datetime, timezone
from tensorflow.keras.models import load_model

import os


def get_work_completed(row):
    """
    Calculates the percentage of work completed based on the current labor hours and estimated hours.
    """
    current_labor_hours = row["currentLaborHours"]
    estimated_hours = row["estimatedHours"]

    return (
        (current_labor_hours / estimated_hours)
        if estimated_hours and estimated_hours > 0
        else 1
    )

def get_hope_index(row):
    """
    Calculates the hope index for a given maintenance task.
    """

    completed_at = row["completedOn"]
    due_date = row["dueDate"]
    is_completed = completed_at is not pd.NaT
    percent_work_complete = row["percentWorkCompleted"]
    available_work_hours = (
        row["totalWorkHoursAvailable"] if row["totalWorkHoursAvailable"] > 0 else 24
    )

    # due_date - completed_at

    # if not is_completed:
    #     return 0

    # if is_completed and completed_at > due_date:
    #     variance = (
    #         1 - (completed_at - due_date).total_seconds() / 3600 / available_work_hours
    #     )
    #     # return max(min(variance, 1), -1)
    #     return max(percent_work_complete, 1)

    # percent_hours_left = (
    #     (row["hoursLeftToWork"] / row["totalWorkHoursAvailable"])
    #     if row["totalWorkHoursAvailable"] > 0 and row["hoursLeftToWork"] > 0
    #     else 0.5
    # )

    # average_val = (percent_work_complete  + percent_hours_left) / 2

    # return 2 if is_completed else 1 +  max(average_val , 1)


def transform_data(df):
    now = datetime.now(timezone.utc)

    # Extract features and labels
    X = df.drop("isCompletedOnTime", axis=1)
    y = df["isCompletedOnTime"].astype(int)

    # median_date = X['completedOn'].median()



    X["estimatedHours"] = X["estimatedHours"].astype(float)
    X["currentLaborHours"] = X["currentLaborHours"].astype(float)

    X["dueDate"] = pd.to_datetime(X["dueDate"], utc=True)
    X["createdAt"] = pd.to_datetime(X["createdAt"], utc=True)
    X["completedOn"] = pd.to_datetime(X["completedOn"], utc=True)

    X["totalWorkHoursAvailable"] = (
        X["dueDate"] - X["createdAt"]
    ).dt.total_seconds() / 3600

    X["hoursLeftToWork"] = X["estimatedHours"] - X["currentLaborHours"]
    X["percentWorkCompleted"] = X.apply(get_work_completed, axis=1)
    # X["hopeIndex"] = X.apply(get_hope_index, axis=1)
    # X["hoursLeftUntilTaskDue"] = (X["dueDate"] - now).dt.total_seconds() / 3600

    # def get_completed_before_due(row):
    #     return (row['completedOn'] - row['dueDate']).dt.total_seconds() if  row['completedOn'] and row['dueDate']  else 0

    # X['completedBeforeDue'] = X.apply(get_completed_before_due, axis=1)
    # One-hot encode categorical columns
    # X = pd.get_dummies(X, columns=['categoryName'], dtype=int)




    X = X.drop(
        [
            "dueDate",
            "completedOn",
            "name",
            "createdAt",
            "siteName",
            "categoryName",
        ],
        axis=1,
    )

    # X = X.dropna(axis=1, how='all')



    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(X.describe())

    return X, y


def make_prediction(tasks_data):
    """
    Makes predictions on maintenance tasks completion time using a pre-trained model.
    """

    loaded_model = load_model(
        os.path.join(os.path.dirname(__file__), "../../maintenance_task_model5.keras")
    )

    df = pd.DataFrame(tasks_data)
    X, _ = transform_data(df)

    predictions = loaded_model.predict(X)
    flattened_prediction = [item for sublist in predictions for item in sublist]

    print("@#$ prediction", flattened_prediction)
    return flattened_prediction
