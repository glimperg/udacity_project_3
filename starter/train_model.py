# Script to train machine learning model.
import csv
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, model_slice_performance

# Add code to load in the data.
data = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=420)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
print("Model trained successfully, saved model and encoder in model directory.")

# Compute overall model performance
preds = inference(model, X_test)
metrics = compute_model_metrics(y_test, preds)
print(f"Model metrics: precision: {metrics[0]}, recall: {metrics[1]}, fbeta: {metrics[2]}")

# Compute model slice performance, save to txt
header = ["value", "precision", "recall", "fbeta"]
with open("model/slice_output.txt", "w") as out_file:
    writer = csv.writer(out_file)
    for feature in cat_features:
        performance = model_slice_performance(feature, test, y_test, preds)
        out_file.write(f"Feature: {feature}\n")
        writer.writerow(header)
        for value in performance:
            precision, recall, fbeta = performance[value].values()
            writer.writerow([value, precision, recall, fbeta])
print("Saved model slice performance output to model/slice_output.txt.")
