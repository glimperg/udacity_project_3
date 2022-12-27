import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference, model_slice_performance


@pytest.fixture
def dataset():
    """
    Toy dataset to test ML methods.
    """
    df = pd.DataFrame(
        {
            "a": [0, 1, 2, 3],
            "b": ["w", "x", "y", "z"],
            "salary": [">50K", "<=50K", "<=50K", ">50K"]
        }
    )
    train, test = train_test_split(df)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=["a", "b"],
        label="salary",
        training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=["a", "b"],
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    return X_train, X_test, y_train, y_test, test


def test_train_model(dataset):
    """
    Test if the model trains correctly on the toy dataset.
    """
    X_train, X_test, y_train, y_test, test = dataset
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Output model is not a RandomForestClassifier"


def test_inference(dataset):
    """
    Test running inference using a trained model.
    """
    X_train, X_test, y_train, y_test, test = dataset
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray), "Model inference output is not a numpy array"
    assert len(X_test) == len(preds), "Prediction output size differs from input data size"


def test_compute_model_metrics(dataset):
    """
    Test model metrics computation by comparing model predictions to the test labels.
    """
    X_train, X_test, y_train, y_test, test = dataset
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)    
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float), "Output precision is not a float"
    assert isinstance(recall, float), "Output recall is not a float"
    assert isinstance(fbeta, float), "Output fbeta is not a float"


def test_model_slice_performance(dataset):
    """
    Test method to compute performance metrics on one model slice.
    """
    X_train, X_test, y_train, y_test, test = dataset
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    feature = test.columns[0]
    performance = model_slice_performance(feature, test, y_test, preds)
    assert isinstance(performance, dict), "Output performance is not a dict"
