from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=420)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def model_slice_performance(feature, test_set, y_test, preds):
    """
    Compute performance metrics on the model slice defined by the input feature.

    Inputs
    ------
    feature: str
        Name of the feature to be held fixed.
    test_set: df
        Unprocessed test dataset.
    y_test: np.array
        Processed test labels.
    preds: np.array
        Output test predictions.
    Returns
    -------
    performance: dict
        Dictionary containing performance metrics for each value of the feature.
    """
    performance = {}
    for value in test_set[feature].unique():
        mask_value = test_set[feature] == value
        precision, recall, fbeta = compute_model_metrics(y_test[mask_value], preds[mask_value])
        performance[value] = {"precision": precision, "recall": recall, "fbeta": fbeta}
    return performance
