from sklearn.metrics import (fbeta_score,
                             precision_score, recall_score)
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import csv


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
    model = GradientBoostingClassifier(n_estimators=250, max_depth=3)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

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


def compute_model_slice_metrics(df,
                                cat_features,
                                input_pth,
                                output_pth,
                                process_data):
    """
    Validates the trained machine learning mogdel
    with slice metrics and write in text file.

    Inputs
    ------
    df : pandas.core.frame.DataFrame
         Test data for computes model slice metrics
    cat_features : list
         List with categorical features.
    Returns
    -------
    None
    """
    encoder = joblib.load(input_pth + "/encoder.pkl")
    lb = joblib.load(input_pth + "/lb.pkl")
    model = joblib.load(input_pth + "/model.pkl")
    header = ["precision", "recall", "fbeta", "feature", "value"]

    with open(output_pth + "/slice_output.txt", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for feature in cat_features:
            for value in df[feature].unique():
                df_temp = df[df[feature] == value]
                X_test, y_test, _, _ = process_data(
                    X=df_temp,
                    categorical_features=cat_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb)

                preds = inference(model, X_test)
                precision, recall, fbeta = compute_model_metrics(
                    y_test, preds)

                data = [precision, recall, fbeta, feature, value]

                writer.writerow(data)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
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
