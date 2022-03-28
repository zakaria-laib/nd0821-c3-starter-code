import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import pytest
import pandas as pd
import time
from src.ml.data import process_data
from src.ml.model import (
    train_model,
    compute_model_metrics,
    compute_model_slice_metrics,
    inference)


@pytest.fixture(scope="session")
def make_dataset():
    data = {"column1": [1, 1, 1, 1, 1, 2, 1],
            "column2": [0.1, 0.1, 0.5, 0.5, 0.1, 0.3, 0.1],
            "column3": ["X", "X", "Y", "Y", "X", "X", "Y"],
            "salary": [0, 1, 1, 1, 0, 0, 0]}
    df = pd.DataFrame(data)
    X, y, encoder, lb = process_data(df,
                                     ["column3"],
                                     "salary",
                                     training=True)
    joblib.dump(encoder, "model/encoder.pkl")
    joblib.dump(lb, "model/lb.pkl")
    return X, y, df


@pytest.fixture(scope="session")
def make_split(make_dataset):
    X, y, _ = make_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def test_train_model(make_split):
    # testing train_model
    X_train, _, y_train, _ = make_split
    model = train_model(X_train, y_train)
    joblib.dump(model, "model/model.pkl")
    assert isinstance(model, sklearn.ensemble._gb.GradientBoostingClassifier)
    time.sleep(5)


def test_inference(make_split):
    # Testing inference
    _, X_test, _, _ = make_split
    model = joblib.load("model/model.pkl")
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)


def test_compute_model_metrics(make_split):
    # Testing compute_model_metrics
    _, X_test, _, y_test = make_split
    model = joblib.load("model/model.pkl")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_compute_model_slice_metrics(make_dataset):
    # Testing compute_model_slice_metrics
    _, _, df = make_dataset
    input_pth = "model"
    output_pth = "data"
    compute_model_slice_metrics(df, ["column3"], input_pth, output_pth, process_data)
    assert os.path.exists(output_pth + "/slice_output.txt")
    assert os.stat(output_pth + "/slice_output.txt").st_size > 0.0


def test_clean_dummy_files():
    input_pth = "model"
    output_pth = "data"
    # os.remove(input_pth + "/encoder.pkl")
    # os.remove(input_pth + "/lb.pkl")
    # os.remove(input_pth + "/model.pkl")
    # os.remove(output_pth + "/slice_output.txt")
    assert os.path.exists(input_pth + "/encoder.pkl")
    assert os.path.exists(input_pth + "/lb.pkl")
    assert os.path.exists(input_pth + "/model.pkl")
    assert os.path.exists(output_pth + "/slice_output.txt")
