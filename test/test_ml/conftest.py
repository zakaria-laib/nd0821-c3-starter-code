import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data():
    df = pd.read_csv("data/census_clean.csv")
    return df


@pytest.fixture(scope="session")
def categorical_features():
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
    return cat_features
