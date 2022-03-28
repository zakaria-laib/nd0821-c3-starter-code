# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference
from ml.model import compute_model_slice_metrics

# Add code to load in the data.


# Optional enhancement, use K-fold cross validation instead of a
# train-test split.

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
if __name__ == '__main__':

    print("Data ingestion step")
    data = pd.read_csv("data/census_clean.csv")
    print("Data segregation step")
    train, test = train_test_split(data, test_size=0.3)

    print("Data preprocessing step for train set")
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Proces the test data with the process_data function.
    print("Data preprocessing step for test set")
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train and save a model.
    print("Train model step")
    model = train_model(X_train, y_train)

    print("Inference step")
    preds = inference(model, X_test)

    print("Scoring step")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"fbeta: {fbeta}")

    print("Save model")
    joblib.dump(model, "model/model.pkl")

    print("Save one-hot encoder")
    joblib.dump(encoder, "model/encoder.pkl")

    print("Save label encoder")
    joblib.dump(lb, "model/lb.pkl")

    # Computing performance on model slices
    print("Compute model slice metrics")
    input_pth = "model"
    output_pth = "data"
    compute_model_slice_metrics(
        test,
        cat_features,
        input_pth,
        output_pth,
        process_data)
