"""
Script to train the model

Author : Moh. Rosidi
Date   : August 2021
"""
import os
import json
from joblib import dump
import logging
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics, \
    inference
from ml.data import process_data

if __name__ == '__main__':

    CWD = os.getcwd()

    logging.basicConfig(
        filename=os.path.join(
            CWD,
            'logs',
            'model.log'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w',
        level=logging.INFO,
    )

    with open(os.path.join(CWD, "starter", 'params.yaml'), 'r', encoding="UTF-8") as fp:
        CONFIG = yaml.safe_load(fp)

    CAT_FEATURES = CONFIG['categorical_features']

    DATA_FILENAME = CONFIG['data']
    DATA_DIR = os.path.join(
        CWD,
        'data'
    )
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

    PREPROCESSED_DF = pd.read_csv(DATA_PATH)
    PREPROCESSED_DF.columns = PREPROCESSED_DF.columns.str.replace(' ', '')
    TRAIN, TEST = train_test_split(
        PREPROCESSED_DF,
        test_size=0.20,
        random_state=CONFIG['random_seed']
    )

    # Feature Engineering
    X_TRAIN, Y_TRAIN, ENCODER, LABEL = process_data(
        TRAIN,
        categorical_features=CAT_FEATURES,
        label='salary',
        training=True
    )

    X_TEST, Y_TEST, _, _ = process_data(
        TEST,
        categorical_features=CAT_FEATURES,
        label='salary', training=False,
        encoder=ENCODER,
        lb=LABEL)

    logging.info("Model parameters: %s}", CONFIG['train_params'])

    MODEL = train_model(X_TRAIN, Y_TRAIN, CONFIG['train_params'])

    Y_TEST_PREDS = inference(MODEL, X_TEST)
    PRECISION, RECALL, FBETA = compute_model_metrics(Y_TEST, Y_TEST_PREDS)

    logging.info(
        'Overall model predictions - precision: %s, recall: %s, fbeta: %s',
        PRECISION, RECALL, FBETA
    )

    with open(os.path.join(CWD, "logs", 'scores.json'), "w", encoding="UTF-8") as f:
        json.dump({"precision": PRECISION,
                   "recall": RECALL,
                   "fbeta": FBETA},
                  f)

    MODEL_DIR = os.path.join(
        CWD,
        'model')

    MODEL_DEST_PATH = os.path.join(MODEL_DIR, CONFIG['model_output'])

    dump(MODEL, MODEL_DEST_PATH)
    dump(ENCODER, os.path.join(MODEL_DIR, CONFIG['encoder_output']))
    dump(LABEL, os.path.join(MODEL_DIR, CONFIG['label_binarizer_output']))
