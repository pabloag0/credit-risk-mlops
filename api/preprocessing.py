import json
import os
import numpy as np
import pandas as pd

METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "preprocessing_metadata.json")

with open(METADATA_PATH) as f:
    _meta = json.load(f)

COLUMNS = _meta["columns"]
MEANS = _meta["means"]
STDS = _meta["stds"]
CATEGORICAL_COLS = _meta["categorical_cols"]
DROP_FIRST = _meta["drop_first"]


def preprocess_input(data: dict) -> np.ndarray:
    df = pd.DataFrame(index=[0], columns=COLUMNS, data=0.0)

    for key, value in data.items():
        if key in df.columns:
            df.loc[0, key] = value
            continue

        dummy_col = f"{key}_{value}"
        if dummy_col in df.columns:
            df.loc[0, dummy_col] = 1.0

    for col in COLUMNS:
        df[col] = (df[col] - MEANS[col]) / STDS[col]

    return df.values.astype("float32")
