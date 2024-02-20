import pandas as pd
import numpy as np


def encode_sin_cos(X, cols: str) -> pd.DataFrame:
    X = X.copy()
    for col in cols:
        X[col + "_sin"] = np.sin(X[col])
        X[col + "_cos"] = np.cos(X[col])
        X = X.drop(col, axis=1, inplace=False)

    return X


def encode_log(X, cols: str) -> pd.DataFrame:
    X = X.copy()
    for col in cols:
        X[col + "_log"] = np.log(X[col] + 1.0)
        X = X.drop(col, axis=1, inplace=False)

    return X
