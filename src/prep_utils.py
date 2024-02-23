import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def encode_sin_cos(X) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        X[col + "_sin"] = np.sin(X[col])
        X[col + "_cos"] = np.cos(X[col])
        X = X.drop(col, axis=1, inplace=False)

    return X


def encode_log(X) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        X[col + "_log"] = np.log(X[col] + 1.0)
        X = X.drop(col, axis=1, inplace=False)

    return X


def scale_numerical(X):
    """Scale the numerical columns of the dataframe.
    """
    X = X.copy()
    cols = X.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler().set_output(transform='pandas')
    X[cols] = scaler.fit_transform(X[cols])
    return X
