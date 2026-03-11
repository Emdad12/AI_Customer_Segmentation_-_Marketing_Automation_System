import numpy as np

def transform_features(df):

    df["monetary"] = np.log1p(df["monetary"])

    df["avg_order_value"] = np.log1p(df["avg_order_value"])

    return df