import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data(path: str):
    df = pd.read_parquet(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def remove_high_corr_features(df: pd.DataFrame):
    removed_features = [
        "market_rocp", "market_apo", "market_macdsignal_macdfix",
        "market_macd_macdfix", "borrow_block_number", "borrow_timestamp",
        "risky_first_tx_timestamp", "market_macd_macdext", "market_macd",
        "market_macdsignal", "liquidation_count"
    ]
    to_remove = [f for f in removed_features if f in df.columns and f != 'wallet_address']
    df = df.drop(columns=to_remove, errors="ignore")
    return df

def split_data(df: pd.DataFrame, random_state: int = 42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train, temp = train_test_split(df, test_size=0.5, random_state=random_state)
    val, test = train_test_split(temp, test_size=0.5, random_state=random_state)
    return train, val, test

def prepare_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['target', 'wallet_address']
    return [col for col in df.columns if col not in exclude_cols]

def fill_missing_with_median(X_train, X_val, X_test):
    for col in X_train.columns:
        if X_train[col].isna().any():
            med = X_train[col].median()
            X_train[col].fillna(med, inplace=True)
            X_val[col].fillna(med, inplace=True)
            X_test[col].fillna(med, inplace=True)
    return X_train, X_val, X_test
