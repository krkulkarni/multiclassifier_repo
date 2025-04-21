# src/data_utils.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union

def zscore_finite(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Applies z-scoring column-wise, ignoring NaNs.

    Args:
        X: Input data (numpy array or pandas DataFrame).

    Returns:
        Z-scored data in the same format as input.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        df_index = X.index
        df_columns = X.columns
        X_array = X.values.astype(float)
    else:
        X_array = np.array(X).astype(float)

    Xz = np.full(X_array.shape, np.nan)
    for i, col_data in enumerate(X_array.T):
        finite_mask = np.isfinite(col_data)
        if np.any(finite_mask): # Check if there are any finite values
            finite_vals = col_data[finite_mask]
            # Check for zero standard deviation
            if np.std(finite_vals) > 1e-9: # Avoid division by zero/near-zero
                 Xz[finite_mask, i] = stats.zscore(finite_vals)
            else:
                 Xz[finite_mask, i] = 0.0 # Assign 0 if std dev is zero
        # Else: leave column as NaNs if no finite values

    if is_dataframe:
        return pd.DataFrame(Xz, index=df_index, columns=df_columns)
    else:
        return Xz

# Functions requiring the stanford_df lookup table
def get_coord_from_df(df: pd.DataFrame, parcel_name: str, name_col: str = 'stanford_name', coord_col: str = 'coordinates') -> tuple:
    """Gets coordinates for a parcel name from a lookup DataFrame."""
    from ast import literal_eval # Import locally
    try:
        coord_str = df.loc[df[name_col] == parcel_name, coord_col].iloc[0]
        return literal_eval(coord_str)
    except (IndexError, KeyError):
        print(f"Warning: Could not find coordinates for parcel '{parcel_name}' in DataFrame.")
        return (np.nan, np.nan, np.nan) # Return NaNs or raise error

def get_index_from_df(df: pd.DataFrame, parcel_name: str, name_col: str = 'stanford_name', index_col: str = 'parcel_ind') -> int:
    """Gets an index for a parcel name from a lookup DataFrame."""
    try:
        return df.loc[df[name_col] == parcel_name, index_col].iloc[0]
    except (IndexError, KeyError):
        print(f"Warning: Could not find index for parcel '{parcel_name}' in DataFrame.")
        return -1 # Return sentinel value or raise error

def get_description_from_df(df: pd.DataFrame, parcel_name: str, name_col: str = 'stanford_name', desc_col: str = 'description') -> str:
    """Gets a description for a parcel name from a lookup DataFrame."""
    try:
        return df.loc[df[name_col] == parcel_name, desc_col].iloc[0]
    except (IndexError, KeyError):
        print(f"Warning: Could not find description for parcel '{parcel_name}' in DataFrame.")
        return "N/A"