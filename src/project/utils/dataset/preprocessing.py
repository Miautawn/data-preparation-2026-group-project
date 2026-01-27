import numpy as np
import pandas as pd
from tqdm import tqdm


class UserStandardScaler:
    def __init__(self, columns_to_standardize: list, verbose: bool = True):
        self.columns_to_standardize = columns_to_standardize
        self.user_stats = {}
        self.verbose = verbose

    def fit(self, df: pd.DataFrame):
        for user_id, group in tqdm(df.groupby("userId"), disable=not self.verbose):
            self.user_stats[user_id] = {}

            for col in self.columns_to_standardize:
                self.user_stats[user_id][col] = []

                # Concatenate all vectors for this user into one giant 1D array
                all_values = np.concatenate(group[col].values)

                self.user_stats[user_id][col].append(np.mean(all_values))
                self.user_stats[user_id][col].append(np.std(all_values) + 1e-6)

    def transform(self, df: pd.DataFrame, id_column: str = "userId") -> pd.DataFrame:
        for col in self.columns_to_standardize:
            new_col = f"{col}_standardized"
            user_means = df[id_column].map(lambda x: self.user_stats[x][col][0]).values
            user_stds = df[id_column].map(lambda x: self.user_stats[x][col][1]).values

            vals = np.array(df[col])
            vals = (vals - user_means) / user_stds

            df[new_col] = vals

        return df


class StaticFeatureOrdinalEncoder:
    def __init__(self, columns_to_encode: list, verbose: bool = True):
        self.mapping = {}
        self.columns_to_encode = columns_to_encode
        self.verbose = verbose

    def fit(self, df: pd.DataFrame):
        for col in tqdm(self.columns_to_encode, disable=not self.verbose):
            self.mapping[col] = {
                category: idx for idx, category in enumerate(df[col].unique())
            }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.columns_to_encode:
            mapping = self.mapping[col]
            new_col = f"{col}_idx"
            df[new_col] = df[col].map(mapping)

        return df
