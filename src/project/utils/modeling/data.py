import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FitRecDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_columns: list,
        categorical_columns: list,
        heartrate_input_column: str,
        heartrate_output_column: str,
        workout_id_column: str = "id",
        window_size: int = 10,
        use_heartrate_input: bool = True,
    ):
        """Dataset for creating sliding windows from workout data.

        Args:
            df (pd.DataFrame): DataFrame containing the workout data.
            numerical_columns (list): List of numerical feature column names.
            categorical_columns (list): List of categorical feature column names.
            heartrate_input_column (str): Name of the column containing input heart rate data (standardized).
            heartrate_output_column (str): Name of the column containing output heart rate data (unstandardized).
            workout_id_column (str, optional): Name of the column containing workout IDs. Defaults to "id".
            window_size (int, optional): Size of the sliding window. Defaults to 10.
            use_heartrate_input (bool, optional): Whether to use heart rate input data for autoregressive modeling.
                Defaults to True.
        """
        # Since all the workout sequences have identical lengths (300),
        # we can exploit this to create sliding windows efficiently.
        #
        # We first stack all the workout feature sequences together, to form [N, n_features] array.
        # We then pad it from the left, since we will be creating sliding windows
        # right from t = 0 and our window length = 10 (so for the first 9 datapoints, they will be mostly 0s).
        # Finally, we create sliding windows by slicing the padded array.
        #
        # For static features like user, sport and gender ids, we do not duplicate them for each window.
        # Instead, we create a mapping from workout_id to these static features,
        # and then duplicate only workout ids when creating the windows.
        # at runtime, we take the workout id and lookup the static features.

        sequence_length = len(df.iloc[0][heartrate_output_column])

        # STEP 1: Stack data arrays

        numerical_values_array = np.stack(
            [np.stack(df[col]) for col in numerical_columns], axis=-1
        )
        heartrate_input_array = np.stack(df[heartrate_input_column])
        heartrate_output_array = np.stack(df[heartrate_output_column])
        workout_id_array = np.stack(df["id"])

        # STEP 2: Pre-pad the arrays
        padding_config_num = ((0, 0), (window_size - 1, 0), (0, 0))
        padding_config_target_input = ((0, 0), (window_size, 0))

        numerical_values_array = np.pad(
            numerical_values_array,
            padding_config_num,
            mode="constant",
            constant_values=0,
        )

        heartrate_input_array = np.pad(
            heartrate_input_array,
            padding_config_target_input,
            mode="constant",
            constant_values=0,
        )

        # STEP 2: Create workout_id to static feature mapping

        self.workout_id_to_indices = {}
        for row in df[[workout_id_column] + categorical_columns].values:
            self.workout_id_to_indices[row[0]] = [
                row[i + 1] for i in range(len(categorical_columns))
            ]

        # STEP 3: Create sliding windows

        numerical_datapoints = []
        heartrate_output_datapoints = []

        for t in range(window_size, sequence_length + window_size):
            lower_index = t - window_size
            upper_index = t

            num_slice = numerical_values_array[:, lower_index:upper_index, :]
            heartrate_output_value = heartrate_output_array[:, lower_index]

            if use_heartrate_input:
                heartrate_input_slice = heartrate_input_array[
                    :, lower_index:upper_index
                ]
                num_slice = np.concatenate(
                    [num_slice, heartrate_input_slice[:, :, np.newaxis]], axis=-1
                )

            numerical_datapoints.append(num_slice)
            heartrate_output_datapoints.append(heartrate_output_value)

        self.X = np.vstack(numerical_datapoints)
        self.y = np.hstack(heartrate_output_datapoints)
        self.X_ids = np.tile(workout_id_array, sequence_length)

        # Step 4: Cleanup

        del numerical_values_array, heartrate_input_array, heartrate_output_array
        del numerical_datapoints, heartrate_output_datapoints, workout_id_array

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        user_idx, sport_idx, gender_idx = self.workout_id_to_indices[self.X_ids[idx]]
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(sport_idx, dtype=torch.long),
            torch.tensor(gender_idx, dtype=torch.long),
        )
