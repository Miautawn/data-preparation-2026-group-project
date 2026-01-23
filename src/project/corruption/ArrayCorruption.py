import numpy as np
import pandas as pd
from jenga.basis import DataCorruption


class UnusualValueCorruption(DataCorruption):
    """
    Data corruption that shifts continuous segments to unusual values.

    This corruption identifies segments of data and shifts them to unusual values
    (much higher or lower than typical changes in the array) while preserving
    the internal differences within the segment.
    """

    def __init__(self, columns, row_fraction=1.0, segment_fraction=0.2, std_scale=2, seed=None):
        """
        Initialize WeirdValueCorruption.

        Args:
            columns (list): List of columns to corrupt -- new --
            row_fraction (float): Fraction of rows in dataset to corrupt (0.0 to 1.0)
            segment_fraction (float): Fraction of elements in each array to corrupt (0.0 to 1.0)
            std_scale (float): Scaling std for corruption (must be > 1)
                               std of the array after corruption will be std_scale * std_before
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__()

        if std_scale <= 1:
            raise ValueError(f"std_scale must be > 1, got {std_scale}")

        self.columns = columns
        self.row_fraction = row_fraction
        self.segment_fraction = segment_fraction
        self.std_scale = std_scale
        self.seed = seed

    def transform(self, data):
        corrupted_data = data.copy()

        if self.seed is not None:
            np.random.seed(self.seed)

        for col in self.columns:
            if col not in corrupted_data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        # Determine which rows to corrupt
        n_rows = len(corrupted_data)
        n_rows_to_corrupt = max(1, int(n_rows * self.row_fraction))
        # Use DataFrame index instead of numeric indices
        available_indices = corrupted_data.index.tolist()
        rows_to_corrupt = np.random.choice(available_indices, size=n_rows_to_corrupt, replace=False)

        # Apply corruption to selected rows
        for idx in rows_to_corrupt:
            for col in self.columns:
                corrupted_data.at[idx, col] = self.corrupt_array(corrupted_data.at[idx, col])

        return corrupted_data

    def corrupt_array(self, arr):
        values = np.array(arr, dtype=float)

        # Calculate array statistics for determining shift magnitude
        array_std = np.std(values)
        if array_std == 0:
            array_std = np.mean(np.abs(values)) if np.mean(np.abs(values)) > 0 else 1.0

        target_std = array_std * self.std_scale

        # Determine segment length to corrupt
        segment_length = max(2, int(len(values) * self.segment_fraction))

        # Calculate noise scale to achieve target std
        # Formula: noise_std = sqrt((n/k) * (σ_target² - σ_0²))
        # where n = total length, k = segment length
        variance_diff = target_std ** 2 - array_std ** 2

        if variance_diff <= 0:
            raise ValueError(f"Target std ({target_std}) must be > original std ({array_std})")

        scale = np.sqrt(len(values) / segment_length * variance_diff)
        noise = np.random.randn(segment_length) * scale

        # Choose a random starting position for the corrupted segment
        if len(values) - segment_length < 1:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, len(values) - segment_length + 1)

        # Replace the segment in the array
        corrupted_values = values.copy()
        corrupted_values[start_idx : start_idx + segment_length] += noise

        return corrupted_values