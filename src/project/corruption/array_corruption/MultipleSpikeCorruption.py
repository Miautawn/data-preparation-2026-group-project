import numpy as np
import pandas as pd
from jenga.basis import DataCorruption


class MultipleSpikeCorruption(DataCorruption):
    """
    Data corruption that shifts continuous segments to unusual values.

    This corruption identifies segments of data and shifts them to unusual values
    to reach a certain scale in std.
    """

    def __init__(self, columns, row_fraction=1.0, segment_fraction=0.2, num_segments=2, std_scale=2, seed=None):
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
        self.num_segments = num_segments

    def transform(self, data):
        corrupted_data = data.copy()

        if self.seed is not None:
            np.random.seed(self.seed)

        for col in self.columns:
            if col not in corrupted_data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        # Determine array length from the first element of the first column
        first_val = data[self.columns[0]].iloc[0]
        if hasattr(first_val, '__len__') and not isinstance(first_val, (str, bytes)):
             array_length = len(first_val)
        else:
             raise ValueError(f"Column '{self.columns[0]}' must contain arrays/lists, found {type(first_val)}")

        for col in self.columns:
            if len(data.loc[0, col]) != array_length:
                raise ValueError(f"All columns must have same length: '{col}'")

        # Determine which rows to corrupt
        n_rows = len(corrupted_data)
        n_rows_to_corrupt = max(1, int(n_rows * self.row_fraction))
        # Use DataFrame index instead of numeric indices
        available_indices = corrupted_data.index.tolist()
        rows_to_corrupt = np.random.choice(available_indices, size=n_rows_to_corrupt, replace=False)

        # Determine segment length to corrupt
        segment_length = max(2, int(array_length * self.segment_fraction))

        segments = self.random_segments(array_length, self.num_segments, sum_segment_length=segment_length)

        # Apply corruption to selected rows
        for idx in rows_to_corrupt:
            for col in self.columns:
                corrupted_data.at[idx, col] = self.corrupt_array(corrupted_data.at[idx, col], array_length, segment_length, segments)

        return corrupted_data

    def random_segments(self, array_length, num_segments, sum_segment_length):
        cuts = np.sort(np.random.choice(range(1, sum_segment_length), num_segments - 1, replace=False))
        lengths = np.diff(np.r_[0, cuts, sum_segment_length])

        gaps = np.random.multinomial(array_length - sum_segment_length, [1 / (num_segments + 1)] * (num_segments + 1))
        starts = np.cumsum(np.r_[gaps[0], lengths + gaps[1:]])

        return list(zip(starts, lengths))

    def corrupt_array(self, arr, array_length, segment_length, segments):
        values = np.array(arr, dtype=float)

        # Calculate array statistics
        mean_val = np.mean(values)
        sigma = np.std(values)
        
        # If sigma is 0, we can't scale it. But we can add variation.
        if sigma == 0:
             # If sigma is 0, we can define a base magnitude to shift by.
             # Let's fallback to the approximation formula for c with a dummy sigma
             base_scale = np.mean(np.abs(values)) if np.mean(np.abs(values)) > 0 else 1.0
             sigma = base_scale
             c = sigma * np.sqrt((self.std_scale ** 2 - 1) * array_length ** 2 / (segment_length * (array_length - segment_length)))
        else:
            # Solve quadratic equation for c: A*c^2 + B*c + C = 0
            # A = L_s * (N - L_s) / N^2
            # B = (2/N) * sum_{i in I} (X_i - mean_X)
            # C = - (S^2 - 1) * sigma^2
            
            N = array_length
            L_s = segment_length
            
            A = L_s * (N - L_s) / (N ** 2)
            
            # Identify indices in segments
            segment_indices = []
            for start, length in segments:
                segment_indices.extend(range(start, start + length))
            
            # Calculate Sum of residuals in the segments
            residuals = values[segment_indices] - mean_val
            sum_residuals_in_segment = np.sum(residuals)
            
            B = (2 / N) * sum_residuals_in_segment
            
            C = - (self.std_scale ** 2 - 1) * (sigma ** 2)
            
            # Discriminant
            delta = B**2 - 4*A*C
            
            if delta < 0:
                raise ValueError("Cannot find real solution for corruption magnitude")
            

            if np.random.random() < 0.5:
                c = (-B + np.sqrt(delta)) / (2 * A)
            else:
                c = (-B - np.sqrt(delta)) / (2 * A)
            
        # Replace the segment in the array
        corrupted_values = values.copy()
        for segment in segments:
            corrupted_values[segment[0] : segment[0] + segment[1]] += c
            
        return corrupted_values

if __name__ == "__main__":
    # Create sample dataframe
    array_len = 20
    data = {
        'id': [1, 2, 3],
        'array_1': [list([np.random.randn()] * array_len) for _ in range(3)],
        'array_2': [list(np.random.normal(5, 2, array_len)) for _ in range(3)]
    }
    df = pd.DataFrame(data)

    print("Original Data (First Row):")
    print(df.iloc[0])
    print("\n" + "="*50 + "\n")

    # Initialize corruption
    # std_scale=2.0 (default), segment_fraction=0.2 (default)
    corrupter = MultipleSpikeCorruption(columns=['array_1', 'array_2'], std_scale=3, num_segments=1, seed=42)

    # Apply transform
    corrupted_df = corrupter.transform(df)

    print("Corrupted Data (First Row):")
    print(corrupted_df.iloc[0])
    print("\n" + "="*50 + "\n")

    # Print STD comparison
    print("Standard Deviation Comparison:")
    print(f"{'Column':<10} | {'Row':<5} | {'Original STD':<15} | {'Corrupted STD':<15} | {'Ratio':<10}")
    print("-" * 75)

    for i in range(len(df)):
        for col in ['array_1', 'array_2']:
            orig_std = np.std(df.iloc[i][col])
            corr_std = np.std(corrupted_df.iloc[i][col])
            ratio = corr_std / orig_std if orig_std > 0 else 0
            print(f"{col:<10} | {i:<5} | {orig_std:<15.4f} | {corr_std:<15.4f} | {ratio:<10.2f}")