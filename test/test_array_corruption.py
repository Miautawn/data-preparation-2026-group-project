import sys
import os

# Add the src directory to the python path to allow imports from project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
import pandas as pd
from project.corruption.array_corruption.MultipleSpikeCorruption import MultipleSpikeCorruption


class TestMultipleSpikeCorruption:

    @pytest.fixture
    def random_data(self):
        # Create a dataframe where the data length matches the number of rows
        # as per the implementation detail in MultipleSpikeCorruption
        n_rows = 1000
        array_length = 1000

        # Create random gaussian data
        data = []
        for _ in range(n_rows):
            # Use random noise
            arr = np.random.normal(0, 1, array_length)
            data.append(arr)

        df = pd.DataFrame({'signal': data})
        return df, n_rows, array_length

    def test_initialization(self):
        transformer = MultipleSpikeCorruption(
            columns=['signal'],
            row_fraction=0.5,
            segment_fraction=0.1,
            std_scale=2.0
        )
        assert transformer.columns == ['signal']
        assert transformer.row_fraction == 0.5
        assert transformer.segment_fraction == 0.1
        assert transformer.std_scale == 2.0

    def test_transform_structure(self, random_data):
        df, _, _ = random_data

        # Corrupt all rows to inspect them easily
        transformer = MultipleSpikeCorruption(
            columns=['signal'],
            row_fraction=1.0,
            segment_fraction=0.1,
            std_scale=3.0,
            seed=42
        )

        corrupted_df = transformer.transform(df)

        # Check shape is preserved
        assert corrupted_df.shape == df.shape
        assert list(corrupted_df.columns) == list(df.columns)

        # Check data types
        assert isinstance(corrupted_df.iloc[0]['signal'], (list, np.ndarray))

    def test_std_scaling(self, random_data):
        df, n_rows, array_length = random_data

        target_scale = 3.0
        transformer = MultipleSpikeCorruption(
            columns=['signal'],
            row_fraction=1.0,  # Corrupt all to get good statistics
            segment_fraction=0.2,  # Significant fraction
            std_scale=target_scale,
            seed=123
        )

        corrupted_df = transformer.transform(df)

        # Check statistics for a sample of rows
        ratios = []
        for i in range(100):  # Check first 100 rows
            original_arr = np.array(df.iloc[i]['signal'])
            corrupted_arr = np.array(corrupted_df.iloc[i]['signal'])

            std_orig = np.std(original_arr)
            std_corr = np.std(corrupted_arr)

            if std_orig > 0:
                ratios.append(std_corr / std_orig)

        avg_ratio = np.mean(ratios)

        # Allow some tolerance because of random segments alignment with random data
        # varying covariance term
        assert np.isclose(avg_ratio, target_scale, rtol=0.1), \
            f"Expected std ratio {target_scale}, got {avg_ratio}"

    def test_segment_fraction(self, random_data):
        df, n_rows, array_length = random_data

        seg_fraction = 0.2
        transformer = MultipleSpikeCorruption(
            columns=['signal'],
            row_fraction=1.0,
            segment_fraction=seg_fraction,
            std_scale=5.0,  # Large scale ensures modified values are different
            seed=42
        )

        corrupted_df = transformer.transform(df)

        idx = 0
        original_arr = np.array(df.iloc[idx]['signal'])
        corrupted_arr = np.array(corrupted_df.iloc[idx]['signal'])

        # Find modified elements
        diff = corrupted_arr - original_arr
        modified_mask = np.abs(diff) > 1e-9  # Tolerance for potential float weirdness

        modified_count = np.sum(modified_mask)
        expected_count = int(array_length * seg_fraction)

        # The implementation calculates segments carefully, usually it should be exact or very close
        # Note: random_segments uses multinomial logic, but constrained sum?
        # Let's check line 59: segment_length = max(2, int(array_length * self.segment_fraction))
        # line 61: sum_segment_length=segment_length
        # So it should be exactly segment_length

        expected_exact = max(2, int(array_length * seg_fraction))

        # However, elements might not be modified if the 'c' added is 0?
        # But with std_scale=5 and normal data, c should be non-zero.

        assert modified_count == expected_exact, \
            f"Expected {expected_exact} modified elements, got {modified_count}"

    def test_constant_shift(self, random_data):
        df, n_rows, array_length = random_data

        transformer = MultipleSpikeCorruption(
            columns=['signal'],
            row_fraction=1.0,
            segment_fraction=0.1,
            std_scale=2.0,
            seed=42
        )

        corrupted_df = transformer.transform(df)

        idx = 0
        original_arr = np.array(df.iloc[idx]['signal'])
        corrupted_arr = np.array(corrupted_df.iloc[idx]['signal'])

        diff = corrupted_arr - original_arr
        modified_values = diff[np.abs(diff) > 1e-9]

        # All modified values should be shifted by the same constant 'c'
        # Check standard deviation of the shift amounts is ~0
        if len(modified_values) > 0:
            assert np.std(modified_values) < 1e-9, "Modified segments should have constant shift"

    def test_error_handling(self):
        df = pd.DataFrame({'A': [[1, 2], [3, 4]]})
        # array_length will be 2 (rows). len(element) is 2. Matches.

        # Test scale <= 1 error
        with pytest.raises(ValueError, match="std_scale must be > 1"):
            MultipleSpikeCorruption(columns=['A'], std_scale=0.5)

        # Test missing column
        transformer = MultipleSpikeCorruption(columns=['B'], std_scale=2)
        with pytest.raises(ValueError, match="Column 'B' not found"):
            transformer.transform(df)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

