import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import os

# Add src directory to path
sys.path.insert(0, str(Path("src").resolve()))


from project.corruption.ArrayCorruption import *


class TestUnusualValueCorruption:
    """Test suite for UnusualValueCorruption class"""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters"""
        corruption = UnusualValueCorruption(
            column='test_col',
            row_fraction=0.5,
            segment_fraction=0.3,
            std_scale=2.0,
            seed=42
        )
        assert corruption.column == 'test_col'
        assert corruption.row_fraction == 0.5
        assert corruption.segment_fraction == 0.3
        assert corruption.std_scale == 2.0
        assert corruption.seed == 42


    def test_row_fraction_selection(self):
        """Test that correct fraction of rows are corrupted"""
        np.random.seed(42)

        # Create dataset with 10 rows
        data = pd.DataFrame({
            'arrays': [np.arange(100) + i * 0.1 for i in range(10)]
        })

        print(data)
        original_data = data.copy()

        # Corrupt 50% of rows
        corruption = UnusualValueCorruption(
            column='arrays',
            row_fraction=0.5,
            segment_fraction=0.2,
            std_scale=2.0,
            seed=42
        )

        corrupted_data = corruption.transform(data)

        # Count how many rows were actually corrupted
        corrupted_count = 0
        for idx in range(10):
            if not np.array_equal(original_data.at[idx, 'arrays'], corrupted_data.at[idx, 'arrays']):
                corrupted_count += 1

        # Should corrupt exactly 5 rows (50% of 10)
        assert corrupted_count == 5

    def test_segment_fraction_selection(self):
        """Test that correct fraction of elements in array are corrupted"""
        np.random.seed(42)

        # Create simple array
        array_length = 100
        data = pd.DataFrame({
            'arrays': [np.arange(array_length, dtype=float)]
        })

        corruption = UnusualValueCorruption(
            column='arrays',
            row_fraction=1.0,
            segment_fraction=0.2,  # 20% of 100 = 20 elements
            std_scale=2.0,
            seed=42
        )

        original_array = data.at[0, 'arrays'].copy()
        corrupted_data = corruption.transform(data)
        corrupted_array = corrupted_data.at[0, 'arrays']

        # Find corrupted segment by comparing arrays
        diff = corrupted_array - original_array
        corrupted_indices = np.where(np.abs(diff) > 1e-10)[0]

        # Should corrupt approximately 20 elements (segment_fraction * array_length)
        expected_segment_length = max(2, int(array_length * 0.2))
        assert len(corrupted_indices) == expected_segment_length

        # Corrupted elements should be consecutive
        if len(corrupted_indices) > 1:
            assert np.all(np.diff(corrupted_indices) == 1)

    def test_std_scale_increase(self):
        """Test that std increases by approximately the specified scale"""
        np.random.seed(42)

        # Create array with known std
        data = pd.DataFrame({
            'arrays': [np.random.randn(1000) * 10]  # std ≈ 10
        })

        original_std = np.std(data.at[0, 'arrays'])

        corruption = UnusualValueCorruption(
            column='arrays',
            row_fraction=1.0,
            segment_fraction=0.3,
            std_scale=2.0,
            seed=42
        )

        corrupted_data = corruption.transform(data)
        corrupted_std = np.std(corrupted_data.at[0, 'arrays'])

        # Check that std increased by approximately the scale factor
        # Allow 20% tolerance due to randomness and approximation in formula
        expected_std = original_std * 2.0
        assert corrupted_std > original_std  # Must increase
        assert abs(corrupted_std - expected_std) / expected_std < 0.2

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""
        data = pd.DataFrame({
            'arrays': [np.random.randn(100) for _ in range(5)]
        })

        corruption1 = UnusualValueCorruption(
            column='arrays',
            row_fraction=0.6,
            segment_fraction=0.2,
            std_scale=1.5,
            seed=123
        )

        corruption2 = UnusualValueCorruption(
            column='arrays',
            row_fraction=0.6,
            segment_fraction=0.2,
            std_scale=1.5,
            seed=123
        )

        result1 = corruption1.transform(data.copy())
        result2 = corruption2.transform(data.copy())

        # Results should be identical with same seed
        for idx in range(5):
            np.testing.assert_array_almost_equal(
                result1.at[idx, 'arrays'],
                result2.at[idx, 'arrays']
            )

    def test_zero_std_array(self):
        """Test handling of constant (zero std) arrays"""
        data = pd.DataFrame({
            'arrays': [np.ones(100) * 5.0]  # Constant array
        })

        corruption = UnusualValueCorruption(
            column='arrays',
            row_fraction=1.0,
            segment_fraction=0.2,
            std_scale=2.0,
            seed=42
        )

        # Should not raise error, uses fallback std calculation
        corrupted_data = corruption.transform(data)

        # Should have added noise to the segment
        original_array = data.at[0, 'arrays']
        corrupted_array = corrupted_data.at[0, 'arrays']
        assert not np.array_equal(original_array, corrupted_array)

    def test_preserves_uncorrupted_rows(self):
        """Test that uncorrupted rows remain unchanged"""
        np.random.seed(42)

        data = pd.DataFrame({
            'arrays': [np.arange(50, dtype=float) for _ in range(10)]
        })

        original_data = data.copy()

        corruption = UnusualValueCorruption(
            column='arrays',
            row_fraction=0.3,  # Only corrupt 3 rows
            segment_fraction=0.2,
            std_scale=1.5,
            seed=42
        )

        corrupted_data = corruption.transform(data)

        # Count unchanged rows
        unchanged_count = 0
        for idx in range(10):
            if np.array_equal(original_data.at[idx, 'arrays'], corrupted_data.at[idx, 'arrays']):
                unchanged_count += 1

        # Should have 7 unchanged rows (70% of 10)
        assert unchanged_count == 7

    def test_dataframe_indices_preserved(self):
        """Test that DataFrame indices are preserved after corruption"""
        data = pd.DataFrame({
            'arrays': [np.arange(50, dtype=float) for _ in range(5)]
        }, index=[10, 20, 30, 40, 50])

        corruption = UnusualValueCorruption(
            column='arrays',
            row_fraction=0.6,
            segment_fraction=0.2,
            std_scale=1.5,
            seed=42
        )

        corrupted_data = corruption.transform(data)

        # Check indices are preserved
        assert list(corrupted_data.index) == [10, 20, 30, 40, 50]

    def test_minimum_segment_length(self):
        """Test that minimum segment length is enforced"""
        data = pd.DataFrame({
            'arrays': [np.arange(10, dtype=float)]
        })

        # segment_fraction=0.05 would give 0.5 elements, but min is 2
        corruption = UnusualValueCorruption(
            column='arrays',
            row_fraction=1.0,
            segment_fraction=0.05,
            std_scale=2.0,
            seed=42
        )

        corrupted_data = corruption.transform(data)
        original_array = data.at[0, 'arrays']
        corrupted_array = corrupted_data.at[0, 'arrays']

        # Count corrupted elements
        diff = corrupted_array - original_array
        corrupted_count = np.sum(np.abs(diff) > 1e-10)

        # Should have at least 2 elements corrupted (minimum)
        assert corrupted_count >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
