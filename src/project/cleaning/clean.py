import numpy as np
import pandas as pd
from jenga.basis import DataCleaning


class RandomNoiseCleaning(DataCleaning):
    """
    Heuristic cleaner for RandomNoiseCorruption.

    Detects high-variance contiguous segments and smooths them using
    linear interpolation + optional local averaging.
    """

    def __init__(
        self,
        columns,
        z_threshold=3.5,
        min_segment_length=2,
        smoothing_window=5,
    ):
        """
        Args:
            columns (list): Columns to clean
            z_threshold (float): Robust z-score threshold to detect noise
            min_segment_length (int): Minimum length of a noisy segment
            smoothing_window (int): Window size for final smoothing
        """
        super().__init__()
        self.columns = columns
        self.z_threshold = z_threshold
        self.min_segment_length = min_segment_length
        self.smoothing_window = smoothing_window

    def transform(self, data):
        cleaned = data.copy()

        for col in self.columns:
            if col not in cleaned.columns:
                raise ValueError(f"Column '{col}' not found")

            for idx in cleaned.index:
                cleaned.at[idx, col] = self.clean_array(
                    cleaned.at[idx, col]
                )

        return cleaned

    def clean_array(self, arr):
        values = np.asarray(arr, dtype=float)

        if values.ndim != 1 or len(values) < 3:
            return values

        # ---- Robust z-score (median & MAD) ----
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        mad = mad if mad > 0 else 1.0

        z = 0.6745 * (values - median) / mad
        noisy_mask = np.abs(z) > self.z_threshold

        # ---- Identify contiguous noisy segments ----
        segments = self.find_segments(noisy_mask)

        cleaned = values.copy()

        for start, end in segments:
            if end - start < self.min_segment_length:
                continue

            cleaned[start:end] = self.interpolate_segment(
                cleaned, start, end
            )

        # ---- Optional final smoothing ----
        if self.smoothing_window > 1:
            cleaned = self.moving_average(cleaned, self.smoothing_window)

        return cleaned

    @staticmethod
    def find_segments(mask):
        segments = []
        start = None

        for i, flag in enumerate(mask):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(mask)))

        return segments

    @staticmethod
    def interpolate_segment(arr, start, end):
        left = arr[start - 1] if start > 0 else arr[end]
        right = arr[end] if end < len(arr) else arr[start - 1]

        return np.linspace(left, right, end - start)

    @staticmethod
    def moving_average(x, window):
        pad = window // 2
        padded = np.pad(x, pad, mode="edge")
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode="valid")
