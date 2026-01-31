# Array Corruption Module

This module provides classes for injecting noise into nested array data, specifically for latitude and longitude sequences.

## Components

- **ConstantNoiseCorruption**: Shifts continuous segments by a constant value to achieve a target standard deviation scale.
- **RandomNoiseCorruption**: Adds random noise to continuous segments to achieve a target standard deviation scale.

## Configuration Parameters

The error injection script (`main_error_injection.py`) uses the following configuration for data corruption:

### Standard Deviation Scales
The following `std_scale` values are used to generate corrupted datasets:
- **2.0**: The resulting array's standard deviation is twice that of the original.
- **5.0**: The resulting array's standard deviation is five times that of the original.
- **10.0**: The resulting array's standard deviation is ten times that of the original.

### Data Partitioning
The input dataset is split into two equal halves (50% each) for different corruption types:
- **50% Constant Noise**
- **50% Random Noise**

### Segment Fractions and Probabilities
Within each noise type, the data is further partitioned by the fraction of the sequence to be corrupted (`segment_fraction`). Each fraction has a specific distribution for the number of segments (`num_segments`):

| Segment Fraction | Population % | Num Segments Distribution |
|------------------|--------------|---------------------------|
| **0.2**          | 30%          | 80% (ns=1), 20% (ns=2)    |
| **0.4**          | 30%          | 80% (ns=1), 20% (ns=2)    |
| **0.6**          | 40%          | 60% (ns=1), 20% (ns=2), 20% (ns=3) |

### Global Parameters
- **row_fraction**: 1.0 (all rows in the split are processed).
- **seed**: 42 (ensures reproducibility).

## Usage

To generate the corrupted parquet files for all sports and scales, run:

```bash
python src/project/corruption/main_error_injection.py
```

This will produce files with the prefix `erroneous_scale_<scale>_` in the corresponding sport data folders and generate a visualization plot `corruption_comparison.png`.
