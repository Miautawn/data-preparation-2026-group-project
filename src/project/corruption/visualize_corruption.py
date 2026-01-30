import os
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Calculate paths relative to the script location to allow running from anywhere
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
# src is 2 levels up from this script (corruption/visualize_corruption.py -> project/corruption -> src)
src_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def main():
    # File path for scale 2 biking data
    data_path = os.path.join(project_root, 'data', 'biking', 'erroneous_scale_2_biking_data.parquet')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    table = pq.read_table(data_path)
    df = pd.DataFrame(table.to_pydict())
    
    # Select the first data point (row 0)
    row_idx = 0
    orig_long = df['longitude'].iloc[row_idx]
    err_long = df['erroneous_longitude'].iloc[row_idx]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(orig_long, label='Original Longitude', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(err_long, label='Erroneous Longitude (Scale 2)', color='red', alpha=0.5, linestyle='--', linewidth=1)
    
    plt.title(f"Longitude vs Erroneous Longitude (Scale 2) - Biking Sample {row_idx}")
    plt.xlabel("Index in Sequence")
    plt.ylabel("Longitude Value")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    output_image = os.path.join(script_dir, 'corruption_comparison.png')
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot saved successfully to {output_image}")

    # Also print some stats
    print(f"\nStats for Sample {row_idx}:")
    print(f"Original STD:  {np.std(orig_long):.6f}")
    print(f"Erroneous STD: {np.std(err_long):.6f}")
    print(f"Actual Ratio:  {np.std(err_long)/np.std(orig_long):.4f}")

if __name__ == "__main__":
    main()
