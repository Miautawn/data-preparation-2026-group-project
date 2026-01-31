import os
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from project.corruption.array_corruption.ConstantNoiseCorruption import ConstantNoiseCorruption
from project.corruption.array_corruption.RandomNoiseCorruption import RandomNoiseCorruption

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
src_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def run_error_injection(std_scale: float):
    """
    Inject errors to longitude and latitude columns with a specific std_scale.
    """
    print(f"\n{'='*20} Running Injection for std_scale={std_scale} {'='*20}")
    data_root = os.path.join(project_root, "data")
    sports = ["biking", "running", "walking"]
    seed = 42
    np.random.seed(seed)
    
    for sport in sports:
        print(f"Processing {sport}...")
        sport_dir = os.path.join(data_root, sport)
        input_file = os.path.join(sport_dir, f"{sport}_test_raw.parquet")
        output_file = os.path.join(sport_dir, f"erroneous_scale_{std_scale}_{sport}_data.parquet")
        
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            continue
            
        # Load parquet file and convert to pandas
        table = pq.read_table(input_file)
        # Convert to pydict first to avoid list conversion issues in to_pandas()
        df = pd.DataFrame(table.to_pydict())
        
        # Prepare erroneous columns
        df['erroneous_longitude'] = df['longitude'].copy()
        df['erroneous_latitude'] = df['latitude'].copy()
        
        # Shuffle indices for fair splitting
        indices = df.index.tolist()
        np.random.shuffle(indices)
        
        # Define proportions
        n = len(df)
        p_constant = 0.5
        
        # Proportions within constant/random parts
        # 30% Config A (0.2 sf), 30% Config B (0.4 sf), 40% Config C (0.6 sf)
        config_proportions = [0.3, 0.3, 0.4]
        
        # Split into Constant and Random parts
        mid = int(n * p_constant)
        constant_indices = indices[:mid]
        random_indices = indices[mid:]
        
        def apply_nested_splits(target_df, subset_indices, CorruptionClass):
            n_sub = len(subset_indices)
            
            # Config splits
            s1 = int(n_sub * config_proportions[0])
            s2 = int(n_sub * (config_proportions[0] + config_proportions[1]))
            
            subsets = [
                subset_indices[:s1],
                subset_indices[s1:s2],
                subset_indices[s2:]
            ]
            
            sf_list = [0.2, 0.4, 0.6]
            
            results = []
            for i, (subset, sf) in enumerate(zip(subsets, sf_list)):
                if len(subset) == 0:
                    continue
                    
                chunk = target_df.loc[subset].copy()
                n_chunk = len(chunk)
                
                # Further split by num_segments
                if sf == 0.2 or sf == 0.4:
                    # 80% ns=1, 20% ns=2
                    ns1 = int(n_chunk * 0.8)
                    chunk_ns1_idx = chunk.index[:ns1]
                    chunk_ns2_idx = chunk.index[ns1:]
                    
                    if len(chunk_ns1_idx) > 0:
                        corrupter = CorruptionClass(columns=['erroneous_longitude', 'erroneous_latitude'], 
                                                   segment_fraction=sf, num_segments=1, std_scale=std_scale, seed=seed)
                        chunk.loc[chunk_ns1_idx] = corrupter.transform(chunk.loc[chunk_ns1_idx])
                    
                    if len(chunk_ns2_idx) > 0:
                        corrupter = CorruptionClass(columns=['erroneous_longitude', 'erroneous_latitude'], 
                                                   segment_fraction=sf, num_segments=2, std_scale=std_scale, seed=seed)
                        chunk.loc[chunk_ns2_idx] = corrupter.transform(chunk.loc[chunk_ns2_idx])
                
                elif sf == 0.6:
                    # 60% ns=1, 20% ns=2, 20% ns=3
                    ns1 = int(n_chunk * 0.6)
                    ns2 = int(n_chunk * 0.8)
                    chunk_ns1_idx = chunk.index[:ns1]
                    chunk_ns2_idx = chunk.index[ns1:ns2]
                    chunk_ns3_idx = chunk.index[ns2:]
                    
                    if len(chunk_ns1_idx) > 0:
                        corrupter = CorruptionClass(columns=['erroneous_longitude', 'erroneous_latitude'], 
                                                   segment_fraction=sf, num_segments=1, std_scale=std_scale, seed=seed)
                        chunk.loc[chunk_ns1_idx] = corrupter.transform(chunk.loc[chunk_ns1_idx])
                    
                    if len(chunk_ns2_idx) > 0:
                        corrupter = CorruptionClass(columns=['erroneous_longitude', 'erroneous_latitude'], 
                                                   segment_fraction=sf, num_segments=2, std_scale=std_scale, seed=seed)
                        chunk.loc[chunk_ns2_idx] = corrupter.transform(chunk.loc[chunk_ns2_idx])
                    
                    if len(chunk_ns3_idx) > 0:
                        corrupter = CorruptionClass(columns=['erroneous_longitude', 'erroneous_latitude'], 
                                                   segment_fraction=sf, num_segments=3, std_scale=std_scale, seed=seed)
                        chunk.loc[chunk_ns3_idx] = corrupter.transform(chunk.loc[chunk_ns3_idx])
                
                results.append(chunk)
            
            return pd.concat(results) if results else pd.DataFrame()

        print(f"Applying Constant Noise to {len(constant_indices)} rows...")
        df_constant = apply_nested_splits(df, constant_indices, ConstantNoiseCorruption)
        
        print(f"Applying Random Noise to {len(random_indices)} rows...")
        df_random = apply_nested_splits(df, random_indices, RandomNoiseCorruption)
        
        # Combine all parts
        final_df = pd.concat([df_constant, df_random]).sort_index()
        
        print(f"Saving to {output_file}...")
        final_df.to_parquet(output_file)
        print(f"Done for {sport}.\n")

def visualize_sample(scale: float, sport: str):
    """
    Visualize original vs erroneous longitude for a sample file.
    """
    print(f"\n{'#'*10} Visualizing Sample for {sport} (Scale {scale}) {'#'*10}")
    data_path = os.path.join(project_root, 'data', sport, f'erroneous_scale_{scale}_{sport}_data.parquet')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    table = pq.read_table(data_path)
    # We only need a sample, but loading the whole file into pandas is fine for this scale
    df = pd.DataFrame(table.to_pydict())
    
    # Select the first data point (row 0)
    row_idx = 0
    orig_long = df['longitude'].iloc[row_idx]
    err_long = df['erroneous_longitude'].iloc[row_idx]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(orig_long, label='Original Longitude', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(err_long, label=f'Erroneous Longitude (Scale {scale})', color='red', alpha=0.5, linestyle='--', linewidth=1)
    
    plt.title(f"Longitude vs Erroneous Longitude (Scale {scale}) - {sport.capitalize()} Sample {row_idx}")
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
    scales = [2, 5, 10]
    for scale in scales:
        run_error_injection(scale)
    
    print("\n" + "#"*40)
    print("Verifying outputs (Column presence, Length match, Std Scale):")
    print("#"*40)
    
    data_root = os.path.join(project_root, "data")
    for scale in scales:
        print(f"\n--- SCALE: {scale} ---")
        for sport in ["biking", "running", "walking"]:
            output_file = os.path.join(data_root, sport, f"erroneous_scale_{scale}_{sport}_data.parquet")
            if os.path.exists(output_file):
                table = pq.read_table(output_file)
                cols = table.column_names
                
                # Check column existence
                has_cols = 'erroneous_longitude' in cols and 'erroneous_latitude' in cols
                
                # Check first row
                orig_long = table.column('longitude')[0].as_py()
                err_long = table.column('erroneous_longitude')[0].as_py()
                
                # Check array length match
                len_match = len(orig_long) == len(err_long)
                
                # Check std scale for first data point
                orig_std = np.std(orig_long)
                err_std = np.std(err_long)
                ratio = err_std/orig_std if orig_std > 0 else 0
                
                status = "OK" if (has_cols and len_match and abs(ratio - scale) < 1e-4) else "NOK"
                print(f"{status} [{sport.upper()}] Cols: {has_cols}, Len Match: {len_match}, Std Ratio: {ratio:.4f} (Target: {scale})")
    
    # Finally, visualize one sample (e.g., Biking Scale 2)
    visualize_sample(2, "biking")