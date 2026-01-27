import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import random
import duckdb

# --- Helper function for distance ---
def haversine_distance_meters(lat1, lon1, lat2, lon2):
    R = 6378137.0 
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

class CoordinatePrecisionCorruption:
    
    @staticmethod
    def run_corruption(input_filename: str, output_filename: str, corruption_percentage: int = 10):
        try:
            table = pq.read_table(input_filename)
            df = pd.DataFrame(table.to_pydict())
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        def corrupt_coordinate_value(val):
            if pd.isna(val) or np.isinf(val):
                return val
            if random.randint(1, 100) <= corruption_percentage:
                # Simulates missing digits by rounding to random low precision (2-6 decimals)
                decimals = random.randint(2, 6)
                return round(val, decimals)
            return val

        def process_list_column(coord_list):
            if coord_list is None: return None
            return [corrupt_coordinate_value(x) for x in coord_list]

        if 'longitude' in df.columns:
            df['longitude'] = df['longitude'].apply(process_list_column)
        if 'latitude' in df.columns:
            df['latitude'] = df['latitude'].apply(process_list_column)
            
        df.to_parquet(output_filename, engine='pyarrow')

    @staticmethod
    def run_cleaning(input_filename: str, output_filename: str, activity: str = 'bike'):
        try:
            table = pq.read_table(input_filename)
            df = pd.DataFrame(table.to_pydict())
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        def clean_row_vectorized(row):
            try:
                lons = np.array(row.get('longitude', []), dtype='float64')
                lats = np.array(row.get('latitude', []), dtype='float64')
            except (ValueError, TypeError):
                return row.get('longitude', []), row.get('latitude', [])

            if len(lons) == 0: return lons, lats

            df_track = pd.DataFrame({'lat': lats, 'lon': lons})

            # --- DETECT MISSING DIGITS ---
            # Check if values are "too round" (fewer than ~14 decimals)
            is_rounded_lat = np.abs(df_track['lat'] - df_track['lat'].round(13)) < 1e-14
            is_rounded_lon = np.abs(df_track['lon'] - df_track['lon'].round(13)) < 1e-14
            
            # Mask the detected low-precision values
            if not is_rounded_lat.all():
                df_track.loc[is_rounded_lat, 'lat'] = np.nan
            if not is_rounded_lon.all():
                df_track.loc[is_rounded_lon, 'lon'] = np.nan

            # --- REPAIR TO FULL SIZE ---
            # Use average of valid neighbors (ffill/bfill) to restore precision
            lat_before = df_track['lat'].ffill()
            lat_after = df_track['lat'].bfill()
            df_track['lat'] = df_track['lat'].fillna((lat_before + lat_after) / 2)

            lon_before = df_track['lon'].ffill()
            lon_after = df_track['lon'].bfill()
            df_track['lon'] = df_track['lon'].fillna((lon_before + lon_after) / 2)

            # Cleanup edges
            df_track = df_track.ffill().bfill()
            return df_track['lon'].values, df_track['lat'].values

        cleaned_results = df.apply(clean_row_vectorized, axis=1, result_type='expand')
        df['longitude'] = cleaned_results[0]
        df['latitude'] = cleaned_results[1]
        df.to_parquet(output_filename, engine='pyarrow')






if __name__ == "__main__":
    input_file = "first_10_rows.parquet"
    corrupted_file = "corrupted_first_10_rows.parquet"
    cleaned_file = "cleaned_first_10_rows.parquet"

    # 1. Corruption
    print("--- 1. RUNNING CORRUPTION (10%, dropping digits) ---")
    CoordinatePrecisionCorruption.run_corruption(input_file, corrupted_file, corruption_percentage=10)

    # 2. Cleaning
    print("--- 2. RUNNING CLEANING (Restoring to full precision) ---")
    CoordinatePrecisionCorruption.run_cleaning(corrupted_file, cleaned_file)

    # 3. Analysis
    print("\n--- 3. ANALYSIS ---")
    try:
        df_real = duckdb.query(f"SELECT * FROM '{input_file}' LIMIT 1").df()
        df_clean = duckdb.query(f"SELECT * FROM '{cleaned_file}' LIMIT 1").df()
        
        real_lats = df_real.iloc[0]['latitude']
        real_lons = df_real.iloc[0]['longitude']
        clean_lats = df_clean.iloc[0]['latitude']
        clean_lons = df_clean.iloc[0]['longitude']
        
        # Wider Header to fit all data
        header = f"{'Idx':<4} | {'Orig Lat':<18} {'Orig Lon':<18} | {'Clean Lat':<18} {'Clean Lon':<18} | {'Error (m)':<12}"
        print(header)
        print("-" * len(header))
        
        for i in range(min(10, len(real_lats))):
            dist = haversine_distance_meters(
                real_lats[i], real_lons[i], 
                clean_lats[i], clean_lons[i]
            )
            
            # Prepare formatted strings
            lat_str = f"{clean_lats[i]:<18.15f}"
            lon_str = f"{clean_lons[i]:<18.15f}"
            
            # --- HIGHLIGHT CHANGES IN RED ---
            RED = "\033[91m"
            RESET = "\033[0m"
            
            # If Lat differs more than 1 nanometer (approx)
            if abs(real_lats[i] - clean_lats[i]) > 1e-9:
                lat_str = f"{RED}{lat_str}{RESET}"
                
            # If Lon differs more than 1 nanometer (approx)
            if abs(real_lons[i] - clean_lons[i]) > 1e-9:
                lon_str = f"{RED}{lon_str}{RESET}"
            
            print(f"{i:<4} | {real_lats[i]:<18.15f} {real_lons[i]:<18.15f} | {lat_str} {lon_str} | {dist:.6f}")

    except Exception as e:
        print(f"Analysis Error: {e}")