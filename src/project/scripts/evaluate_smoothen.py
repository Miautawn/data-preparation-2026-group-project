import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Add src to sys.path
script_dir = Path(__file__).parent
project_src = script_dir.parent.parent
if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

from project.cleaning.smoothen_cleaner import SmoothenCleaner
from project.utils.dataset import derive_features
from project.utils.modeling import predict_model

def calculate_mae(df, pred_col, target_col):
    all_preds = []
    all_targets = []
    for _, row in df.iterrows():
        all_preds.extend(row[pred_col])
        all_targets.extend(row[target_col])
    return mean_absolute_error(all_targets, all_preds)

def evaluate_scale(scale, model, standard_scaler, static_encoder, dataset_arguments, cleaner, running_dir):
    data_path = running_dir / f"erroneous_scale_{scale}_running_data.parquet"
    if not data_path.exists():
        print(f"Skipping scale {scale}: File {data_path} not found.")
        return None

    print(f"\nEvaluating Scale {scale}...")
    df = pd.read_parquet(data_path)
    
    # --- 1. ORIGINAL ---
    df_orig = df.copy()
    df_orig = derive_features(df_orig)
    df_orig = standard_scaler.transform(df_orig)
    df_orig = static_encoder.transform(df_orig)
    preds_orig = predict_model(model, df_orig, dataset_args=dataset_arguments, n_workers=1, verbose=False)
    df_orig["predicted_heart_rate"] = [p for p in preds_orig]
    mae_orig = calculate_mae(df_orig, "predicted_heart_rate", "heart_rate")

    # --- 2. ERRONEOUS ---
    df_err = df.copy()
    df_err["longitude"] = df_err["erroneous_longitude"]
    df_err["latitude"] = df_err["erroneous_latitude"]
    df_err = derive_features(df_err)
    df_err = standard_scaler.transform(df_err)
    df_err = static_encoder.transform(df_err)
    preds_err = predict_model(model, df_err, dataset_args=dataset_arguments, n_workers=1, verbose=False)
    df_err["predicted_heart_rate"] = [p for p in preds_err]
    mae_err = calculate_mae(df_err, "predicted_heart_rate", "heart_rate")

    # --- 3. CLEANED ---
    df_clean = df.copy()
    df_clean["longitude"] = df_clean["erroneous_longitude"].apply(lambda x: cleaner.clean_array(x))
    df_clean["latitude"] = df_clean["erroneous_latitude"].apply(lambda x: cleaner.clean_array(x))
    df_clean = derive_features(df_clean)
    df_clean = standard_scaler.transform(df_clean)
    df_clean = static_encoder.transform(df_clean)
    preds_clean = predict_model(model, df_clean, dataset_args=dataset_arguments, n_workers=1, verbose=False)
    df_clean["predicted_heart_rate"] = [p for p in preds_clean]
    mae_clean = calculate_mae(df_clean, "predicted_heart_rate", "heart_rate")

    print(f"Scale {scale} Results: Original: {mae_orig:.4f}, Erroneous: {mae_err:.4f}, Cleaned: {mae_clean:.4f}")
    
    return {
        "scale": scale,
        "original": mae_orig,
        "erroneous": mae_err,
        "cleaned": mae_clean,
        "df_for_plot": df # return for coordinate plot if needed
    }

def main():
    # Paths
    running_dir = project_src / "project" / "baked_artifacts" / "running"
    model_path = running_dir / "running_fitrec_model.pt"
    scaler_path = running_dir / "running_user_standard_scaler.pkl"
    encoder_path = running_dir / "running_static_ordinal_encoder.pkl"

    # Load artifacts
    print("Loading model and preprocessors...")
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    standard_scaler = pickle.load(open(scaler_path, "rb"))
    static_encoder = pickle.load(open(encoder_path, "rb"))

    dataset_arguments = {
        "numerical_columns": [
            "time_elapsed_standardized",
            "altitude_standardized",
            "derived_speed_standardized",
            "derived_distance_standardized",
        ],
        "categorical_columns": ["userId_idx", "sport_idx", "gender_idx"],
        "heartrate_input_column": "heart_rate_standardized",
        "heartrate_output_column": "heart_rate",
        "workout_id_column": "id",
        "use_heartrate_input": True,
    }

    cleaner = SmoothenCleaner(max_diff=0.001)
    scales = [2, 4, 8]
    summary_results = []

    for s in scales:
        res = evaluate_scale(s, model, standard_scaler, static_encoder, dataset_arguments, cleaner, running_dir)
        if res:
            summary_results.append(res)

    # --- PLOT 1: COORDINATE COMPARISON (for scale 8 to show strength) ---
    print("\nGenerating coordinate comparison plot for scale 8...")
    s8_res = next(r for r in summary_results if r["scale"] == 8)
    df8 = s8_res["df_for_plot"]
    
    max_err = 0
    selected_idx = 0
    for idx in range(min(50, len(df8))):
        err = np.max(np.abs(df8.iloc[idx]["longitude"] - df8.iloc[idx]["erroneous_longitude"]))
        if err > max_err:
            max_err = err
            selected_idx = idx
            
    orig_lon = df8.iloc[selected_idx]["longitude"]
    err_lon = df8.iloc[selected_idx]["erroneous_longitude"]
    clean_lon = cleaner.clean_array(err_lon)
    
    plt.figure(figsize=(12, 6))
    plt.plot(orig_lon, label='Original Longitudinal', color='green', linewidth=2, alpha=0.8)
    plt.plot(err_lon, label='Erroneous Longitudinal', color='red', linestyle='--', alpha=0.5)
    plt.plot(clean_lon, label='Cleaned Longitudinal', color='blue', linestyle='-', linewidth=2)
    plt.title(f"Longitude Comparison - Scale 8 - Workout {selected_idx}")
    plt.xlabel("Step")
    plt.ylabel("Longitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(script_dir / "longitude_comparison_scale_8.png")

    # --- PLOT 2: MAE vs SCALE ---
    print("\nGenerating MAE vs Scale plot...")
    plot_df = pd.DataFrame(summary_results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["scale"], plot_df["original"], marker='o', label='Original', color='green', linewidth=2)
    plt.plot(plot_df["scale"], plot_df["erroneous"], marker='s', label='Erroneous', color='red', linewidth=2)
    plt.plot(plot_df["scale"], plot_df["cleaned"], marker='^', label='Cleaned', color='blue', linewidth=2)
    
    plt.title("MAE vs Error Scale")
    plt.xlabel("Std Scale of Erroneous Injection")
    plt.ylabel("Heart Rate MAE")
    plt.xticks(scales)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    mae_plot_path = script_dir / "mae_vs_scale.png"
    plt.savefig(mae_plot_path)
    print(f"MAE vs Scale plot saved to {mae_plot_path}")

    print("\n" + "="*30)
    print("Multi-Scale Summary:")
    print(plot_df[["scale", "original", "erroneous", "cleaned"]].to_string(index=False))
    print("="*30)

if __name__ == "__main__":
    main()
