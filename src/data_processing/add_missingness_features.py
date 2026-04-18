import pandas as pd
import numpy as np
import os

def main():
    print("Loading datasets...")
    # 1. Load your final cleaned dataset
    clean_path = '../../data/processed_files/patient_timeseries_cleaned.parquet'
    df_clean = pd.read_parquet(clean_path)
    df_clean = df_clean.rename(columns={'wbc_log1p': 'wbc'})  # Ensure consistent naming if needed
    
    # 2. Load the Step 3 intermediate dataset which still has the pure NaNs
    # Note: adjust to "_v2.zst" if your cleaned data was built from the v2 pipeline
    step3_path = '../../data/processed_files/init_traj_step3_standardized_grids_v2.zst'
    df_step3 = pd.read_parquet(step3_path)
    
    # The high-value labs to track
    lab_cols = ['lactate', 'wbc', 'creatinine', 'platelets']
    
    # Filter only the columns we actually need from the raw data
    raw_cols_to_keep = ['stay_id', 'timestep'] + [c for c in lab_cols if c in df_step3.columns]
    df_raw = df_step3[raw_cols_to_keep].copy()
    
    print("Merging raw lab measurements into cleaned data...")
    # Left merge ensures we only keep the valid patients/timesteps that survived the exclusion criteria
    df = df_clean.merge(df_raw, on=['stay_id', 'timestep'], how='left', suffixes=('', '_raw'))
    df = df.sort_values(['stay_id', 'timestep'])
    
    print("Calculating informative missingness features...")
    for col in lab_cols:
        if f"{col}_raw" not in df.columns:
            continue
        
        print(f"\tProcessing {col}...")

        raw_col = f"{col}_raw"
        
        # 1. Binary indicator: 1 if the intermediate data is NOT missing, 0 if NaN
        df[f'{col}_measured'] = df[raw_col].notna().astype(float)
        
        # 2. Hours since last measurement
        # Find the specific timestep where it was measured
        last_measured_step = df['timestep'].where(df[f'{col}_measured'] == 1)
        
        # Forward fill the timestep per patient
        last_measured_step = df.groupby('stay_id')[last_measured_step.name].ffill()
        
        # Calculate hours (timestep diff * 4 hours per timestep)
        df[f'hours_since_{col}'] = (df['timestep'] - last_measured_step) * 4
        
        # Fill NaNs for the timesteps occurring before their very first lab was drawn
        # (We use 999.0 as a distinct float so the neural network knows it means "never")
        df[f'hours_since_{col}'] = df[f'hours_since_{col}'].fillna(999.0)
        
        # Drop the temporary raw column
        df = df.drop(columns=[raw_col])
    

    print('number of missing vals in the final dataset:', df.isna().sum().sum())
    # print(f'missing vals by column:\n{df.isna().sum()}')
    out_path = '../../data/processed_files/patient_timeseries_with_features.parquet'
    df.to_parquet(out_path)
    print(f"Done! Saved to {out_path}.")
    print(f"Added {len([c for c in df.columns if 'measured' in c or 'hours_since' in c])} new features.")

if __name__ == "__main__":
    main()