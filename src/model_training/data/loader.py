import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_and_prepare_data(config, is_tabular=True):
    path_info = config['data']['path']
    print(f"loading dataset from {path_info}...")
    if not os.path.exists(path_info):
        path_info = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", path_info))

    df = pd.read_parquet(path_info)
    
    if 'stay_id' not in df.columns:
        df.reset_index(inplace=True)
        
    df = df.sort_values(['stay_id', 'timestep'])

    structural_cols = ['stay_id', 'timestep', 'sepsis']
    base_features = [c for c in df.columns if c not in structural_cols]

    if is_tabular and 'tabular' in config and 'feature_engineering' in config['tabular']:
        feat_cfg = config['tabular']['feature_engineering']

        if feat_cfg.get('use_lags', False):
            df = _apply_lag(feat_cfg, df, base_features)
            
        if feat_cfg.get('use_rolling', False):
            timestep_hours = config['data']['timestep_duration_hours']
            rolling_windows = feat_cfg.get('rolling_windows_hours', [])
            df = _apply_rolling(feat_cfg, df, base_features, rolling_windows, timestep_hours)

    targets = []
    target_horizon = config['data']['target_horizon']
    target_window = config['data']['target_window']
    
    for w in range(target_window):
        targets.append(df.groupby('stay_id')['sepsis'].shift(-(target_horizon + w)))
        
    df['target'] = pd.concat(targets, axis=1).max(axis=1)
    
    df = df[df['sepsis'] == 0].copy()
    df['target'] = (df['target'].fillna(0) >= 1).astype(int)

    return df, base_features


def _apply_lag(feat_cfg, df, base_features):
    num_lags = feat_cfg.get('num_lags', 0)
    exclude_lags = set(feat_cfg.get('lag_exclude_cols', []))
    lag_features = [c for c in base_features if c not in exclude_lags]
    
    print(f"engineering lag and diff features (up to {num_lags} timesteps)...")
    for lag in range(1, num_lags + 1):
        for col in lag_features:
            df[f'{col}_lag{lag}'] = df.groupby('stay_id')[col].shift(lag)
            df[f'{col}_diff{lag}'] = df[col] - df[f'{col}_lag{lag}']

    return df


def _apply_rolling(feat_cfg, df, base_features, rolling_windows, timestep_hours):
    rolling_windows = feat_cfg.get('rolling_windows_hours', [])
    exclude_rolling = set(feat_cfg.get('rolling_exclude_cols', []))
    roll_features = [c for c in base_features if c not in exclude_rolling]
    
    print(f"engineering rolling statistics for windows: {rolling_windows}h...")
    df[roll_features] = df[roll_features].astype('float32')
    new_feats = {}
    grouped = df.groupby('stay_id', sort=False)[roll_features]
    
    for rh in tqdm(rolling_windows):
        window_steps = max(1, int(rh / timestep_hours))
        roller = grouped.rolling(window=window_steps, min_periods=1)
        
        # mean
        stat_df = roller.mean().reset_index(level=0, drop=True).astype('float32')
        for i, c in enumerate(roll_features):
            new_feats[f"{c}_mean_{rh}h"] = stat_df.iloc[:, i].values
        del stat_df; gc.collect()
        
        # std
        stat_df = roller.std().fillna(0).reset_index(level=0, drop=True).astype('float32')
        for i, c in enumerate(roll_features):
            new_feats[f"{c}_std_{rh}h"] = stat_df.iloc[:, i].values
        del stat_df; gc.collect()
        
        # max
        stat_df = roller.max().reset_index(level=0, drop=True).astype('float32')
        for i, c in enumerate(roll_features):
            new_feats[f"{c}_max_{rh}h"] = stat_df.iloc[:, i].values
        del stat_df; gc.collect()
        
        # min
        stat_df = roller.min().reset_index(level=0, drop=True).astype('float32')
        for i, c in enumerate(roll_features):
            new_feats[f"{c}_min_{rh}h"] = stat_df.iloc[:, i].values
        del stat_df; gc.collect()
        del roller; gc.collect()

    df = pd.concat([df, pd.DataFrame(new_feats, index=df.index)], axis=1)
    del new_feats
    gc.collect()

    return df


def grouped_stratified_split(df, config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frac = config['data']['split']['train_frac']
    val_frac = config['data']['split']['val_frac']
    test_frac = config['data']['split']['test_frac']
    rnd_state = config['experiment']['random_state']

    patient_outcomes = df.groupby('stay_id')['target'].max().reset_index()
    stay_ids = patient_outcomes['stay_id'].values
    y_patient = patient_outcomes['target'].values
    
    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        stay_ids, y_patient, stratify=y_patient, test_size=(val_frac + test_frac), random_state=rnd_state
    )
    
    rel_test_frac = test_frac / (val_frac + test_frac)
    ids_val, ids_test = train_test_split(
        ids_tmp, stratify=y_tmp, test_size=rel_test_frac, random_state=rnd_state
    )
    
    df_train = df[df['stay_id'].isin(ids_train)].copy()
    df_val = df[df['stay_id'].isin(ids_val)].copy()
    df_test = df[df['stay_id'].isin(ids_test)].copy()
    
    return df_train, df_val, df_test
