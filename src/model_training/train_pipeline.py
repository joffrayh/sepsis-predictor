import pandas as pd
import numpy as np
import os
import argparse
import yaml
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_recall_curve, precision_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_and_prepare_data(config):
    """
    dynamically engineer lag and rolling features according to yaml spec.
    """
    path_info = config['data']['path']
    print(f"loading dataset from {path_info}...")
    if not os.path.exists(path_info):
        path_info = os.path.abspath(os.path.join(os.path.dirname(__file__), path_info))

    df = pd.read_parquet(path_info)

    df = df.sort_values(['stay_id', 'timestep'])

    structural_cols = ['stay_id', 'timestep', 'sepsis']
    base_features = [c for c in df.columns if c not in structural_cols]

    print(f'\nFeatures before engineering: \n{base_features}\n')

    feat_cfg = config['ml_pipeline']['feature_engineering']
    timestep_hours = config['data']['timestep_duration_hours']

    # number of features with just lag would be: len(lag_features) * num_lags * 2 (for lag and diff)
    if feat_cfg.get('use_lags', False):
        num_lags = feat_cfg.get('num_lags', 0)
        exclude_lags = set(feat_cfg.get('lag_exclude_cols', []))
        
        lag_features = [c for c in base_features if c not in exclude_lags]
        
        print(f"engineering lag and diff features (up to {num_lags} timesteps)...")
        for lag in range(1, num_lags + 1):
            for col in lag_features:
                df[f'{col}_lag{lag}'] = df.groupby('stay_id')[col].shift(lag)
                df[f'{col}_diff{lag}'] = df[col] - df[f'{col}_lag{lag}']

    # number of features with just rolling would be: len(roll_features) * num_windows * 4 (for mean, std, max, min)
    if feat_cfg.get('use_rolling', False):
        rolling_windows = feat_cfg.get('rolling_windows_hours', [])
        exclude_rolling = set(feat_cfg.get('rolling_exclude_cols', []))
        
        roll_features = [c for c in base_features if c not in exclude_rolling]
        
        print(f"engineering rolling statistics for windows: {rolling_windows}h...")
        
        # cast to float32 before any rolling (halves RAM)
        df[roll_features] = df[roll_features].astype('float32')

        all_rolled = []  # collect outside loop, single join at the end

        grouped = df.groupby('stay_id', sort=False)[roll_features]
        
        for rh in rolling_windows:
            # mathematically map desired real time (yaml hours) directly into sequential steps
            window_steps = max(1, int(rh / timestep_hours))
            roller = grouped.rolling(window=window_steps, min_periods=1)

            # compute each stat separately
            stats = {
                'mean': roller.mean(),
                'std':  roller.std().fillna(0),
                'max':  roller.max(),
                'min':  roller.min(),
            }

            parts = []
            for stat_name, stat_df in stats.items():
                stat_df.columns = [f"{c}_{stat_name}_{rh}h" for c in roll_features]
                stat_df = stat_df.reset_index(level=0, drop=True)
                parts.append(stat_df)
                del stat_df 

            all_rolled.append(pd.concat(parts, axis=1))
            del parts, stats

        df = df.join(pd.concat(all_rolled, axis=1))
        del all_rolled

    # early warning prediction horizon generation
    targets = []
    target_horizon = config['data']['target_horizon']
    target_window = config['data']['target_window']
    
    for w in range(target_window):
        targets.append(df.groupby('stay_id')['sepsis'].shift(-(target_horizon + w)))
        
    df['target'] = pd.concat(targets, axis=1).max(axis=1)
    
    # strictly drop timesteps that happen sequentially after an onset warning is already fired
    df = df[df['sepsis'] == 0].copy()
    df['target'] = (df['target'].fillna(0) >= 1).astype(int)

    # -1 for the target
    print(f"Total number of features after engineering: {len(df.columns) - len(structural_cols) - 1}")

    return df


def suggest_param(trial, name, space):
    """safely unroll tuning matrices given inside the yaml dictionary."""
    low, high, p_type = space
    if p_type == 'int':
        return trial.suggest_int(name, low, high)
    elif p_type == 'int_log':
        return trial.suggest_int(name, low, high, log=True)
    elif p_type == 'float':
        return trial.suggest_float(name, low, high)
    elif p_type == 'log':
        return trial.suggest_float(name, low, high, log=True)
    else:
        raise ValueError(f"param type '{p_type}' in config.yaml is not understood.")


def get_model(model_name, model_kwargs):
    """can easily add new models"""
    m_name = model_name.lower()
    if m_name == 'lightgbm':
        return lgb.LGBMClassifier(**model_kwargs)
    elif m_name == 'xgboost':
        return xgb.XGBClassifier(**model_kwargs)
    else:
        raise ValueError(f"architecture for model {model_name} is not loaded explicitly in generic factory.")


def fit_model(model_name, model, X_train, y_train, X_val, y_val):
    """encapsulates algorithm-specific early stopping constraints from the yaml abstraction."""
    m_name = model_name.lower()
    if m_name == 'lightgbm':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
    elif m_name == 'xgboost':
        model.set_params(early_stopping_rounds=30)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_probs, name="Model"):
    auprc = average_precision_score(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)

    precs, recs, threshs = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precs * recs) / (precs + recs + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    threshold = threshs[optimal_idx] if optimal_idx < len(threshs) else 0.5

    preds = (y_probs >= threshold).astype(int)
    recall = recall_score(y_true, preds)
    precision = precision_score(y_true, preds)
    f1 = f1_scores[optimal_idx] if optimal_idx < len(f1_scores) else 0.0

    print(f"--- {name} Metrics ---")
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 Score: {f1:.4f} (threshold: {threshold:.4f})")
    print(f"\tPrecision: {precision:.4f}")
    print(f"\tRecall: {recall:.4f}")

    return {'auprc': auprc, 'auroc': auroc, 'f1': f1, 'threshold': threshold}


def optimise_and_train(model_name, config, X_train, y_train, X_val, y_val):
    """safely bridge independent optuna runs and mlflow tracking loops cleanly for injected model."""
    print(f"\nRunning optimisation for {model_name.upper()}...")
    opt_config = config['ml_pipeline']['optimisation']
    sys_config = config['system']
    model_config = config['ml_pipeline']['models'][model_name]
    search_space = model_config['search_space']

    def objective(trial):
        kwargs = {}
        for p_name, p_space in search_space.items():
            kwargs[p_name] = suggest_param(trial, p_name, p_space)

        kwargs['random_state'] = config['experiment']['random_state']
        kwargs['n_jobs'] = sys_config['n_jobs']

        if model_name.lower() == 'lightgbm':
            kwargs.update({'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt', 'verbose': -1, 'subsample_freq': 1})
        elif model_name.lower() == 'xgboost':
            kwargs.update({'objective': 'binary:logistic', 'eval_metric': 'aucpr'})

        model = get_model(model_name, kwargs)
        model = fit_model(model_name, model, X_train, y_train, X_val, y_val)

        preds_val = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds_val)

    study_name = f"sepsis-{model_name}-tuning"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=opt_config['num_trials'], show_progress_bar=True)

    print(f"[{model_name.upper()}] Best validation discovered AUPRC: {study.best_value:.4f}")
    
    # securely map optimums directly to the final production model refit
    best_kwargs = study.best_params.copy()
    best_kwargs['random_state'] = config['experiment']['random_state']
    best_kwargs['n_jobs'] = sys_config['n_jobs']

    if model_name.lower() == 'lightgbm':
        best_kwargs.update({'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt', 'verbose': -1, 'subsample_freq': 1})
    elif model_name.lower() == 'xgboost':
        best_kwargs.update({'objective': 'binary:logistic', 'eval_metric': 'aucpr'})

    mlflow.log_params(best_kwargs)
    mlflow.log_metric("best_val_auprc", study.best_value)

    final_model = get_model(model_name, best_kwargs)
    final_model = fit_model(model_name, final_model, X_train, y_train, X_val, y_val)

    return final_model


def explain_model(model, X_test, output_dir, model_name="model"):
    print(f"\nGenerating SHAP explanations for {model_name}...")
    
    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"shap_beeswarm.png")
    plt.savefig(file_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Config-driven Sepsis Training Framework")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML configuration")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config))
    config = load_config(config_path)
    
    exp_base = config['experiment']['base_name']
    
    df = load_and_prepare_data(config)
    
    print("\nSplitting data...")
    # block leakage using grouped stratification rules
    patient_outcomes = df.groupby('stay_id')['target'].max().reset_index()
    stay_ids = patient_outcomes['stay_id'].values
    y_patient = patient_outcomes['target'].values

    trn_split = config['data']['split']['train_frac']
    val_split = config['data']['split']['val_frac']
    tst_split = config['data']['split']['test_frac']
    rnd_state = config['experiment']['random_state']

    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        stay_ids, y_patient, stratify=y_patient, test_size=(val_split + tst_split), random_state=rnd_state
    )
    rel_test_frac = tst_split / (val_split + tst_split)
    ids_val, ids_test = train_test_split(
        ids_tmp, stratify=y_tmp, test_size=rel_test_frac, random_state=rnd_state
    )

    df_train = df[df['stay_id'].isin(ids_train)].copy()
    df_val = df[df['stay_id'].isin(ids_val)].copy()
    df_test = df[df['stay_id'].isin(ids_test)].copy()
    
    print(f"Dataset split: Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)})\n")

    drop_cols = ['stay_id', 'timestep', 'sepsis', 'target']
    features = [c for c in df.columns if c not in drop_cols]

    X_train, y_train = df_train[features], df_train['target']
    X_val, y_val = df_val[features], df_val['target']
    X_test, y_test = df_test[features], df_test['target']

    artifact_dir = config['system'].get('artifact_dir', 'artifacts')
    os.makedirs(artifact_dir, exist_ok=True)

    models_to_run = [m for m, cfg in config['ml_pipeline']['models'].items() if cfg.get('run', False)]
    
    if not models_to_run:
        print("No models were set directly to 'run: true' inside your config.yaml.")
        return

    # suffix based on engineered variables
    modifier_parts = []
    if config['ml_pipeline']['feature_engineering']['use_lags']:
        modifier_parts.append("Lags")
    if config['ml_pipeline']['feature_engineering']['use_rolling']:
        modifier_parts.append("Roll")
    modifier = "_".join(modifier_parts) if modifier_parts else "Base"

    for model_name in models_to_run:
        experiment_target = f"{exp_base}_{modifier}_{model_name.upper()}"
        
        print("\n==============================================")
        print(f" STARTING EXPERIMENT RUN: {experiment_target}")
        print("==============================================")

        mlflow.set_experiment(experiment_target)
        
        with mlflow.start_run(run_name=f"{model_name.upper()}_Execution"):
            # snapshot the exact engineering toggles used
            mlflow.log_param("model_type", model_name.upper())
            mlflow.log_param("use_lags", config['ml_pipeline']['feature_engineering']['use_lags'])
            mlflow.log_param("use_rolling", config['ml_pipeline']['feature_engineering']['use_rolling'])
            mlflow.log_param("total_features_used", len(features))

            final_model = optimise_and_train(model_name, config, X_train, y_train, X_val, y_val)

            print(f"\nEvaluating best {model_name.upper()} on test set...")
            y_probs = final_model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_probs, f"{model_name.upper()} Test Performance")

            mlflow.log_metric("test_auprc", metrics['auprc'])
            mlflow.log_metric("test_auroc", metrics['auroc'])
            mlflow.log_metric("test_f1", metrics['f1'])

            
            curr_artifact_dir = os.path.join(artifact_dir, model_name.lower())
            os.makedirs(curr_artifact_dir, exist_ok=True)
            # visually export the uncalibrated raw probabilistic curve
            prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
            plt.figure(figsize=(6, 6))
            plt.plot(prob_pred, prob_true, marker='o', label=model_name.upper())
            plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly Calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"Clinical Reliability Trace ({model_name.upper()})")
            plt.legend()

            calib_path = os.path.join(curr_artifact_dir, f"calibration_curve.png")
            plt.savefig(calib_path)
            plt.close()
            mlflow.log_artifact(calib_path)

            # trigger explanation module natively inside the evaluation loop
            explain_model(final_model, X_test, curr_artifact_dir, model_name)
            
            # send artifacts to mlflow's remote storage backend
            if os.path.exists(os.path.join(artifact_dir, f"shap_beeswarm.png")):
                mlflow.log_artifact(os.path.join(artifact_dir, f"shap_beeswarm.png"))

            if config['system'].get('save_models', False):
                model_save_path = os.path.join(curr_artifact_dir, f"best_{model_name}.pkl")
                joblib.dump(final_model, model_save_path)
                mlflow.log_artifact(model_save_path)
                print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()