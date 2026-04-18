import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = '../../data/processed_files/patient_timeseries_with_features.parquet'
ARTIFACT_DIR = 'artifacts_v2'

def load_and_prepare_data(path, target_horizon=1, target_window=3):
    """
    load trajectory data and assemble rolling temporal features.
    predict if a patient gets sepsis in a future window (4-12h).
    rows where sepsis already began are cropped.
    """
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    
    df = pd.read_parquet(path)
    df.reset_index(inplace=True)
    
    # sorting is completely essential here so rolling functions don't merge different stays
    df = df.sort_values(['stay_id', 'timestep'])
    
    print("Engineering advanced temporal rolling features...")
    features_to_roll = [c for c in df.columns if c not in ['stay_id', 'timestep', 'sepsis']]
    
    # group by stay_id so values don't leak over the gap between different patients
    # 3 steps = 12 hours, 6 steps = 24 hours
    grouped = df.groupby('stay_id')[features_to_roll]
    
    df_roll_mean_12 = grouped.rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df_roll_std_12 = grouped.rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
    df_roll_max_12 = grouped.rolling(window=3, min_periods=1).max().reset_index(level=0, drop=True)
    df_roll_min_12 = grouped.rolling(window=3, min_periods=1).min().reset_index(level=0, drop=True)
    
    df_roll_mean_24 = grouped.rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
    df_roll_std_24 = grouped.rolling(window=6, min_periods=1).std().reset_index(level=0, drop=True)
    
    # attach generated aggregations back to master set
    df = df.join(df_roll_mean_12.add_suffix('_mean_12h'))
    df = df.join(df_roll_std_12.fillna(0).add_suffix('_std_12h')) # fill missing standard deviations (caused by singe-row windows) with 0
    df = df.join(df_roll_max_12.add_suffix('_max_12h'))
    df = df.join(df_roll_min_12.add_suffix('_min_12h'))
    df = df.join(df_roll_mean_24.add_suffix('_mean_24h'))
    df = df.join(df_roll_std_24.fillna(0).add_suffix('_std_24h'))

    # target shift: if onset triggers anywhere within window, this timestep is positive
    targets = []
    for w in range(target_window):
        targets.append(df.groupby('stay_id')['sepsis'].shift(-(target_horizon + w)))
    df['target'] = pd.concat(targets, axis=1).max(axis=1)
    
    df = df[df['sepsis'] == 0].copy()
    df['target'] = (df['target'].fillna(0) >= 1).astype(int)

    return df

def grouped_stratified_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    prevent data leakage by strictly partitioning by individual stay_id.
    """
    patient_outcomes = df.groupby('stay_id')['target'].max().reset_index()
    stay_ids = patient_outcomes['stay_id'].values
    y_patient = patient_outcomes['target'].values
    
    # isolate training pool
    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        stay_ids, y_patient, stratify=y_patient, test_size=(val_frac + test_frac), random_state=random_state
    )
    
    # resolve val/test distinction
    rel_test_frac = test_frac / (val_frac + test_frac)
    ids_val, ids_test = train_test_split(
        ids_tmp, stratify=y_tmp, test_size=rel_test_frac, random_state=random_state
    )
    
    df_train = df[df['stay_id'].isin(ids_train)].copy()
    df_val = df[df['stay_id'].isin(ids_val)].copy()
    df_test = df[df['stay_id'].isin(ids_test)].copy()
    
    return df_train, df_val, df_test

def evaluate_model(y_true, y_probs, name="Model", threshold=None):
    """compute performance with pr curve optimization for uncalibrated thresholds."""
    auprc = average_precision_score(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)
    
    if threshold is None:
        from sklearn.metrics import precision_recall_curve
        precs, recs, threshs = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (precs * recs) / (precs + recs + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        threshold = threshs[optimal_idx] if optimal_idx < len(threshs) else 0.5
        print(f"Calculated Optimal Threshold (Max F1): {threshold:.4f}")
    
    preds = (y_probs >= threshold).astype(int)
    recall = recall_score(y_true, preds)
    
    print(f"--- {name} Metrics ---")
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Recall (Sensitivity) @ Threshold {threshold:.4f}: {recall:.4f}")
    
    return {'auprc': auprc, 'auroc': auroc, 'recall': recall, 'threshold': threshold}

def tune_and_train_lgb(X_train, y_train, X_val, y_val, X_test, y_test, num_trials=10, random_state=42):
    print("\nTraining Main Model: LightGBM with Optuna...")
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 10),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_samples": trial.suggest_int("min_child_samples", 1000, 10000, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 0.7),
            "subsample_freq": 1,  # heavily enforce bagging
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.4),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 30.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "random_state": random_state,
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "verbose": -1,
            "n_jobs": -1
        }
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        preds_val = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds_val)
    
    study = optuna.create_study(direction="maximize", study_name="sepsis-lgb-tuning-v2")
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
    
    print("\nBest params discovered by Optuna:")
    print(study.best_params)
    print(f"Best Validation AUPRC: {study.best_value:.4f}")
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_val_auprc", study.best_value)
    
    # map discovered optimums to final model setup
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['random_state'] = random_state
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1
    best_params['subsample_freq'] = 1  # carry logic out of trial to prevent overfit collapse
    
    final_lgb = lgb.LGBMClassifier(**best_params)
    final_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print("\nEvaluating Final LightGBM on Test Set:")
    y_probs = final_lgb.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_probs, "LightGBM (Final)")
    
    mlflow.log_metric("lgb_test_auprc", metrics['auprc'])
    mlflow.log_metric("lgb_test_auroc", metrics['auroc'])
    
    return final_lgb

def explain_model(model, X_test, output_dir):
    print("\nGenerating SHAP explanations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # pull full shap context using tree optimization
    explainer = shap.TreeExplainer(model)
    
    # prune test set size to stop computation bottleneck
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # render overall feature importance topology
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"))
    plt.close()
    print(f"SHAP plots saved to {output_dir}/")

def main():
    mlflow.set_experiment("Sepsis_Prediction_V2")
    
    with mlflow.start_run():
        path = '../../data/processed_files/patient_timeseries_with_features.parquet'
        df = load_and_prepare_data(path, target_horizon=1, target_window=3)
        
        df_train, df_val, df_test = grouped_stratified_split(df)
        print(f"Dataset split sizes: Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)})")
        
        # separate features from structural keys
        drop_cols = ['stay_id', 'timestep', 'sepsis', 'target']
        features = [c for c in df.columns if c not in drop_cols]
        print(f"Using {len(features)} advanced temporal features for training.")
        
        X_train = df_train[features]
        y_train = df_train['target']
        X_val = df_val[features]
        y_val = df_val['target']
        X_test = df_test[features]
        y_test = df_test['target']
        
        # run primary optimized evaluation
        final_lgb = tune_and_train_lgb(X_train, y_train, X_val, y_val, X_test, y_test, num_trials=10)
        
        # log calibration geometry mapping prediction stability vs truth
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        y_probs_test_lgb = final_lgb.predict_proba(X_test)[:, 1]
        
        prob_true_lgb, prob_pred_lgb = calibration_curve(y_test, y_probs_test_lgb, n_bins=10)
        
        plt.figure(figsize=(6,6))
        plt.plot(prob_pred_lgb, prob_true_lgb, marker='o', label="LightGBM")
        plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("calibration curve (reliability diagram)")
        plt.legend()
        calib_path = os.path.join(ARTIFACT_DIR, "calibration_curve_v2.png")
        plt.savefig(calib_path)
        plt.close()
        mlflow.log_artifact(calib_path)
        
        explain_model(final_lgb, X_test, ARTIFACT_DIR)
        mlflow.log_artifacts(ARTIFACT_DIR, artifact_path="shap_plots")

if __name__ == "__main__":
    main()