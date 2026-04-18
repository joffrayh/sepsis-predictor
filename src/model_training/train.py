import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = '../../data/processed_files/patient_timeseries_with_features.parquet'
ARTIFACT_DIR = 'artifacts'

def load_and_prepare_data(path, target_horizon=1, target_window=3, num_lags=3):
    """
    load trajectory data and construct future target.
    predict if patient develops sepsis in future window (4-12 hours).
    drop rows where sepsis already started for early-warning focus.
    """
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    
    df = pd.read_parquet(path)
    df.reset_index(inplace=True)

    # sort by patient and time for accurate row shifts
    df = df.sort_values(['stay_id', 'timestep'])
    
    if num_lags > 0:
        features_to_lag = [c for c in df.columns if c not in ['stay_id', 'timestep', 'sepsis']]
        for lag in range(1, num_lags + 1):
            for col in features_to_lag:
                df[f'{col}_lag{lag}'] = df.groupby('stay_id')[col].shift(lag)
                # compute timestep diffs so trees see deterioration trends directly
                df[f'{col}_diff{lag}'] = df[col] - df[f'{col}_lag{lag}']
    
    # shift target backward to create positive class if onset falls in window
    targets = []
    for w in range(target_window):
        targets.append(df.groupby('stay_id')['sepsis'].shift(-(target_horizon + w)))
    
    df['target'] = pd.concat(targets, axis=1).max(axis=1)
    
    # keep only pre-sepsis timesteps
    df = df[df['sepsis'] == 0].copy()
    
    # assume nans at the end of stay are negative
    df['target'] = (df['target'].fillna(0) >= 1).astype(int)

    return df

def grouped_stratified_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    split dataset grouping by stay_id to stop leakage,
    stratifying by whether patient ever gets sepsis.
    """
    # summarize sepsis occurrence per patient
    patient_outcomes = df.groupby('stay_id')['target'].max().reset_index()
    
    stay_ids = patient_outcomes['stay_id'].values
    y_patient = patient_outcomes['target'].values
    
    # split out training cohort
    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        stay_ids, y_patient, stratify=y_patient, test_size=(val_frac + test_frac), random_state=random_state
    )
    
    # split remainder into validation and test
    rel_test_frac = test_frac / (val_frac + test_frac)
    ids_val, ids_test = train_test_split(
        ids_tmp, stratify=y_tmp, test_size=rel_test_frac, random_state=random_state
    )
    
    # map patient ids back to timeseries
    df_train = df[df['stay_id'].isin(ids_train)].copy()
    df_val = df[df['stay_id'].isin(ids_val)].copy()
    df_test = df[df['stay_id'].isin(ids_test)].copy()
    
    return df_train, df_val, df_test

def evaluate_model(y_true, y_probs, name="Model", threshold=None):
    """evaluate model and optionally calculate optimal pr curve threshold."""
    auprc = average_precision_score(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)
    
    # derive threshold that maximizes f1 if not provided
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

def run_baseline(X_train, y_train, X_test, y_test):
    print("\nTraining Baseline: Logistic Regression...")
    
    # impute missing values with mean for baseline logistic regression
    X_train_imputed = X_train.fillna(X_train.mean())
    X_test_imputed = X_test.fillna(X_train.mean())
    
    # heavily penalize missing the minority class
    scale_w = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
    lr = LogisticRegression(class_weight={0: 1.0, 1: scale_w}, max_iter=1000)
    lr.fit(X_train_imputed, y_train)
    
    y_probs = lr.predict_proba(X_test_imputed)[:, 1]
    evaluate_model(y_test, y_probs, "Logistic Regression (Baseline)")
    
    return lr

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
            "subsample_freq": 1,  # force bagging per iteration
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
    
    study = optuna.create_study(direction="maximize", study_name="sepsis-lgb-tuning")
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
    
    print("\nBest params discovered by Optuna:")
    print(study.best_params)
    print(f"Best Validation AUPRC: {study.best_value:.4f}")
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("lgb_best_val_auprc", study.best_value)
    
    # combine optimal parameters with static operations
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['random_state'] = random_state
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1
    best_params['subsample_freq'] = 1  # manually carry over bagging variable
    
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

def tune_and_train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, num_trials=10, random_state=42):
    print("\nTraining Alternative Model: XGBoost with Optuna...")
    
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_weight": trial.suggest_float("min_child_weight", 10.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 0.7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.4),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 30.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "random_state": random_state,
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "n_jobs": -1
        }
        
        clf = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds_val = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds_val)
    
    study = optuna.create_study(direction="maximize", study_name="sepsis-xgb-tuning")
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
    
    print("\nBest params discovered by Optuna (XGBoost):")
    print(study.best_params)
    print(f"Best Validation AUPRC (XGBoost): {study.best_value:.4f}")
    
    mlflow.log_params({f"xgb_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("xgb_best_val_auprc", study.best_value)
    
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['random_state'] = random_state
    best_params['n_jobs'] = -1
    
    final_xgb = xgb.XGBClassifier(**best_params, early_stopping_rounds=50)
    final_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print("\nEvaluating Final XGBoost on Test Set:")
    y_probs = final_xgb.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_probs, "XGBoost (Final)")
    
    mlflow.log_metric("xgb_test_auprc", metrics['auprc'])
    mlflow.log_metric("xgb_test_auroc", metrics['auroc'])
    
    return final_xgb


def explain_model(model, X_test, output_dir):
    print("\nGenerating SHAP explanations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # gives exact shap values for tree ensembles
    explainer = shap.TreeExplainer(model)
    
    # sample subset to speed up shap calculations
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # global beeswarm plot showing feature impact
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"))
    plt.close()
    
    print(f"SHAP plots saved to {output_dir}/")

def main():
    mlflow.set_experiment("Sepsis_Prediction")
    
    with mlflow.start_run():
        df = load_and_prepare_data(DATA_PATH, target_horizon=1, target_window=3, num_lags=3)
        
        df_train, df_val, df_test = grouped_stratified_split(df)
        print(f"Dataset splits: Train ({len(df_train)} rows), Val ({len(df_val)} rows), Test ({len(df_test)} rows)")
        
        drop_cols = ['stay_id', 'timestep', 'sepsis', 'target']
        features = [c for c in df.columns if c not in drop_cols]
        
        X_train = df_train[features]
        y_train = df_train['target']
        X_val = df_val[features]
        y_val = df_val['target']
        X_test = df_test[features]
        y_test = df_test['target']
        
        # simplistic baselines to judge uplift
        run_baseline(X_train, y_train, X_test, y_test)
        
        # run xgboost to compare with lightgbm
        tune_and_train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, num_trials=10)
        
        # train and tune lightgbm
        final_lgb = tune_and_train_lgb(X_train, y_train, X_val, y_val, X_test, y_test, num_trials=10)
        
        # explainability plots
        mlflow.log_artifacts(ARTIFACT_DIR, artifact_path="shap_plots")

if __name__ == "__main__":
    main()
