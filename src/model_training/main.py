import os
import sys
from pathlib import Path

import argparse
import yaml
import mlflow
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.model_training.data.loader import load_and_prepare_data, grouped_stratified_split
from src.model_training.models.factory import get_model
from src.model_training.utils.metrics import evaluate_model, plot_calibration_curve, plot_pr_curve, plot_roc_curve

def load_config(args_config) -> dict:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args_config))
    with open(config_path, 'r') as f:
        return yaml.safe_load(f), config_path

def main():

    # loading the configs
    parser = argparse.ArgumentParser(description="Config-driven Sepsis Training Framework")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML configuration")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    

    # getting the model
    active_model = config['experiment'].get('active_model')

    if not active_model:
        print("No 'active_model' specified in config.yaml under 'experiment'.")
        return

    is_tabular = False
    model_cfg = None
    if 'tabular' in config and 'models' in config['tabular'] and active_model in config['tabular']['models']:
        is_tabular = True
        model_cfg = config['tabular']['models'][active_model]
    elif 'sequence' in config and active_model in config['sequence']:
        is_tabular = False
        model_cfg = config['sequence'][active_model]
    else:
        print(f"Model '{active_model}' not found in tabular or sequence config sections.")
        return


    # loading and spllitting the data
    df, base_features = load_and_prepare_data(config, is_tabular=is_tabular)

    print("\nSplitting data...")
    df_train, df_val, df_test = grouped_stratified_split(df, config)
    print(f"Dataset split: Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)})\n")


    # naming the experiment based on config settings
    exp_base = config['experiment']['base_name']
    experiment_target = f"{exp_base}_{active_model.upper()}"
    if is_tabular:
        modifier_parts = []
        if 'feature_engineering' in config['tabular']:
            feat_cfg = config['tabular']['feature_engineering']
            if feat_cfg.get('use_lags'):
                modifier_parts.append("Lags")
            if feat_cfg.get('use_rolling'):
                modifier_parts.append("Roll")
        modifier = "_".join(modifier_parts) if modifier_parts else "Base"
        experiment_target = f"{exp_base}_{modifier}_{active_model.upper()}"


    
    print("\n==============================================")
    print(f" STARTING EXPERIMENT RUN: {experiment_target}")
    print("==============================================")

    mlflow.set_experiment(experiment_target)
    
    with mlflow.start_run(run_name=f"{active_model.upper()}_Execution"):
        
        # logging parameters and artifacts
        mlflow.log_param("model_type", active_model.upper())
        mlflow.log_param("random_state", config['experiment']['random_state'])
        
        mlflow.log_artifact(config_path, artifact_path="reproduction_configs")
        
        if is_tabular and 'tabular' in config and 'feature_engineering' in config['tabular']:
            for k, v in config['tabular']['feature_engineering'].items():
                if "exclude" not in k: 
                    mlflow.log_param(f"feat_eng_{k}", v)
        
        if model_cfg:
            for k, v in model_cfg.items():
                for sub_k, sub_v in v.items():
                    mlflow.log_param(f"{k}_{sub_k}", sub_v)
        

        # extracting learning features natively dynamically available
        drop_cols = ['stay_id', 'timestep', 'sepsis', 'target']
        train_features = [c for c in df_train.columns if c not in drop_cols]
        
        # log counts
        mlflow.log_metric("train_samples", len(df_train))
        mlflow.log_metric("val_samples", len(df_val))
        mlflow.log_metric("test_samples", len(df_test))
        mlflow.log_metric("feature_count", len(train_features))

        # training the model
        model_wrapper = get_model(active_model, config, model_cfg, train_features)

        model_wrapper.build_and_train(df_train, df_val)

        # predicting and evaluating on the test set
        print(f"\nEvaluating best {active_model.upper()} on test set...")
        y_probs, y_test = model_wrapper.predict_proba(df_test)
        
        metrics = evaluate_model(y_test, y_probs, f"{active_model.upper()} Test Performance")
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})


        calib_fig = plot_calibration_curve(y_test, y_probs, active_model.upper())
        mlflow.log_figure(calib_fig, f"calibration_curve_{active_model.lower()}.png")
        plt.close(calib_fig)

        pr_fig = plot_pr_curve(y_test, y_probs, active_model.upper())
        mlflow.log_figure(pr_fig, f"pr_curve_{active_model.lower()}.png")
        plt.close(pr_fig)

        roc_fig = plot_roc_curve(y_test, y_probs, active_model.upper())
        mlflow.log_figure(roc_fig, f"roc_curve_{active_model.lower()}.png")
        plt.close(roc_fig)

        model_wrapper.custom_func(df_train, df_val, df_test, y_test, y_probs)

        model_wrapper.save_model(active_model.lower())

if __name__ == "__main__":
    main()
