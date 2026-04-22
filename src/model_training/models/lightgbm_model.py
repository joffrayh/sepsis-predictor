import numpy as np
import optuna
import mlflow
import lightgbm as lgb
import joblib
from sklearn.metrics import average_precision_score
from .base_model import BaseSepsisModel

class LightGBMWrapper(BaseSepsisModel):
    def __init__(self, config, model_params, features):
        super().__init__(config, model_params, features)
        self.search_space = model_params.get('search_space', {})
        
    def _suggest_param(self, trial, name, space):
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
            raise ValueError(f"param type '{p_type}' not understood.")

    def build_and_train(self, df_train, df_val):
        
        X_train, y_train = df_train[self.features], df_train['target']
        X_val, y_val = df_val[self.features], df_val['target']
        
        opt_config = self.config.get('tabular', {}).get('optimisation', {})
        sys_config = self.config['system']
        rnd_state = self.config['experiment']['random_state']

        def objective(trial):
            kwargs = {}
            for p_name, p_space in self.search_space.items():
                kwargs[p_name] = self._suggest_param(trial, p_name, p_space)

            kwargs['random_state'] = rnd_state
            kwargs['n_jobs'] = sys_config['n_jobs']
            kwargs.update({'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt', 'verbose': -1, 'subsample_freq': 1})

            model = lgb.LGBMClassifier(**kwargs)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            preds_val = model.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, preds_val)

        study = optuna.create_study(direction="maximize", study_name="sepsis-lightgbm-tuning")
        study.optimize(objective, n_trials=opt_config.get('num_trials', 20), show_progress_bar=True)

        best_kwargs = study.best_params.copy()
        best_kwargs['random_state'] = rnd_state
        best_kwargs['n_jobs'] = sys_config['n_jobs']
        best_kwargs.update({'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt', 'verbose': -1, 'subsample_freq': 1})

        mlflow.log_params(best_kwargs)
        mlflow.log_metric("best_val_auprc", study.best_value)

        self.model = lgb.LGBMClassifier(**best_kwargs)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

    def predict_proba(self, df_test):
        y_probs = self.model.predict_proba(df_test[self.features])[:, 1]
        y_test = df_test['target'].values
        return y_probs, y_test
        
    def save_model(self, model_name):
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(self.model, artifact_path=model_name)
        
    def get_custom_plots(self, df_test):
        from ..utils.metrics import explain_model
        plots = {}
        shap_fig = explain_model(self.model, df_test[self.features], "LightGBM")
        if shap_fig is not None:
            plots["shap_beeswarm"] = shap_fig
        return plots
