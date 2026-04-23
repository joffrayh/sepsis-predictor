import optuna
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from .base_model import BaseTabularModel
from src.model_training.custom_funcs.custom_plots import shap_explanations

class LightGBMWrapper(BaseTabularModel):
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

    def fit_model(self, X_train, y_train, X_val, y_val):
        
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
        
        if hasattr(self.model, 'best_iteration_'):
            mlflow.log_metric("best_iteration", self.model.best_iteration_)

    def predict_model(self, X_test):
        y_probs = self.model.predict_proba(X_test)[:, 1]
        return y_probs
        
    def save_model(self, model_name):
        boost_model = self.model.booster_
        mlflow.lightgbm.log_model(boost_model, name=model_name)
    
    def custom_func(self, df_train, df_val, df_test, y_test, y_probs):    
        shap_fig = shap_explanations(self.model, df_test[self.features])
        mlflow.log_figure(shap_fig, "custom_plots/shap_summary_plot.png")
        plt.close(shap_fig)