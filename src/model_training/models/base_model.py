from abc import ABC, abstractmethod
import os

class BaseSepsisModel(ABC):
    def __init__(self, config, model_params, features):
        self.config = config
        self.model_params = model_params
        self.features = features
        self.model = None
        self.device = config.get('system', {}).get('device', None)

    @abstractmethod
    def build_and_train(self, df_train, df_val):
        """Train the model using the provided training and validation dataframes."""
        pass

    @abstractmethod
    def predict_proba(self, df_test):
        """
        Return positive class probabilities and the perfectly aligned true labels.
        Returns: Tuple of (y_probs, y_true)
        """
        pass
    
    @abstractmethod
    def save_model(self, model_name):
        """Save the trained model to MLflow."""
        pass
        
    def custom_func(self, m_test, m_val, m_train, y_test, y_probs):
        """
        A custom function which allows the user to add extra logic after 
        training and evaluation, and included in the mlflow experiment. 
        Examples of how this can be used include:
            - Logging additional custom plots to MLflow
            - Running additional analyses on the test set
            - Saving some plots/data locally 
        """
        pass


class BaseTabularModel(BaseSepsisModel):
    def build_and_train(self, df_train, df_val):
        X_train, y_train = df_train[self.features], df_train['target']
        X_val, y_val = df_val[self.features], df_val['target']
        self.fit_model(X_train, y_train, X_val, y_val)

    def predict_proba(self, df_test):
        X_test, y_test = df_test[self.features], df_test['target'].values
        y_probs = self.predict_model(X_test)
        return y_probs, y_test

    @abstractmethod
    def fit_model(self, X_train, y_train, X_val, y_val):
        """Implement training logic using clean arrays/dataframes."""
        pass

    @abstractmethod
    def predict_model(self, X_test):
        """Return 1D probabilities array for positive class."""
        pass


class BaseSequenceModel(BaseSepsisModel):
    def build_and_train(self, df_train, df_val):
        """
        Standardises preprocessing sequence datasets before dispatching 
        loaders to the underlying neural network fit method.
        """
        import torch
        from torch.utils.data import DataLoader
        from sklearn.preprocessing import StandardScaler
        from ..data.sequence_utils import SepsisSequenceDataset, collate_sequences

        print("Scaling features using StandardScaler...")
        self.scaler = StandardScaler()
        
        # scale safely avoiding modifying the external dataframe
        df_train_scaled = df_train.copy()
        df_val_scaled = df_val.copy()
        df_train_scaled[self.features] = self.scaler.fit_transform(df_train[self.features])
        df_val_scaled[self.features] = self.scaler.transform(df_val[self.features])
        
        train_ds = SepsisSequenceDataset(df_train_scaled, self.features)
        val_ds = SepsisSequenceDataset(df_val_scaled, self.features)
        
        # retrieve dynamic configs, fallback naturally to 256 and optimal cores
        batch_size = self.model_params.get('params', {}).get('batch_size', 256)
        n_jobs = self.config['system'].get('n_jobs', 1)
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        
        torch.set_num_threads(n_jobs)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences, num_workers=n_jobs, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences, num_workers=n_jobs)
        
        self.fit_model(train_loader, val_loader, df_train_scaled)

    def predict_proba(self, df_test):
        from torch.utils.data import DataLoader
        from ..data.sequence_utils import SepsisSequenceDataset, collate_sequences
        
        df_test_scaled = df_test.copy()
        df_test_scaled[self.features] = self.scaler.transform(df_test[self.features])
        
        test_ds = SepsisSequenceDataset(df_test_scaled, self.features)
        
        batch_size = self.model_params.get('params', {}).get('batch_size', 256)
        n_jobs = self.config['system'].get('n_jobs', 1)
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences, num_workers=n_jobs)
        
        return self.predict_model(test_loader)

    @abstractmethod
    def fit_model(self, train_loader, val_loader, df_train_scaled):
        """Train logic isolated entirely to Tensors arrays and DataLoaders."""
        pass

    @abstractmethod
    def predict_model(self, test_loader):
        """Predict outcomes isolated strictly from the original dataframe."""
        pass
