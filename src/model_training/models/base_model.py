from abc import ABC, abstractmethod

class BaseSepsisModel(ABC):
    def __init__(self, config, model_params, features):
        self.config = config
        self.model_params = model_params
        self.features = features
        self.model = None

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
