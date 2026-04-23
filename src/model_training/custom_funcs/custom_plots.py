import shap
import matplotlib.pyplot as plt

def shap_explanations(model, X_test):
    print(f"\nGenerating SHAP explanations for XGBoost...")

    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()

    return fig