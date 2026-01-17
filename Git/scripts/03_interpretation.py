import shap
import numpy as np

class KernelLSTMWrapper:
    """
    Wrapper to handle multi-input (Sequence + Static) for SHAP KernelExplainer.
    Ensures consistent data reshaping during the attribution process.
    """
    def __init__(self, model, seq_shape, static_shape):
        self.model = model
        self.seq_shape = seq_shape
        self.static_shape = static_shape

    def predict(self, X_flat):
        n_samples = X_flat.shape[0]
        seq_size = np.prod(self.seq_shape)
        
        # Reconstruct inputs from flattened representation
        X_seq = X_flat[:, :seq_size].reshape((n_samples,) + self.seq_shape)
        X_static = X_flat[:, seq_size:]
        
        # Extract primary delirium prediction
        return self.model.predict([X_seq, X_static])[0]

def calculate_global_shap(model, X_seq, X_static, feature_names):
    """
    Computes Global Mean Absolute SHAP Importance for clinical transparency.
    """
    X_flat = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_static])
    wrapper = KernelLSTMWrapper(model, X_seq.shape[1:], (X_static.shape[1],))
    
    # Using KernelExplainer for stable interpretation of the recurrent backbone
    explainer = shap.KernelExplainer(wrapper.predict, shap.sample(X_flat, 20))
    shap_values = explainer.shap_values(X_flat)
    
    # Global attribution analysis
    importance = np.mean(np.abs(shap_values), axis=0)
    return dict(zip(feature_names, importance))