import pandas as pd
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_prepare_data(config):
    """
    load data and set sequence targets.
    """
    path_info = config['data']['path']
    target_horizon = config['data']['target_horizon']
    target_window = config['data']['target_window']
    
    print(f"Loading data from {path_info}...")
    if not os.path.exists(path_info):
        path_info = os.path.abspath(os.path.join(os.path.dirname(__file__), path_info))
    
    df = pd.read_parquet(path_info)
    
    # stay_id and timestep are accessible if they were indices
    if 'stay_id' not in df.columns:
        df.reset_index(inplace=True)
        
    df = df.sort_values(['stay_id', 'timestep'])
    
    # shift target backward to capture onset within the early-warning window
    targets = []
    for w in range(target_window):
        targets.append(df.groupby('stay_id')['sepsis'].shift(-(target_horizon + w)))
    df['target'] = pd.concat(targets, axis=1).max(axis=1)
    
    # explicitly isolate the pre-onset computational window
    df = df[df['sepsis'] == 0].copy()
    df['target'] = (df['target'].fillna(0) >= 1).astype(float)

    return df

def grouped_stratified_split(df, config):
    """
    partition dataset strictly by individual stay_id to guarantee no data leakage across splits.
    """
    train_frac = config['split']['train_frac']
    val_frac = config['split']['val_frac']
    test_frac = config['split']['test_frac']
    rnd_state = config['split']['random_state']

    patient_outcomes = df.groupby('stay_id')['target'].max().reset_index()
    stay_ids = patient_outcomes['stay_id'].values
    y_patient = patient_outcomes['target'].values
    
    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        stay_ids, y_patient, stratify=y_patient, test_size=(val_frac + test_frac), random_state=rnd_state
    )
    
    rel_test_frac = test_frac / (val_frac + test_frac)
    ids_val, ids_test = train_test_split(
        ids_tmp, stratify=y_tmp, test_size=rel_test_frac, random_state=rnd_state
    )
    
    df_train = df[df['stay_id'].isin(ids_train)].copy()
    df_val = df[df['stay_id'].isin(ids_val)].copy()
    df_test = df[df['stay_id'].isin(ids_test)].copy()
    
    return df_train, df_val, df_test

class SepsisSequenceDataset(Dataset):
    """
    dataset class to for patient trajectories for lstm.
    """
    def __init__(self, df, features):
        self.grouped = list(df.groupby('stay_id'))
        self.features = features
        
    def __len__(self):
        return len(self.grouped)
        
    def __getitem__(self, idx):
        stay_id, group = self.grouped[idx]
        X = torch.tensor(group[self.features].values, dtype=torch.float32)
        y = torch.tensor(group['target'].values, dtype=torch.float32)
        return X, y

def collate_sequences(batch):
    """
    pad variable-length patient stays into rectangular tensors for batch processing.
    generates a boolean mask alongside targets so the loss function ignores synthetic zeros.
    """
    Xs, ys = zip(*batch)
    X_pad = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0.0)
    mask_pad = pad_sequence([torch.ones_like(y) for y in ys], batch_first=True, padding_value=0.0)
    return X_pad, y_pad, mask_pad

class SepsisLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, num_heads=4, fc_dim=32):
        super(SepsisLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # apply causal mask to the attention layer to entirely prevent the network 
        # from "cheating" by absorbing diagnostic clues from future timesteps.
        seq_len = lstm_out.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=causal_mask)
        
        # residual connection stabilizes gradients during deep temporal propagation
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        logits = self.classifier(lstm_out).squeeze(-1)
        return logits

def evaluate_model(y_true, y_probs, name="Model", threshold=None):
    """compute performance with pr curve optimization for dynamic uncalibrated logit thresholds."""
    from sklearn.metrics import precision_recall_curve, precision_score
    auprc = average_precision_score(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)

    precs, recs, threshs = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precs * recs) / (precs + recs + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    
    if threshold is None:
        threshold = threshs[optimal_idx] if optimal_idx < len(threshs) else 0.5
        print(f"Calculated Optimal Threshold (Max F1): {threshold:.4f}")
    
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

def main():
    parser = argparse.ArgumentParser(description="Config-driven LSTM Training Framework")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML configuration")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config))
    config = load_config(config_path)

    # check if lstm is enabled
    lstm_config = config.get('models', {}).get('lstm', {})
    if not lstm_config.get('run', False):
        print("LSTM is disabled in config.yaml.")
        return

    lstm_params = lstm_config.get('params', {})
    
    exp_base = config['experiment']['base_name']
    mlflow.set_experiment(f"{exp_base}_LSTM")
    
    artifact_dir = config['system'].get('artifact_dir', 'artifacts_pipeline')
    os.makedirs(artifact_dir, exist_ok=True)
    
    avail_cores = os.cpu_count() or 1
    num_cores = config['system'].get('n_jobs', avail_cores)
    num_cores = avail_cores if num_cores < 0 else num_cores
    
    torch.set_num_threads(num_cores)
    print(f"Set PyTorch to use {num_cores} CPU threads.")
    
    with mlflow.start_run(run_name="LSTM_Execution"):
        df = load_and_prepare_data(config)
        
        df_train, df_val, df_test = grouped_stratified_split(df, config)
        print(f"Dataset split sizes: Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)})")
        
        # separate diagnostic variables from structural indices
        drop_cols = ['stay_id', 'timestep', 'sepsis', 'target']
        features = [c for c in df.columns if c not in drop_cols]
        
        print("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        df_train[features] = scaler.fit_transform(df_train[features])
        df_val[features] = scaler.transform(df_val[features])
        df_test[features] = scaler.transform(df_test[features])
        
        train_ds = SepsisSequenceDataset(df_train, features)
        val_ds = SepsisSequenceDataset(df_val, features)
        test_ds = SepsisSequenceDataset(df_test, features)
        
        batch_size = lstm_params.get('batch_size', 256)
        workers = min(8, num_cores) 
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences, num_workers=workers, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences, num_workers=workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences, num_workers=workers)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {device}")
        
        model = SepsisLSTM(
            input_dim=len(features), 
            hidden_dim=lstm_params.get('hidden_dim', 64), 
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2),
            num_heads=lstm_params.get('num_heads', 4),
            fc_dim=lstm_params.get('fc_dim', 32)
        ).to(device)
        
        # extract positive occurrence ratio to drastically bump minority backpropagation weight
        num_pos = df_train['target'].sum()
        num_neg = len(df_train) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lstm_params.get('learning_rate', 1e-3), 
            weight_decay=float(lstm_params.get('weight_decay', 1e-5))
        )
        
        epochs = lstm_params.get('epochs', 20)
        patience = lstm_params.get('patience', 4)
        best_val_auprc = 0.0
        patience_counter = 0
        
        # Log params
        mlflow.log_params(lstm_params)
        mlflow.log_param("model_type", "LSTM")
        
        model_save_path = os.path.join(artifact_dir, "best_lstm.pth")
        
        print("\nStarting LSTM Training...")
        for epoch in tqdm(range(epochs), desc="Epochs"):
            model.train()
            train_loss = 0.0
            
            for X_b, y_b, mask_b in train_loader:
                X_b, y_b, mask_b = X_b.to(device), y_b.to(device), mask_b.to(device)
                
                optimizer.zero_grad()
                logits = model(X_b)
                
                loss = criterion(logits, y_b)
                # mask padding
                loss = (loss * mask_b).sum() / mask_b.sum()
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(lstm_params.get('clip_grad_norm', 1.0)))
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_b, y_b, mask_b in val_loader:
                    X_b, y_b, mask_b = X_b.to(device), y_b.to(device), mask_b.to(device)
                    probs = torch.sigmoid(model(X_b))
                    
                    valid_idx = mask_b.bool()
                    val_preds.append(probs[valid_idx].cpu().numpy())
                    val_targets.append(y_b[valid_idx].cpu().numpy())
                    
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_auprc = average_precision_score(val_targets, val_preds)
            
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val AUPRC: {val_auprc:.4f}")
            mlflow.log_metric("val_auprc_epoch", val_auprc, step=epoch)
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
                    
        print("\nEvaluating Best LSTM on Test Set...")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        test_preds, test_targets = [], []
        with torch.no_grad():
            for X_b, y_b, mask_b in test_loader:
                X_b, y_b, mask_b = X_b.to(device), y_b.to(device), mask_b.to(device)
                probs = torch.sigmoid(model(X_b))
                
                valid_idx = mask_b.bool()
                test_preds.append(probs[valid_idx].cpu().numpy())
                test_targets.append(y_b[valid_idx].cpu().numpy())
                
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        
        metrics = evaluate_model(test_targets, test_preds, "LSTM Sequence Model")
        mlflow.log_metric("test_auprc", metrics['auprc'])
        mlflow.log_metric("test_auroc", metrics['auroc'])
        mlflow.log_metric("test_f1", metrics['f1'])
        mlflow.log_param("total_features_used", len(features))
        
        if config['system'].get('save_models', False):
            mlflow.log_artifact(model_save_path)
        
        prob_true, prob_pred = calibration_curve(test_targets, test_preds, n_bins=10)
        plt.figure(figsize=(6,6))
        plt.plot(prob_pred, prob_true, marker='o', label="LSTM")
        plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (LSTM Reliability)")
        plt.legend()
        calib_path = os.path.join(artifact_dir, "calibration_curve_lstm.png")
        plt.savefig(calib_path)
        plt.close()
        mlflow.log_artifact(calib_path)

if __name__ == "__main__":
    main()