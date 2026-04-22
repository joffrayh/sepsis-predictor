import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from .base_model import BaseSepsisModel
from ..data.sequence_utils import SepsisSequenceDataset, collate_sequences

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
        seq_len = lstm_out.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=causal_mask)
        lstm_out = self.layer_norm(lstm_out + attn_out)
        logits = self.classifier(lstm_out).squeeze(-1)
        return logits

class LSTMModelWrapper(BaseSepsisModel):
    def __init__(self, config, model_params, features):
        super().__init__(config, model_params, features)
        self.lstm_params = model_params.get('params', {})
        self.scaler = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_and_train(self, df_train, df_val):
        
        print("Scaling features using StandardScaler...")
        self.scaler = StandardScaler()
        df_train[self.features] = self.scaler.fit_transform(df_train[self.features])
        df_val[self.features] = self.scaler.transform(df_val[self.features])
        
        train_ds = SepsisSequenceDataset(df_train, self.features)
        val_ds = SepsisSequenceDataset(df_val, self.features)
        
        batch_size = self.lstm_params.get('batch_size', 256)
        avail_cores = os.cpu_count() or 1
        num_cores = self.config['system'].get('n_jobs', avail_cores)
        num_cores = avail_cores if num_cores < 0 else num_cores
        workers = min(8, num_cores)
        
        torch.set_num_threads(num_cores)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences, num_workers=workers, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences, num_workers=workers)
        
        self.model = SepsisLSTM(
            input_dim=len(self.features), 
            hidden_dim=self.lstm_params.get('hidden_dim', 64), 
            num_layers=self.lstm_params.get('num_layers', 2),
            dropout=self.lstm_params.get('dropout', 0.2),
            num_heads=self.lstm_params.get('num_heads', 4),
            fc_dim=self.lstm_params.get('fc_dim', 32)
        ).to(self.device)
        
        num_pos = df_train['target'].sum()
        num_neg = len(df_train) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lstm_params.get('learning_rate', 1e-3), 
            weight_decay=float(self.lstm_params.get('weight_decay', 1e-5))
        )
        
        epochs = self.lstm_params.get('epochs', 20)
        patience = self.lstm_params.get('patience', 4)
        best_val_auprc = 0.0
        patience_counter = 0
        
        mlflow.log_params(self.lstm_params)
        
        temp_model_path = "temp_lstm.pth"
        
        print("\nStarting LSTM Training...")
        for epoch in tqdm(range(epochs), desc="Epochs"):
            self.model.train()
            train_loss = 0.0
            
            for X_b, y_b, mask_b in train_loader:
                X_b, y_b, mask_b = X_b.to(self.device), y_b.to(self.device), mask_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                loss = (loss * mask_b).sum() / mask_b.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.lstm_params.get('clip_grad_norm', 1.0)))
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_b, y_b, mask_b in val_loader:
                    X_b, y_b, mask_b = X_b.to(self.device), y_b.to(self.device), mask_b.to(self.device)
                    probs = torch.sigmoid(self.model(X_b))
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
                torch.save(self.model.state_dict(), temp_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        
        self.model.load_state_dict(torch.load(temp_model_path))
        os.remove(temp_model_path)

    def predict_proba(self, df_test):
        df_test_scaled = df_test.copy()
        df_test_scaled[self.features] = self.scaler.transform(df_test[self.features])
        
        test_ds = SepsisSequenceDataset(df_test_scaled, self.features)
        test_loader = DataLoader(test_ds, batch_size=self.lstm_params.get('batch_size', 256), shuffle=False, collate_fn=collate_sequences, num_workers=min(8, self.config['system'].get('n_jobs', 1)))
        
        self.model.eval()
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for X_b, y_b, mask_b in test_loader:
                X_b, mask_b = X_b.to(self.device), mask_b.to(self.device)
                probs = torch.sigmoid(self.model(X_b))
                valid_idx = mask_b.bool()
                test_preds.append(probs[valid_idx].cpu().numpy())
                test_targets.append(y_b[valid_idx].numpy())
                
        return np.concatenate(test_preds), np.concatenate(test_targets)
        
    def save_model(self, model_name):
        import mlflow.pytorch
        mlflow.pytorch.log_model(self.model, name=model_name)

