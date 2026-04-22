import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SepsisSequenceDataset(Dataset):
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
    Xs, ys = zip(*batch)
    X_pad = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0.0)
    mask_pad = pad_sequence([torch.ones_like(y) for y in ys], batch_first=True, padding_value=0.0)
    return X_pad, y_pad, mask_pad
