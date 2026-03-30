import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

class MCDropoutNet(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=256, mc_samples=20):
        super().__init__()
        self.save_hyperparameters()
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate

        # CNN backbone
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout_rate), # Using standard nn.Dropout which works on any tensor shape
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )

        # RNN
        self.rnn = nn.LSTM(128, hidden_dim // 2, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=dropout_rate)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True, dropout=dropout_rate)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 600)
        )

        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.cls_weight = nn.Parameter(torch.tensor(1.0))
        self.reg_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
     
        x = x.transpose(1, 2)  # (B, 4, 600) -> (B, 600, 4)
        x = self.conv(x)       # (B, 128, 600)
        x = x.transpose(1, 2)  # (B, 600, 128)

        rnn_out, (h, _) = self.rnn(x)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        pooled = torch.mean(attn_out, dim=1)

        cls_out = self.cls_head(pooled)
        reg_out = self.reg_head(pooled)

        return cls_out, reg_out

    def mc_predict(self, x, n_samples=None):
        if n_samples is None:
            n_samples = self.mc_samples

        self.train()  
        cls_predictions = []
        reg_predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                cls_out, reg_out = self.forward(x)
                cls_predictions.append(F.softmax(cls_out, dim=-1))
                reg_predictions.append(reg_out)

        self.eval()  # Disable dropout layers for normal inference later

        cls_stack = torch.stack(cls_predictions)
        reg_stack = torch.stack(reg_predictions)

        cls_mean = torch.mean(cls_stack, dim=0)
        cls_std = torch.std(cls_stack, dim=0)
        cls_entropy = -torch.sum(cls_mean * torch.log(cls_mean + 1e-9), dim=-1)

        reg_mean = torch.mean(reg_stack, dim=0)
        reg_std = torch.std(reg_stack, dim=0)

        return {
            'cls_mean': cls_mean, 'cls_std': cls_std, 'cls_entropy': cls_entropy,
            'reg_mean': reg_mean, 'reg_std': reg_std
        }