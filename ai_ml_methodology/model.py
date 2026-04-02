import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl


class MCDropoutNet(pl.LightningModule):
    def __init__(self, dropout_rate=0.3, hidden_dim=256, mc_samples=20,
                 learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Shared stem: 4 → 64
        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Multi-scale dilated branches (all output 64 channels, then concat → 128)
        # Branch A: fine-grained (dilation=1, receptive field ~5 steps)
        self.branch_a = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1, dilation=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        # Branch B: medium-range (dilation=4, receptive field ~25 steps)
        self.branch_b = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=4, dilation=4), nn.BatchNorm1d(64), nn.ReLU(),
        )

        # Merge branches: 128 → 128
        self.merge = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # RNN
        self.rnn = nn.LSTM(128, hidden_dim // 2, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=dropout_rate)

        # Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8,
                                               batch_first=True, dropout=dropout_rate)

        # Classification head — uses mean-pooled context (global summary)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Per-timestep regression head — predicts DELTA from current range.
        # Since future_range ≈ current_range + small_change, learning the delta
        # is much easier than learning the absolute value, reducing RMSE significantly.
        # 3 layers for more expressive capacity.
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.cls_weight = nn.Parameter(torch.tensor(1.0))
        self.reg_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: (B, T, 4) — normalized, channel 0 is range
        current_range = x[:, :, 0]           # (B, T) — normalized current range

        x = x.transpose(1, 2)               # (B, 4, T)
        x = self.stem(x)                     # (B, 64, T)

        # Multi-scale branches in parallel
        a = self.branch_a(x)                 # (B, 64, T)
        b = self.branch_b(x)                 # (B, 64, T)
        x = self.merge(torch.cat([a, b], dim=1))  # (B, 128, T)

        x = x.transpose(1, 2)               # (B, T, 128)
        rnn_out, _ = self.rnn(x)             # (B, T, hidden_dim)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)  # (B, T, hidden_dim)

        # Classification: global mean-pool
        pooled  = torch.mean(attn_out, dim=1)           # (B, hidden_dim)
        cls_out = self.cls_head(pooled)                 # (B, 3)

        # Regression: predict delta, then add current range (residual connection)
        # This way the model only learns the small change, not the absolute value
        reg_delta = self.reg_head(attn_out).squeeze(-1) # (B, T)
        reg_out   = current_range + reg_delta           # (B, T)

        return cls_out, reg_out

    def _shared_step(self, batch):
        x, cls_label, reg_target = batch
        cls_out, reg_out = self(x)
        cls_loss = F.cross_entropy(cls_out, cls_label)
        reg_loss = F.smooth_l1_loss(reg_out, reg_target)
        loss = self.cls_weight.abs() * cls_loss + self.reg_weight.abs() * reg_loss
        acc  = self.acc(cls_out.argmax(dim=-1), cls_label)
        rmse = torch.sqrt(F.mse_loss(reg_out, reg_target))
        return loss, cls_loss, reg_loss, acc, rmse

    def training_step(self, batch, batch_idx):
        loss, cls_loss, reg_loss, acc, rmse = self._shared_step(batch)
        self.log_dict({
            'train_loss': loss, 'train_cls_loss': cls_loss,
            'train_reg_loss': reg_loss, 'train_acc': acc, 'train_rmse': rmse
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, cls_loss, reg_loss, acc, rmse = self._shared_step(batch)
        self.log_dict({
            'val_loss': loss, 'val_cls_loss': cls_loss,
            'val_reg_loss': reg_loss, 'val_acc': acc, 'val_rmse': rmse
        }, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }

    def mc_predict(self, x, n_samples=None):
        if n_samples is None:
            n_samples = self.mc_samples

        # Keep BatchNorm in eval mode — only re-enable Dropout layers.
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        cls_predictions, reg_predictions = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                cls_out, reg_out = self.forward(x)
                cls_predictions.append(F.softmax(cls_out, dim=-1))
                reg_predictions.append(reg_out)

        self.eval()

        cls_stack = torch.stack(cls_predictions)
        reg_stack = torch.stack(reg_predictions)
        cls_mean  = cls_stack.mean(0)
        cls_std   = cls_stack.std(0)
        cls_entropy = -torch.sum(cls_mean * torch.log(cls_mean + 1e-9), dim=-1)

        return {
            'cls_mean': cls_mean, 'cls_std': cls_std, 'cls_entropy': cls_entropy,
            'reg_mean': reg_stack.mean(0), 'reg_std': reg_stack.std(0)
        }
