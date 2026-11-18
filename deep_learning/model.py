"""
EEGNet + LSTM.

EEGNet (Lawhern et al., 2018) is a compact CNN designed for EEG:
  - temporal convolution     -> learns frequency filters (like theta / beta bands)
  - depthwise spatial conv   -> learns which electrodes matter, per filter
  - separable convolution    -> combines them efficiently
We keep EEGNet's output as a SEQUENCE over time and feed it to an LSTM, so the model can also
learn how activity evolves across the 2-second window instead of only averaging it.
"""
import torch
import torch.nn as nn


class EEGNetLSTM(nn.Module):
    def __init__(self, n_channels=19, n_classes=2, F1=8, D=2, F2=16, kern_len=64, dropout=0.5, lstm_hidden=32):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),   # depthwise spatial filter
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # separable conv
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(F2, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):                 # x: (batch, channels, time)
        x = self.block2(self.block1(x.unsqueeze(1)))   # (batch, F2, 1, T')
        x = x.squeeze(2).transpose(1, 2)               # (batch, T', F2)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])
