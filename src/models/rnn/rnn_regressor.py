import torch
import torch.nn as nn


class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, rnn_type='gru'):
        super().__init__()
        
        rnn = nn.GRU if rnn_type.lower() == 'gru' else nn.LSTM
        
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        
        last_time_step_out = out[:, -1, :]
        
        return self.out(last_time_step_out)