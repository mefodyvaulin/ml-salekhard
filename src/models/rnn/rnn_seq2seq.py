import torch
import torch.nn as nn
import numpy as np


class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_size):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.target_size = target_size

    def forward(
        self,
        encoder_x,
        decoder_feature,
        decoder_targets=None,
        teacher_forcing_ratio=0.0
    ):
        horizon = decoder_feature.size(1)
        
        hidden = self.encoder(encoder_x)
        prev_target = encoder_x[:, -1, -self.target_size:]
        
        preds = []

        for step in range(horizon):
            pred, hidden = self.decoder(
                decoder_feature[:, step, :],
                prev_target,
                hidden
            )
            
            preds.append(pred.unsqueeze(1))
            
            use_teacher_forcing = (
                self.training
                and decoder_targets is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )

            prev_target = decoder_targets[:, step, :] if use_teacher_forcing else pred
        
        return torch.cat(preds, dim=1)

class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int, 
        num_layers: int,
        dropout=0.2, 
        rnn_type='gru'
    ):
        super().__init__()
        
        rnn = nn.GRU if rnn_type.lower() == 'gru' else nn.LSTM
        
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        _, hidden = self.rnn(x)
        return hidden
    
class Decoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        target_size: int,
        hidden_size: int,
        num_layers: int,
        dropout=0.2,
        rnn_type='gru'
    ):
        super().__init__()
        
        rnn = nn.GRU if rnn_type.lower() == 'gru' else nn.LSTM
        
        self.rnn = rnn(
            input_size=feature_size + target_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.out = nn.Linear(hidden_size, target_size)

    def forward(self, decoder_feature, prev_target, hidden):
        x = torch.cat([decoder_feature, prev_target], dim=1)
        x = x.unsqueeze(1)
        
        output, hidden = self.rnn(x, hidden)
        pred = self.out(output[:, -1, :])
        
        return pred, hidden
