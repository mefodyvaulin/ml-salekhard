import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
from sklearn.metrics import root_mean_squared_error


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout = 0.2 if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        
        last_time_step_out = out[:, -1, :]
        
        return self.out(last_time_step_out)


class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features,
        num_depths,
        kernel_depth=3,
        kernel_time=1,
        out_channels=16,
        lstm_hidden_size=16,
        lstm_num_layers=1
    ):
        super().__init__()
        self.num_features = num_features
        self.num_depths = num_depths
    
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(kernel_depth, kernel_time),
            padding=(kernel_depth // 2, kernel_time // 2)
        )

        self.relu = nn.ReLU()
        
        lstm_input_size = (num_depths * out_channels) + num_features
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout = 0.2 if lstm_num_layers > 1 else 0
        )
        self.out = nn.Linear(lstm_hidden_size, num_depths)
    
    def forward(self, x):
        x_other = x[:, :, :self.num_features]
        x_depths = x[:, :, self.num_features:]
        
        c = x_depths.permute(0, 2, 1).unsqueeze(1)

        c = self.conv2d(c)
        c = self.relu(c)

        c = c.permute(0, 3, 1, 2)
        batch_size, seq_len, _, _ = c.shape
        c = c.reshape(batch_size, seq_len, -1)

        lstm_input = torch.cat([x_other, c], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)
        last_out = lstm_out[:, -1, :]
        
        return self.out(last_out)


def predict(model: nn.Module, start_lags, future_features, device='cpu'):
    model.eval()
    current_lags = start_lags.clone().to(device)
    future_features = future_features.to(device)
    y_pred = []
    
    with torch.no_grad():
        for i in range(len(future_features)):     
            lags_tensor = current_lags.unsqueeze(0)
            pred = model(lags_tensor)
            y_pred.append(pred.cpu().numpy()[0])
            
            next_step_features = future_features[i]
            next_step_vector = torch.cat((next_step_features, pred[0]))
            current_lags = torch.vstack((current_lags[1:], next_step_vector))

    return np.array(y_pred)


def train_lstm_recursive_val(
    model: nn.Module,
    criterion,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    val_future_features: torch.Tensor = None,
    reg_type='none',
    lambda_l1=1e-3,
    epochs: int = 100,
    device: str = 'cpu',
    max_epochs_no_improvement=10,
    verbose=False
):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_rmse = float('inf')
    epochs_no_improvement = 0
    
    val_rmse_hist = []
    
    if val_loader is not None:
        X_val, y_val_true = val_loader.dataset.tensors

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            
            loss = criterion(predictions, y_batch)

            if reg_type == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += lambda_l1 * l1_norm

            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)

        if val_loader is not None and val_future_features is not None:
            start_lags = X_val[0]
            
            y_pred = predict(model, start_lags, val_future_features, device=device)

            val_rmse = root_mean_squared_error(y_val_true, y_pred)
            val_rmse_hist.append(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improvement = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improvement += 1

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")

            if epochs_no_improvement >= max_epochs_no_improvement:
                print(f"Early stopping on {epoch} epoch")
                break
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
                best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)
    return model, val_rmse_hist