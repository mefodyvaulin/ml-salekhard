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
    

def train_lstm_recursive_val(
    model: nn.Module,
    criterion,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    reg_type='none',
    lambda_l1=1e-3,
    target_scaler: StandardScaler = None,
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
        X_val, y_val_scaled_true = val_loader.dataset.tensors
        num_targets = y_val_scaled_true.shape[1]
        num_features = X_val.shape[2] - num_targets

        y_val_true = target_scaler.inverse_transform(y_val_scaled_true.numpy())

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

        if val_loader is not None:
            model.eval()
            current_lags = X_val[0].clone().to(device)
            y_pred_scaled = []
            
            with torch.no_grad():
                for i in range(len(X_val)):
                    lags_tensor = current_lags.unsqueeze(0)
                    pred = model(lags_tensor).cpu()
                    y_pred_scaled.append(pred.numpy()[0])
                    
                    if i < len(X_val) - 1:
                        next_step_features = X_val[i+1, -1, :num_features].to(device)
                        pred_tensor = pred[0].to(device)
                        
                        next_step_vector = torch.cat((next_step_features, pred_tensor))
                        current_lags = torch.vstack((current_lags[1:], next_step_vector))
                        
            y_pred_scaled = np.array(y_pred_scaled)
            y_pred = target_scaler.inverse_transform(y_pred_scaled)

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