import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


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


def train_lstm_model(
    model: nn.Module,
    criterion,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    reg_type='none',
    lambda_l1=1e-3,
    device: str = 'cpu',
    epochs=50,
    max_epochs_no_improvement=10,
    verbose=True
) -> nn.Module:
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improvement = 0
    
    val_hist = []
    
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
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)
            val_hist.append(val_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= max_epochs_no_improvement:
                    print(f"Early stopping on {epoch} epoch")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
                best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)
    return model, val_hist