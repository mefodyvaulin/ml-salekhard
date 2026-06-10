import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from sklearn.metrics import root_mean_squared_error
from src.models.rnn.predict import predict


def train_rnn_recursive(
    model: nn.Module,
    criterion,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    val_future_features: torch.Tensor = None,
    reg_type='none',
    lambda_l1=1e-3,
    scheduler=None,
    epochs: int = 100,
    max_epochs_no_improvement=10,
    device: str = 'cpu',
    verbose=False
):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_rmse = float('inf')
    epochs_no_improvement = 0
    
    history = {
        'train_loss': [],
        'val_rmse': [],
        'best_val_rmse': []
    }
    
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

            if reg_type.lower() == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += lambda_l1 * l1_norm

            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        history['train_loss'].append(train_loss)

        if val_loader is not None and val_future_features is not None:
            start_lags = X_val[0]
            
            y_pred = predict(model, start_lags, val_future_features, device=device)

            val_rmse = root_mean_squared_error(y_val_true, y_pred)
            history['val_rmse'].append(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            history['best_val_rmse'].append(best_val_rmse)

            if epochs_no_improvement >= max_epochs_no_improvement:
                print(f'Early stopping on {epoch} epoch')
                break
            
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_rmse if val_loader is not None else train_loss
                scheduler.step(metric)
            else:
                scheduler.step()

        if verbose:
            verbose_output = f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}'
            if val_loader is not None:
                verbose_output += f' | Val RMSE: {val_rmse:.4f} | Best val RMSE: {best_val_rmse}'
            print(verbose_output)
        
        best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)
    return model, history