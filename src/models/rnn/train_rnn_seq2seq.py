import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import copy
from sklearn.metrics import root_mean_squared_error


def train_rnn_seq2seq(
    model: nn.Module,
    criterion,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    teacher_forcing_ratio=0.5,
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
        'best_val_rmse': [],
        'best_epoch': None
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_enc_batch, X_dec_batch, y_batch in train_loader:
            X_enc_batch = X_enc_batch.to(device)
            X_dec_batch = X_dec_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            pred = model(
                encoder_x=X_enc_batch,
                decoder_feature=X_dec_batch,
                decoder_targets=y_batch,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            loss = criterion(pred, y_batch)

            if reg_type.lower() == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += lambda_l1 * l1_norm

            loss.backward()            
            optimizer.step()
            
            train_loss += loss.item() * X_enc_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        if val_loader is not None:
            val_rmse = validate_rnn_seq2seq(model, val_loader, device=device)
            history['val_rmse'].append(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improvement = 0
                history['best_epoch'] = epoch + 1
                best_model_weights = copy.deepcopy(model.state_dict())
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

    if val_loader is not None:
        model.load_state_dict(best_model_weights)

    return model, history

def validate_rnn_seq2seq(
    model: nn.Module,
    val_loader: DataLoader,
    device=None
):
    model.eval()
    
    preds = []
    targets = []
    
    with torch.no_grad():
        for X_enc_batch, X_dec_batch, y_batch in val_loader:
            X_enc_batch = X_enc_batch.to(device)
            X_dec_batch = X_dec_batch.to(device)
            
            pred = model(
                encoder_x=X_enc_batch,
                decoder_feature=X_dec_batch,
                decoder_targets=None,
                teacher_forcing_ratio=0.0
            )
            
            preds.append(pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    
    val_rmse = root_mean_squared_error(
        y_true.reshape(-1, y_true.shape[-1]),
        y_pred.reshape(-1, y_true.shape[-1])
    )
    
    return val_rmse