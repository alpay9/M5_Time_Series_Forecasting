"""
Training and Evaluation Script - FIXED VERSION
===============================================

Fixes applied:
- Improved early stopping with min_delta
- Better learning rate scheduling
- Gradient monitoring
- Mixed precision support built-in (FP16/BF16)
- Better logging
- Diagnostic outputs

Author: CS 415 Deep Learning Project Team
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import math
from sklearn.metrics import mean_squared_error
import os
import sys
import logging


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""
    
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DualLogger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_file='training.log'):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
        # Also setup Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    FIXED: Added min_delta for better stopping behavior.
    """
    
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement required to reset patience
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch=0):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            # No meaningful improvement
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience} (best: {self.best_loss:.4f} at epoch {self.best_epoch + 1})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Meaningful improvement
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


class Trainer:
    """
    Handles model training, validation, and testing.
    FIXED: Improved LR scheduling, gradient monitoring, mixed precision support.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        lr=0.001,
        device='cuda',
        model_name='model',
        precision='fp16',
        grad_clip_max_norm=5.0,
        grad_clip_enabled=True,
        series_scale_factors=None,
        series_ids=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        normalized_precision = precision.lower()
        if normalized_precision not in {'fp32', 'fp16', 'bf16'}:
            raise ValueError(f"Unsupported precision '{precision}'. Use 'fp32', 'fp16', or 'bf16'.")
        self.precision = normalized_precision
        self.use_amp = device == 'cuda' and self.precision in {'fp16', 'bf16'}
        self.autocast_dtype = None
        if self.precision == 'fp16':
            self.autocast_dtype = torch.float16
        elif self.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        self.series_scale_factors = series_scale_factors
        self.series_ids = series_ids
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Loss function - Huber is more robust than MSE
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Warmup + cosine decay scheduler (initialized in train once total steps are known)
        self.scheduler = None
        self.warmup_ratio = 0.1
        self.scheduler_step_per_batch = True
        
        # Mixed precision scaler
        self.scaler = None
        if self.precision == 'fp16' and device == 'cuda':
            self.scaler = GradScaler()

        self.grad_clip_max_norm = grad_clip_max_norm
        self.grad_clip_enabled = grad_clip_enabled
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'lr': [],
            'grad_norm': []
        }

    def _inverse_transform_targets(self, values: np.ndarray) -> np.ndarray:
        if self.series_scale_factors is None or self.series_ids is None:
            return np.expm1(values)
        if len(values) != len(self.series_ids):
            raise ValueError(
                "series_ids length must match number of samples for inverse scaling."
            )
        scale_factors = np.array(
            [self.series_scale_factors.get(series_id, 1.0) for series_id in self.series_ids],
            dtype=np.float32
        )
        return np.expm1(values) * scale_factors[:, None]
        
    def _compute_grad_norm(self):
        """Compute total gradient norm for monitoring."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _autocast_context(self):
        if self.use_amp:
            return autocast(dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def _clip_gradients(self):
        if not self.grad_clip_enabled or self.grad_clip_max_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.grad_clip_max_norm
        )

    def _build_warmup_cosine_scheduler(self, total_steps):
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _step_scheduler(self):
        if self.scheduler is None:
            return
        self.scheduler.step()
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
    def train_epoch(self):
        """Train for one epoch with mixed precision support."""
        self.model.train()
        total_loss = 0
        sse = 0.0
        count = 0
        grad_norms = []
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False, mininterval=5)
        postfix_interval = 100
        for batch_idx, (X_batch, Y_batch) in enumerate(pbar):
            X_batch = X_batch.to(self.device, non_blocking=True)
            Y_batch = Y_batch.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with self._autocast_context():
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, Y_batch)
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    self._clip_gradients()
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler_step_per_batch:
                        self._step_scheduler()
                else:
                    loss.backward()
                    self._clip_gradients()
                    self.optimizer.step()
                    if self.scheduler_step_per_batch:
                        self._step_scheduler()
            else:
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, Y_batch)
                
                loss.backward()
                self._clip_gradients()
                self.optimizer.step()
                if self.scheduler_step_per_batch:
                    self._step_scheduler()
            
            total_loss += loss.item()
            
            # Monitor gradients periodically
            if batch_idx % 500 == 0:
                grad_norm = self._compute_grad_norm()
                grad_norms.append(grad_norm)
            
            diff = (predictions.detach().cpu() - Y_batch.detach().cpu()).double()
            if not torch.isfinite(diff).all():
                print(f"  ⚠️ WARNING: Non-finite RMSE diff at batch {batch_idx}; skipping accumulation.")
            else:
                sse += torch.sum(diff ** 2).item()
                count += diff.numel()
            
            if batch_idx % postfix_interval == 0:
                pbar.set_postfix({'loss': loss.item()})
            
            # DIAGNOSTIC: Check for constant predictions (first batch of first epoch)
            if batch_idx == 0:
                pred_min = predictions.min().item()
                pred_max = predictions.max().item()
                pred_std = predictions.std().item()
                if pred_std < 0.01:
                    print(f"\n  ⚠️ WARNING: Predictions have very low variance (std={pred_std:.6f})")
                    print(f"     Pred range: [{pred_min:.4f}, {pred_max:.4f}]")
                    print(f"     Target range: [{Y_batch.min().item():.4f}, {Y_batch.max().item():.4f}]")
        
        avg_loss = total_loss / len(self.train_loader)
        rmse = np.sqrt(sse / count) if count else 0
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        
        return avg_loss, rmse, avg_grad_norm
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        sse = 0.0
        count = 0
        
        with torch.no_grad():
            for batch_idx, (X_batch, Y_batch) in enumerate(self.val_loader):
                X_batch = X_batch.to(self.device, non_blocking=True)
                Y_batch = Y_batch.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with self._autocast_context():
                        predictions = self.model(X_batch)
                        loss = self.criterion(predictions, Y_batch)
                else:
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, Y_batch)
                
                total_loss += loss.item()
                
                diff = (predictions.detach().cpu() - Y_batch.detach().cpu()).double()
                if not torch.isfinite(diff).all():
                    print(f"  ⚠️ WARNING: Non-finite RMSE diff at val batch {batch_idx}; skipping accumulation.")
                else:
                    sse += torch.sum(diff ** 2).item()
                    count += diff.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        rmse = np.sqrt(sse / count) if count else 0
        
        return avg_loss, rmse, None, None
    
    def test(self):
        """Test the model and return predictions."""
        if self.test_loader is None:
            return None
        
        self.model.eval()
        predictions_all = []
        targets_all = []
        
        with torch.no_grad():
            for X_batch, Y_batch in self.test_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                Y_batch = Y_batch.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with self._autocast_context():
                        predictions = self.model(X_batch)
                else:
                    predictions = self.model(X_batch)
                
                predictions_all.append(predictions.float().cpu().numpy())
                targets_all.append(Y_batch.float().cpu().numpy())
        
        predictions_all = np.concatenate(predictions_all)
        targets_all = np.concatenate(targets_all)
        predictions_unscaled = self._inverse_transform_targets(predictions_all)
        targets_unscaled = self._inverse_transform_targets(targets_all)
        
        rmse = np.sqrt(mean_squared_error(targets_unscaled, predictions_unscaled))
        
        return {
            'rmse': rmse,
            'predictions': predictions_all,
            'targets': targets_all,
            'predictions_unscaled': predictions_unscaled,
            'targets_unscaled': targets_unscaled
        }
    
    def train(self, epochs=50, early_stopping_patience=10):
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'='*80}")
        print(f"  Precision: {self.precision}")
        print(f"  Initial LR: {self.optimizer.param_groups[0]['lr']}")
        
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            min_delta=1e-4,
            verbose=True
        )
        best_val_loss = float('inf')
        checkpoint_path = f'best_{self.model_name}.pt'
        checkpoint_saved = False
        start_time = time.time()

        steps_per_epoch = len(self.train_loader)
        total_steps = max(1, steps_per_epoch * epochs)
        self.scheduler = self._build_warmup_cosine_scheduler(total_steps)
        self.history['lr'] = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_rmse, avg_grad_norm = self.train_epoch()
            
            # Validate
            val_loss, val_rmse, _, _ = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['lr'].append(current_lr)
            self.history['grad_norm'].append(avg_grad_norm)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f} | Train RMSE (log-space): {train_rmse:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val RMSE (log-space):   {val_rmse:.4f}")
            print(f"  LR: {current_lr:.6f} | Grad Norm: {avg_grad_norm:.4f}")
            
            # Check for learning issues
            if avg_grad_norm < 1e-6:
                print(f"  ⚠️ WARNING: Very small gradients - model may not be learning!")
            elif avg_grad_norm > 100:
                print(f"  ⚠️ WARNING: Large gradients - consider reducing learning rate!")
            
            # Save best model
            if not np.isfinite(val_loss):
                print("  ⚠️ WARNING: Validation loss is NaN/inf; skipping checkpoint save.")
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                checkpoint_saved = True
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                print(f"\n  Early stopping triggered at epoch {epoch + 1}")
                print(f"  Best epoch was {early_stopping.best_epoch + 1} with val_loss: {early_stopping.best_loss:.4f}")
                break
        
        training_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"{'='*80}")
        
        # Load best model if available; otherwise fall back to latest weights.
        if checkpoint_saved:
            self.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print("  ⚠️ WARNING: No best checkpoint saved; using latest model weights.")
            torch.save(self.model.state_dict(), checkpoint_path)
        
        return self.history, training_time
    
    def plot_history(self, save_path=None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'{self.model_name} - Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE
        axes[0, 1].plot(self.history['train_rmse'], label='Train RMSE (log-space)')
        axes[0, 1].plot(self.history['val_rmse'], label='Val RMSE (log-space)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE (log-space)')
        axes[0, 1].set_title(f'{self.model_name} - RMSE (log-space)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title(f'{self.model_name} - Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norm
        axes[1, 1].plot(self.history['grad_norm'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title(f'{self.model_name} - Gradient Norm')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved plot: {save_path}")
        
        plt.close()


def compare_models(results_dict, save_path='model_comparison.png'):
    """Compare multiple models side by side."""
    models = list(results_dict.keys())
    rmse_values = [results_dict[m]['rmse'] for m in models]
    wrmsse_values = [results_dict[m].get('wrmsse', 0) for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sort by RMSE for better visualization
    sorted_indices = np.argsort(rmse_values)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_rmse = [rmse_values[i] for i in sorted_indices]
    sorted_wrmsse = [wrmsse_values[i] for i in sorted_indices]
    
    # RMSE comparison
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_models)))
    bars1 = axes[0].bar(sorted_models, sorted_rmse, color=colors)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Root Mean Squared Error Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, v in zip(bars1, sorted_rmse):
        axes[0].text(bar.get_x() + bar.get_width()/2, v, f'{v:.4f}', 
                    ha='center', va='bottom', fontsize=9)
    
    # WRMSSE comparison
    bars2 = axes[1].bar(sorted_models, sorted_wrmsse, color=colors)
    axes[1].set_ylabel('WRMSSE')
    axes[1].set_title('Weighted Root Mean Squared Scaled Error')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, v in zip(bars2, sorted_wrmsse):
        if v > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, v, f'{v:.4f}', 
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison: {save_path}")
    plt.close()


def save_results_json(results_dict, training_times, save_path='final_results.json'):
    """Save results to JSON."""
    output = {}
    
    for model_name in results_dict.keys():
        output[model_name] = {
            'rmse': float(results_dict[model_name]['rmse']),
            'wrmsse': float(results_dict[model_name].get('wrmsse', 0)),
            'training_time_seconds': float(training_times.get(model_name, 0)),
            'training_time_minutes': float(training_times.get(model_name, 0)) / 60
        }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved results: {save_path}")


def create_results_table(results_dict):
    """Create formatted results table."""
    print("\n" + "="*80)
    print("FINAL RESULTS - MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'RMSE':>10} {'WRMSSE':>10} {'Rank':>8}")
    print("-"*80)
    
    # Sort by RMSE
    sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['rmse'])
    
    for rank, (name, results) in enumerate(sorted_models, 1):
        rmse = results['rmse']
        wrmsse = results.get('wrmsse', 0)
        wrmsse_str = f"{wrmsse:.4f}" if wrmsse > 0 else "N/A"
        
        print(f"{name:<15} {rmse:>10.4f} {wrmsse_str:>10} {rank:>8}")
    
    print("="*80)
    
    # Best model summary
    best_model = sorted_models[0][0]
    best_rmse = sorted_models[0][1]['rmse']
    print(f"\n🏆 Best Model: {best_model} (RMSE: {best_rmse:.4f})")


def diagnose_data(X_train, Y_train, X_val=None, Y_val=None):
    """
    Diagnostic function to check data quality before training.
    Run this before training to catch potential issues.
    """
    print("\n" + "="*80)
    print("DATA DIAGNOSTICS")
    print("="*80)
    
    print("\n📊 Training Data Statistics:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_train - mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"  X_train - min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    print(f"  X_train has NaN: {np.isnan(X_train).any()}")
    print(f"  X_train has Inf: {np.isinf(X_train).any()}")
    
    print(f"\n  Y_train shape: {Y_train.shape}")
    print(f"  Y_train - mean: {Y_train.mean():.4f}, std: {Y_train.std():.4f}")
    print(f"  Y_train - min: {Y_train.min():.4f}, max: {Y_train.max():.4f}")
    print(f"  Y_train has NaN: {np.isnan(Y_train).any()}")
    print(f"  Y_train has Inf: {np.isinf(Y_train).any()}")
    
    # Check for potential issues
    issues = []
    
    if X_train.std() < 0.1:
        issues.append("⚠️ X_train has very low variance - features may not be informative")
    if X_train.std() > 100:
        issues.append("⚠️ X_train has very high variance - consider additional normalization")
    
    if Y_train.mean() > 10 and Y_train.std() > 10:
        issues.append("⚠️ Y_train is not normalized - consider log transform or scaling")
    
    if np.isnan(X_train).any() or np.isnan(Y_train).any():
        issues.append("❌ Data contains NaN values - must fix before training!")
    
    if np.isinf(X_train).any() or np.isinf(Y_train).any():
        issues.append("❌ Data contains Inf values - must fix before training!")
    
    # Y distribution analysis
    y_zeros = (Y_train == 0).sum() / Y_train.size * 100
    print(f"\n  Y_train zero percentage: {y_zeros:.1f}%")
    if y_zeros > 50:
        issues.append(f"⚠️ {y_zeros:.1f}% of targets are zero - sparse data, consider special handling")
    
    if issues:
        print("\n🚨 POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No obvious data issues detected")
    
    if X_val is not None:
        print(f"\n📊 Validation Data Statistics:")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  Y_val shape: {Y_val.shape}")
    
    print("="*80)
    
    return len(issues) == 0  # Return True if no issues
