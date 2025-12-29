"""
M5 Forecasting - All Model Architectures (FIXED VERSION)
=========================================================

Models included:
- RNN (baseline) - IMPROVED
- GRU (baseline) - IMPROVED  
- LSTM (baseline) - IMPROVED
- TCN (Temporal Convolutional Network)
- Informer (ProbSparse attention) - FIXED
- Autoformer (Auto-correlation + decomposition) - FIXED FFT
- FEDformer (Frequency domain) - FIXED FFT

Fixes applied:
- Added input projection and LayerNorm to RNN/GRU/LSTM
- Increased hidden sizes
- Multi-layer output heads
- Fixed FFT operations for FP16 compatibility
- Fixed in-place operations

Author: CS 415 Deep Learning Project Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================================
# BASELINE RNN MODELS - IMPROVED
# ============================================================================

class SimpleRNN(nn.Module):
    """Improved RNN for time series forecasting."""
    
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        output_len=28
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection with normalization
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-layer output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_len)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predictions: (batch, output_len)
        """
        # Project and normalize input
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # RNN forward
        out, hidden = self.rnn(x)
        
        # Use last timestep
        last_output = out[:, -1, :]
        
        # Multi-layer projection
        predictions = self.output_head(last_output)
        
        return predictions


class GRU(nn.Module):
    """Improved GRU for time series forecasting."""
    
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_len=28
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection with normalization
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer norm after GRU
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # Multi-layer output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_len)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predictions: (batch, output_len)
        """
        # Project and normalize input
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # GRU forward
        out, hidden = self.gru(x)
        
        # Use last timestep with normalization
        last_output = out[:, -1, :]
        last_output = self.output_norm(last_output)
        
        # Multi-layer projection
        predictions = self.output_head(last_output)
        
        return predictions


class LSTM(nn.Module):
    """Improved LSTM for time series forecasting."""
    
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_len=28
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection with normalization
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer norm after LSTM
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # Multi-layer output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_len)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predictions: (batch, output_len)
        """
        # Project and normalize input
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # LSTM forward
        out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep with normalization
        last_output = out[:, -1, :]
        last_output = self.output_norm(last_output)
        
        # Multi-layer projection
        predictions = self.output_head(last_output)
        
        return predictions


# ============================================================================
# SHARED COMPONENTS FOR TRANSFORMERS
# ============================================================================

class SeriesDecomposition(nn.Module):
    """Series decomposition block."""
    
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            seasonal, trend components
        """
        x_permuted = x.permute(0, 2, 1)
        trend = self.avg_pool(x_permuted)
        
        if trend.shape[-1] != x_permuted.shape[-1]:
            trend = F.pad(trend, (0, x_permuted.shape[-1] - trend.shape[-1]))
        
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        
        return seasonal, trend


class PositionalEmbedding(nn.Module):
    """Positional embedding for transformers."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1), :]


# ============================================================================
# INFORMER - FIXED
# ============================================================================

class ProbSparseSelfAttention(nn.Module):
    """ProbSparse Self-Attention from Informer - FIXED."""
    
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.query_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        u = int(self.factor * np.log(seq_len + 1))
        u = min(u, seq_len)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        max_scores = scores.max(dim=-1)[0]
        mean_scores = scores.mean(dim=-1)
        sparsity = max_scores - mean_scores
        
        _, top_indices = torch.topk(sparsity, u, dim=-1)
        
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        Q_selected = torch.gather(Q, 2, top_indices_expanded)
        
        scores_selected = torch.matmul(Q_selected, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores_selected, dim=-1)
        attn = self.dropout(attn)
        
        out_selected = torch.matmul(attn, V)
        
        # FIXED: Clone before scatter to avoid in-place operation issues
        out_full = V.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1).clone()
        out_full.scatter_(2, top_indices_expanded, out_selected)
        
        out = out_full.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out


class InformerEncoderLayer(nn.Module):
    """Informer encoder layer."""
    
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1, factor=5):
        super().__init__()
        
        self.attention = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Changed from ReLU to GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Pre-norm architecture (more stable)
        new_x = self.attention(self.norm1(x))
        x = x + self.dropout1(new_x)
        
        new_x = self.ff(self.norm2(x))
        x = x + new_x
        
        return x


class Informer(nn.Module):
    """Informer model for time series forecasting."""
    
    def __init__(
        self,
        input_dim,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=1024,
        dropout=0.1,
        factor=5,
        output_len=28
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, output_len)
        
    def forward(self, x):
        # Input embedding
        x = self.input_projection(x)
        x = x + self.position_embedding(x)
        x = self.dropout(x)
        
        # Encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        x = self.norm(x)
        
        # Use last timestep for prediction
        x = x[:, -1, :]
        output = self.projection(x)
        
        return output


# ============================================================================
# AUTOFORMER - FIXED FFT
# ============================================================================

class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism from Autoformer - FIXED for FP16."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.query_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # FIXED: Disable autocast for FFT operations (FP16 compatibility)
        with torch.amp.autocast('cuda', enabled=False):
            Q_float = Q.float()
            K_float = K.float()
            
            Q_fft = torch.fft.rfft(Q_float, dim=2)
            K_fft = torch.fft.rfft(K_float, dim=2)
            
            corr_fft = Q_fft * torch.conj(K_fft)
            corr = torch.fft.irfft(corr_fft, n=seq_len, dim=2)
        
        # Convert back to original dtype
        corr = corr.to(Q.dtype)
        corr = corr / seq_len
        
        # Find top-k correlations
        top_k = max(1, int(seq_len * 0.25))
        corr_mean = corr.mean(dim=-1)
        _, top_indices = torch.topk(corr_mean, top_k, dim=-1)
        
        weights = torch.softmax(torch.gather(corr_mean, -1, top_indices), dim=-1)
        
        # Aggregate values based on correlations
        out = torch.zeros_like(V)
        for i in range(top_k):
            delay = top_indices[:, :, i].unsqueeze(-1).unsqueeze(-1)
            w = weights[:, :, i:i+1].unsqueeze(-1)
            
            # Simple weighted aggregation (simplified from paper)
            out = out + w * V
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class AutoformerEncoderLayer(nn.Module):
    """Autoformer encoder layer."""
    
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.attention = AutoCorrelation(d_model, n_heads, dropout)
        self.decomp1 = SeriesDecomposition(kernel_size=25)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.decomp2 = SeriesDecomposition(kernel_size=25)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        new_x = self.attention(x)
        x = x + self.dropout1(new_x)
        x, _ = self.decomp1(x)
        
        new_x = self.ff(x)
        x = x + self.dropout2(new_x)
        x, _ = self.decomp2(x)
        
        return x


class Autoformer(nn.Module):
    """Autoformer model for time series forecasting."""
    
    def __init__(
        self,
        input_dim,
        d_model=256,
        n_heads=8,
        e_layers=2,
        d_ff=1024,
        dropout=0.1,
        output_len=28
    ):
        super().__init__()
        
        self.decomp = SeriesDecomposition(kernel_size=25)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, output_len)
        self.trend_projection = nn.Linear(input_dim, output_len)
        
    def forward(self, x):
        # Decompose input
        seasonal, trend = self.decomp(x)
        
        # Encode seasonal component
        enc_out = self.input_projection(seasonal)
        enc_out = enc_out + self.position_embedding(enc_out)
        enc_out = self.dropout(enc_out)
        
        for encoder_layer in self.encoder_layers:
            enc_out = encoder_layer(enc_out)
        
        enc_out = self.norm(enc_out)
        
        # Use last timestep
        enc_out = enc_out[:, -1, :]
        seasonal_output = self.projection(enc_out)
        
        # Project trend
        trend_output = self.trend_projection(trend[:, -1, :])
        
        # Combine
        output = seasonal_output + trend_output
        
        return output


# ============================================================================
# FEDFORMER - FIXED FFT
# ============================================================================

class FourierBlock(nn.Module):
    """Fourier block for FEDformer - FIXED for FP16."""
    
    def __init__(self, d_model, modes=32):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        
        # Complex weights for frequency domain
        self.weights = nn.Parameter(
            torch.randn(modes, d_model, d_model, dtype=torch.cfloat) * 0.02
        )
        
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # FIXED: Disable autocast for FFT operations (FP16 compatibility)
        with torch.amp.autocast('cuda', enabled=False):
            x_float = x.float()
            x_fft = torch.fft.rfft(x_float, dim=1)
            freq_len = x_fft.shape[1]
            
            indices = list(range(min(self.modes, freq_len)))
            
            out_fft = torch.zeros_like(x_fft)
            
            for i, idx in enumerate(indices):
                if i < len(self.weights):
                    out_fft[:, idx, :] = torch.einsum(
                        'bd,de->be',
                        x_fft[:, idx, :],
                        self.weights[i]
                    )
            
            out = torch.fft.irfft(out_fft, n=seq_len, dim=1)
        
        return out.to(x.dtype)


class FEDformerEncoderLayer(nn.Module):
    """FEDformer encoder layer."""
    
    def __init__(self, d_model, modes=32, dropout=0.1, d_ff=2048):
        super().__init__()
        
        self.attention = FourierBlock(d_model, modes)
        self.decomp1 = SeriesDecomposition(kernel_size=25)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.decomp2 = SeriesDecomposition(kernel_size=25)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        new_x = self.attention(x)
        x = x + self.dropout1(new_x)
        x, _ = self.decomp1(x)
        
        new_x = self.ff(x)
        x = x + self.dropout2(new_x)
        x, _ = self.decomp2(x)
        
        return x


class FEDformer(nn.Module):
    """FEDformer model for time series forecasting."""
    
    def __init__(
        self,
        input_dim,
        d_model=256,
        modes=32,
        e_layers=2,
        d_ff=1024,
        dropout=0.1,
        output_len=28
    ):
        super().__init__()
        
        self.decomp = SeriesDecomposition(kernel_size=25)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(d_model, modes, dropout, d_ff)
            for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, output_len)
        self.trend_projection = nn.Linear(input_dim, output_len)
        
    def forward(self, x):
        # Decompose input
        seasonal, trend = self.decomp(x)
        
        # Encode seasonal
        enc_out = self.input_projection(seasonal)
        enc_out = enc_out + self.position_embedding(enc_out)
        enc_out = self.dropout(enc_out)
        
        for encoder_layer in self.encoder_layers:
            enc_out = encoder_layer(enc_out)
        
        enc_out = self.norm(enc_out)
        
        # Use last timestep
        enc_out = enc_out[:, -1, :]
        seasonal_output = self.projection(enc_out)
        
        # Project trend
        trend_output = self.trend_projection(trend[:, -1, :])
        
        # Combine
        output = seasonal_output + trend_output
        
        return output


# ============================================================================
# TCN - IMPROVED
# ============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    """Temporal block with dilated convolutions."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(
        self,
        input_dim,
        num_channels=[64, 128, 128],
        kernel_size=3,
        dropout=0.2,
        output_len=28
    ):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
        
        # Multi-layer output head
        self.output_head = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], output_len)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)
        
        # Use last timestep
        x = x[:, :, -1]
        output = self.output_head(x)
        
        return output


# ============================================================================
# MODEL FACTORY
# ============================================================================

def get_model(model_name, input_dim, output_len, config):
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model
        input_dim: Number of input features
        output_len: Forecast horizon
        config: Model-specific configuration dict
        
    Returns:
        Initialized model
    """
    models = {
        'RNN': SimpleRNN,
        'GRU': GRU,
        'LSTM': LSTM,
        'TCN': TCN,
        'Informer': Informer,
        'Autoformer': Autoformer,
        'FEDformer': FEDformer
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](input_dim=input_dim, output_len=output_len, **config)
