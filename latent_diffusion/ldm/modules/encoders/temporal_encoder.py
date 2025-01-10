import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_channels: int,
        output_height: int,
        output_width: int
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.output_height = output_height
        self.output_width = output_width
        
        # LSTM to process the sequence
        self.lstm = nn.LSTM(
            input_size=input_channels * output_height * output_width,  # Flattened input size
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        
        # Project LSTM output to desired spatial feature map
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size*4, output_channels * output_height * output_width),
        )
    # TODO: maybe use a CNN to process the sequence
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, C, H, W]
                B: batch size
                T: sequence length
                C: input channels
                H: height
                W: width
        Returns:
            output: Tensor of shape [B, output_channels, output_height, output_width]
        """
        #import pdb; pdb.set_trace()
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        assert sequence_length == 14, "Sequence length must be 14"
        
        # Flatten spatial dimensions: [B, T, C, H, W] -> [B, T, C*H*W]

        x_flat = x.reshape(batch_size, sequence_length, -1)
        
        # Process through LSTM
        lstm_out, (hidden, _) = self.lstm(x_flat)
        
        # Use the last hidden state
        hidden_last = hidden[-1]
        
        # Project to desired output shape
        output = self.projection(hidden_last)
        
        # Reshape to spatial feature map: [B, output_channels*output_height*output_width] -> [B, output_channels, output_height, output_width]
        output = output.reshape(batch_size, self.output_channels, self.output_height, self.output_width)
        
        return output

    def encode(self, x):
        """Alias for forward() to match the encoder interface"""
        return self.forward(x) 