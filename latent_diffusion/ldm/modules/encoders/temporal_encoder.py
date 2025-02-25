import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
        KEYS = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
        ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
        'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
        'browserback', 'browserfavorites', 'browserforward', 'browserhome',
        'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
        'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
        'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
        'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
        'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
        'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
        'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
        'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
        'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
        'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
        'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
        'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
        'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
        'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
        'command', 'option', 'optionleft', 'optionright']
        INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
        KEYS = [key for key in KEYS if key not in INVALID_KEYS]

        self.itos = KEYS
        self.stoi = {key: i for i, key in enumerate(KEYS)}
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channels = output_channels
        #self.TRIM_BEGINNING = 1
        #self.TRIM_BEGINNING = 1
        #if self.TRIM_BEGINNING == 1:
        #    self.output_channels = output_channels + 2
        self.output_height = output_height
        self.output_width = output_width

        self.initial_state_padding_h_lower = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_state_unknown_h_lower = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_state_padding_h_upper = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_state_unknown_h_upper = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_state_padding_c_lower = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_state_unknown_c_lower = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_state_padding_c_upper = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_state_unknown_c_upper = nn.Parameter(torch.randn(1, 1, hidden_size))
        assert hidden_size % 8 == 0, "hidden_size must be divisible by 8"

        self.image_position_embeddings = nn.Parameter(torch.randn(1, self.output_height*self.output_width, hidden_size))
        self.image_feature_projection = nn.Linear(4, hidden_size)
        self.embedding_x = nn.Embedding(self.output_width * 8, hidden_size)
        self.embedding_y = nn.Embedding(self.output_height * 8, hidden_size)
        self.embedding_is_leftclick = nn.Embedding(2, hidden_size)
        self.embedding_is_rightclick = nn.Embedding(2, hidden_size)
        self.embedding_key_events = nn.Embedding(len(self.itos)*2, hidden_size)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_size*4, hidden_size*4),
            nn.ReLU(),
        )
        self.initial_feedback_padding = nn.Parameter(torch.randn(1, hidden_size))
        self.initial_feedback_unknown = nn.Parameter(torch.randn(1, hidden_size))
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # LSTM to process the sequence
        self.lstm_lower = nn.LSTM(
            input_size=hidden_size*4,  # Flattened input size
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_upper = nn.LSTM(
            input_size=hidden_size,  # Flattened input size
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        
        # Project LSTM output to desired spatial feature map
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.ReLU(),
            nn.Linear(hidden_size*4, self.output_channels * output_height * output_width),
        )
    # TODO: maybe use a CNN to process the sequence
    # TODO: maybe use aligned images and position maps
    # TODO: maybe use layernorm to preprocess the input
    #(Pdb) p inputs[0]['image_features'].shape

    def forward(self, inputs):
        """
        Args:
            inputs: a list of dictionaries, each containing the following keys:
                'image_features': Tensor of shape [B, C, H, W]
                'is_padding': Tensor of shape [B]
                'x': Tensor of shape [B, 1]
                'y': Tensor of shape [B, 1]
                'is_leftclick': Tensor of shape [B, 1]
                'is_rightclick': Tensor of shape [B, 1]
                'key_events': Tensor of shape [B, len(self.itos)]
        Returns:
            output: Tensor of shape [B, output_channels, output_height, output_width]
        """
        # initial RNN state: if starts with padding, then use padding state, otherwise use unknown state
        #import pdb; pdb.set_trace()
        if not hasattr(self, 'num_times'):
            self.num_times = 0
        self.num_times += 1

        batch_size = inputs[0]['image_features'].shape[0]
        hidden_states_h_lower = self.initial_state_unknown_h_lower.repeat(1, batch_size, 1) # bsz, hidden_size
        hidden_states_h_upper = self.initial_state_unknown_h_upper.repeat(1, batch_size, 1) # bsz, hidden_size
        hidden_states_c_lower = self.initial_state_unknown_c_lower.repeat(1, batch_size, 1) # bsz, hidden_size
        hidden_states_c_upper = self.initial_state_unknown_c_upper.repeat(1, batch_size, 1) # bsz, hidden_size
        feedback = self.initial_feedback_unknown.repeat(batch_size, 1) # bsz, hidden_size
        sequence_length = len(inputs)
        #import pdb; pdb.set_trace()
        for t in range(sequence_length):
            inputs_t = inputs[t]
            is_padding = inputs_t['is_padding']
            x = inputs_t['x'].squeeze(-1)
            y = inputs_t['y'].squeeze(-1)
            is_leftclick = inputs_t['is_leftclick'].squeeze(-1).long()
            is_rightclick = inputs_t['is_rightclick'].squeeze(-1).long()
            key_events = inputs_t['key_events'] # bsz, len(self.itos)
            if is_padding.any():
                hidden_states_h_lower = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_h_lower, hidden_states_h_lower) # 1, bsz, hidden_size
                hidden_states_h_upper = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_h_upper, hidden_states_h_upper) # 1, bsz, hidden_size
                hidden_states_c_lower = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_c_lower, hidden_states_c_lower) # 1, bsz, hidden_size
                hidden_states_c_upper = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_c_upper, hidden_states_c_upper) # 1, bsz, hidden_size
                feedback = torch.where(is_padding.unsqueeze(-1), self.initial_feedback_padding, feedback) # bsz, hidden_size
            
            embedding_x = self.embedding_x(x) # bsz, hidden_size    
            embedding_y = self.embedding_y(y) # bsz, hidden_size
            embedding_is_leftclick = self.embedding_is_leftclick(is_leftclick) # bsz, hidden_size
            embedding_is_rightclick = self.embedding_is_rightclick(is_rightclick) # bsz, hidden_size

            # compute embedding for key events
            offset = torch.arange(len(self.itos), device=embedding_x.device) * 2
            embedding_key_events = self.embedding_key_events(key_events + offset.unsqueeze(0)) # bsz, num_keys, hidden_size
            embedding_key_events = embedding_key_events.sum(dim=1) # bsz, hidden_size

            embedding_all = embedding_is_leftclick + embedding_is_rightclick + embedding_key_events
            embedding_input = torch.cat([embedding_x, embedding_y, embedding_all, feedback], dim=-1) # bsz, hidden_size*4
            embedding_input = self.input_projection(embedding_input) # bsz, hidden_size*4
            embedding_input = embedding_input.unsqueeze(1) # bsz, 1, hidden_size*4

            
            lstm_out_lower, (hidden_states_h_lower, hidden_states_c_lower) = self.lstm_lower(embedding_input, (hidden_states_h_lower, hidden_states_c_lower))
            image_features = inputs_t['image_features'] # bsz, num_channels, height, width
            image_features = torch.einsum('bchw->bhwc', image_features).reshape(batch_size, -1, 4)
            image_features = self.image_feature_projection(image_features)
            image_features_with_position = image_features + self.image_position_embeddings
            # apply multi-headed attention to attend lstm_out_lower to image_features_with_position
            context, attention_weights = self.multi_head_attention(lstm_out_lower, image_features_with_position, image_features_with_position, need_weights=True, average_attn_weights=False)
            context = context + lstm_out_lower

            # visualize attention weights and also x and y positions in the same image, but only for the first element in the batch
            if self.num_times % 1000 == 0:
                # Get first batch element's attention weights and coordinates
                attn = attention_weights[0].reshape(-1, self.output_height, self.output_width)  # [num_heads, H, W]
                x_pos = x[0].item()
                y_pos = y[0].item()
                is_click = is_leftclick[0].item()
                
                # Create subplot for each attention head
                num_heads = attn.shape[0]
                fig, axes = plt.subplots(1, num_heads, figsize=(4*num_heads, 4))
                if num_heads == 1:
                    axes = [axes]
                
                for head_idx, ax in enumerate(axes):
                    # Plot attention heatmap
                    im = ax.imshow(attn[head_idx].detach().cpu(), cmap='Greens')
                    
                    # Plot click/no-click circle
                    circle_color = 'red' if is_click else 'yellow'
                    circle = plt.Circle((x_pos/8, y_pos/8), 3, color=circle_color, fill=False, linewidth=2)
                    ax.add_patch(circle)
                    
                    ax.set_title(f'Head {head_idx+1}')
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                plt.savefig(f'attention_vis_{self.num_times}_{t:03d}.png')
                plt.close()
            
            lstm_out_upper, (hidden_states_h_upper, hidden_states_c_upper) = self.lstm_upper(context, (hidden_states_h_upper, hidden_states_c_upper))
            feedback = lstm_out_upper.squeeze(1)
        
        hidden_last = lstm_out_upper
        
        # Project to desired output shape
        output = self.projection(hidden_last)
        
        # Reshape to spatial feature map: [B, output_channels*output_height*output_width] -> [B, output_channels, output_height, output_width]
        output = output.reshape(batch_size, self.output_channels, self.output_height, self.output_width)
        
        return output

    def encode(self, x):
        """Alias for forward() to match the encoder interface"""
        return self.forward(x) 
