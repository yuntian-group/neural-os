import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

def sinusoidal_init(num_positions, hidden_size):
    print ('INIT')
    """Generate sinusoidal embeddings similar to positional embeddings in Transformer."""
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() *
                         (-math.log(10000.0) / hidden_size))
    embeddings = torch.zeros((num_positions, hidden_size))
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings

def sinusoidal_init_2d(height, width, hidden_size):
    print ('INIT 2D')
    """
    Generate 2D sinusoidal positional embeddings.
    Args:
        height (int): Height of the spatial dimension.
        width (int): Width of the spatial dimension.
        hidden_size (int): Embedding dimension (must be divisible by 4).
    Returns:
        embeddings: (height * width, hidden_size)
    """
    if hidden_size % 4 != 0:
        raise ValueError("hidden_size must be divisible by 4")

    embeddings = torch.zeros((height, width, hidden_size))

    half_dim = hidden_size // 2
    div_term = torch.exp(torch.arange(0, half_dim, 2).float() *
                         -(math.log(10000.0) / half_dim))

    # Positional indices
    y_pos = torch.arange(height).unsqueeze(1)
    x_pos = torch.arange(width).unsqueeze(1)

    # Y-dimension embeddings
    embeddings[:, :, 0:half_dim:2] = torch.sin(y_pos * div_term).unsqueeze(1).repeat(1, width, 1)
    embeddings[:, :, 1:half_dim:2] = torch.cos(y_pos * div_term).unsqueeze(1).repeat(1, width, 1)

    # X-dimension embeddings
    embeddings[:, :, half_dim::2] = torch.sin(x_pos * div_term).unsqueeze(0).repeat(height, 1, 1)
    embeddings[:, :, half_dim+1::2] = torch.cos(x_pos * div_term).unsqueeze(0).repeat(height, 1, 1)

    # Flatten embeddings
    embeddings = embeddings.reshape(1, height * width, hidden_size)
    return embeddings


class TemporalEncoder(nn.Module):
    def init_weights(self):
        # Initialize LSTM layers
        for lstm in [self.lstm_lower, self.lstm_upper]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                #param.data.uniform_(-0.05, 0.05)
                #param.data.uniform_(-0.15, 0.15)
    
        # Initialize projection layers
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        #for layer in [self.lstm_projection_pre, self.lstm_projection_post, self.image_feature_projection]:
        for layer in [self.lstm_projection_pre, self.lstm_projection_post, self.image_feature_projection]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.embedding_x.weight.data = sinusoidal_init(self.output_width*8, self.hidden_size)
        self.embedding_y.weight.data = sinusoidal_init(self.output_height*8, self.hidden_size)
        self.image_position_embeddings.data = sinusoidal_init_2d(self.output_height, self.output_width, self.input_channels*8*8)
        
        for param in [
            self.initial_state_padding_h_lower,
            self.initial_state_unknown_h_lower,
            self.initial_state_padding_h_upper,
            self.initial_state_unknown_h_upper,
            self.initial_state_padding_c_lower,
            self.initial_state_unknown_c_lower,
            self.initial_state_padding_c_upper,
            self.initial_state_unknown_c_upper,
            self.initial_feedback_padding,
            self.initial_feedback_unknown,
        ]:
            nn.init.zeros_(param)


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

        self.image_position_embeddings = nn.Parameter(torch.randn(1, self.output_height*self.output_width, self.input_channels*8*8))
        self.image_feature_projection = nn.Linear(self.input_channels, self.input_channels*8*8)
        self.lstm_projection_pre = nn.Linear(hidden_size, self.input_channels*8*8)
        self.lstm_projection_post = nn.Linear(self.input_channels*8*8, hidden_size)
        self.embedding_x = nn.Embedding(self.output_width * 8, hidden_size)
        self.embedding_y = nn.Embedding(self.output_height * 8, hidden_size)
        self.embedding_is_leftclick = nn.Embedding(2, hidden_size)
        self.embedding_is_rightclick = nn.Embedding(2, hidden_size)
        self.embedding_key_events = nn.Embedding(len(self.itos)*2, hidden_size)
        #print ('dfsdfsfgsgdsd')
        #self.input_projection = nn.Sequential(
        #    nn.Linear(hidden_size*4, hidden_size*4),
        #    nn.LeakyReLU(),
        #)
        self.initial_feedback_padding = nn.Parameter(torch.randn(1, hidden_size))
        self.initial_feedback_unknown = nn.Parameter(torch.randn(1, hidden_size))
        self.multi_head_attention = nn.MultiheadAttention(self.input_channels*8*8, num_heads=8, batch_first=True)
        
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

        self.log_sigma = nn.Parameter(torch.tensor(math.log(1.0)))
        print ('fixing sigma during pretraining')
        self.log_sigma.requires_grad = False
        
        
        # Project LSTM output to desired spatial feature map
        self.projection = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size*4, self.output_channels * output_height * output_width),
        )
        self.init_weights()
    # TODO: maybe use a CNN to process the sequence
    # TODO: maybe use layernorm to preprocess the input

    def forward(self, inputs, sampler=None, scheduled_sampling_length=None, scheduled_sampling_ddim_steps=None, first_stage_model=None):
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
        offset = torch.arange(len(self.itos), device=self.embedding_x.weight.data.device) * 2
        embedding_key_events_baseline = self.embedding_key_events(offset.unsqueeze(0)) # bsz, num_keys, hidden_size
        embedding_key_events_baseline = embedding_key_events_baseline.sum(dim=1) # bsz, hidden_size
        for t in range(sequence_length):
            #if t == sequence_length - 1:
            #    import pdb; pdb.set_trace()
            inputs_t = inputs[t]
            is_padding = inputs_t['is_padding']
            x = inputs_t['x'].squeeze(-1)
            y = inputs_t['y'].squeeze(-1)
            is_leftclick = inputs_t['is_leftclick'].squeeze(-1).long()
            is_rightclick = inputs_t['is_rightclick'].squeeze(-1).long()
            key_events = inputs_t['key_events'] # bsz, len(self.itos)
            if True or is_padding.any(): # for ddp
                hidden_states_h_lower = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_h_lower, hidden_states_h_lower) # 1, bsz, hidden_size
                hidden_states_h_upper = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_h_upper, hidden_states_h_upper) # 1, bsz, hidden_size
                hidden_states_c_lower = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_c_lower, hidden_states_c_lower) # 1, bsz, hidden_size
                hidden_states_c_upper = torch.where(is_padding.view(1, batch_size, 1), self.initial_state_padding_c_upper, hidden_states_c_upper) # 1, bsz, hidden_size
                feedback = torch.where(is_padding.unsqueeze(-1), self.initial_feedback_padding, feedback) # bsz, hidden_size
            #print ('nounk')
            
            embedding_x = self.embedding_x(x) # bsz, hidden_size    
            embedding_y = self.embedding_y(y) # bsz, hidden_size
            embedding_is_leftclick = self.embedding_is_leftclick(is_leftclick) # bsz, hidden_size
            embedding_is_rightclick = self.embedding_is_rightclick(is_rightclick) # bsz, hidden_size

            # compute embedding for key events
            embedding_key_events = self.embedding_key_events(key_events + offset.unsqueeze(0)) # bsz, num_keys, hidden_size
            embedding_key_events = embedding_key_events.sum(dim=1) # bsz, hidden_size
            #import pdb; pdb.set_trace()
            embedding_key_events = embedding_key_events - embedding_key_events_baseline

            embedding_all = embedding_is_leftclick + embedding_is_rightclick + embedding_key_events
            #embedding_input = embedding_x + embedding_y + embedding_all*0 + feedback*0
            #print ('only x y')
            #embedding_input = embedding_x + embedding_y + embedding_all + feedback
            embedding_input = torch.cat([embedding_x, embedding_y, embedding_all, feedback], dim=-1) # bsz, hidden_size*4
            #embedding_input = self.input_projection(embedding_input) # bsz, hidden_size*4
            embedding_input = embedding_input.unsqueeze(1) # bsz, 1, hidden_size*4

            
            image_features = inputs_t['image_features'] # bsz, num_channels, height, width

            if sampler is not None and t >= sequence_length - scheduled_sampling_length:
                #import pdb; pdb.set_trace()
                # replace image_features with the sampled image
                with torch.no_grad():
                    hidden_last = torch.cat([lstm_out_upper, lstm_out_lower], dim=-1)
                    output = self.projection(hidden_last)
                    output = output.reshape(batch_size, self.output_channels, self.output_height, self.output_width)
                    # concatenate output with Gaussian kernel of positions
                    device = output.device
                    y_grid = torch.arange(self.output_height, device=device).view(1, -1, 1)
                    x_grid = torch.arange(self.output_width, device=device).view(1, 1, -1)
                    sigma = torch.exp(self.log_sigma)
                    #import pdb; pdb.set_trace()
                    kernel = torch.exp(-((x_grid - (x_prev/8.0).view(-1, 1, 1))**2 + (y_grid - (y_prev/8.0).view(-1, 1, 1))**2) / (2 * sigma**2)).unsqueeze(1)
                    output = torch.cat([output[:, :-1], kernel], dim=1)
                    c_dict = {'c_concat': output}
                    samples_ddim, _ = sampler.sample(S=scheduled_sampling_ddim_steps,
                                            conditioning=c_dict,
                                            batch_size=batch_size,
                                            shape=[self.input_channels, self.output_height, self.output_width],
                                            verbose=False,)
                    # only apply sampling mask where is_padding is False
                    # save images for debugging
                    DEBUG = False
                    if DEBUG:
                        import pdb; pdb.set_trace()
                        decode_batch_size = 1
                        samples = samples_ddim * self.per_channel_std.view(1, -1, 1, 1) + self.per_channel_mean.view(1, -1, 1, 1)
                        for idx in range(0, samples.shape[0], decode_batch_size):
                                batch_samples = samples[idx:min(idx + decode_batch_size, samples.shape[0])]
                                batch_decoded = first_stage_model.decode(batch_samples)
                                batch_encoded = torch.clamp(batch_decoded, min=-1.0, max=1.0)
                                batch_encoded_images = batch_encoded * 127.5 + 127.5
                                for kkk in range(batch_samples.shape[0]):
                                    image = batch_encoded_images[kkk].permute(1, 2, 0).cpu().numpy()
                                    image = image.astype(np.uint8)
                                    image = Image.fromarray(image)
                                    image.save(f'scheduled_sampling_ddim_step_{idx}_{t}_sample_{kkk}.png')
                                #x_samples_ddim.append(batch_decoded)
                                batch_gt = image_features[idx:min(idx + decode_batch_size, samples.shape[0])]
                                batch_samples = batch_gt * self.per_channel_std.view(1, -1, 1, 1) + self.per_channel_mean.view(1, -1, 1, 1)
                                batch_decoded = first_stage_model.decode(batch_samples)
                                batch_encoded = torch.clamp(batch_decoded, min=-1.0, max=1.0)
                                batch_encoded_images = batch_encoded * 127.5 + 127.5
                                for kkk in range(batch_samples.shape[0]):
                                    image = batch_encoded_images[kkk].permute(1, 2, 0).cpu().numpy()
                                    image = image.astype(np.uint8)
                                    image = Image.fromarray(image)
                                    image.save(f'scheduled_sampling_ddim_step_{idx}_{t}_sample_{kkk}.gt.png')
                                #batch_encoded = self.encode_first_stage(batch_decoded).sample()
                                #z_samples.append(batch_encoded)
                                #z_samples.append(batch_samples)
                    image_features = torch.where(is_padding.view(-1, 1, 1, 1), image_features, samples_ddim)
            assert image_features.shape[1] == self.input_channels, f"image_features.shape[1] = {image_features.shape[-1]} != self.input_channels = {self.input_channels}"
            image_features = torch.einsum('bchw->bhwc', image_features).reshape(batch_size, -1, self.input_channels)
            #image_features_with_position = image_features + self.image_position_embeddings
            image_features_with_position = image_features
            image_features_with_position = self.image_feature_projection(image_features_with_position)
            image_features_with_position = image_features_with_position + self.image_position_embeddings

            lstm_out_lower, (hidden_states_h_lower, hidden_states_c_lower) = self.lstm_lower(embedding_input, (hidden_states_h_lower, hidden_states_c_lower))

            # apply multi-headed attention to attend lstm_out_lower to image_features_with_position
            context, attention_weights = self.multi_head_attention(self.lstm_projection_pre(lstm_out_lower), image_features_with_position, image_features_with_position, need_weights=False, average_attn_weights=False)
            #context, attention_weights = self.multi_head_attention(lstm_out_lower[..., :image_features_with_position.shape[-1]], image_features_with_position, image_features_with_position, need_weights=False, average_attn_weights=False)
            #print ('NO CONTEXT')
            #context = 0*self.lstm_projection_post(context) + lstm_out_lower
            context = self.lstm_projection_post(context) + lstm_out_lower

            # visualize attention weights and also x and y positions in the same image, but only for the first element in the batch
            if False and self.num_times % 1000 == 0:
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
            x_prev = x
            y_prev = y
        
        hidden_last = torch.cat([lstm_out_upper, lstm_out_lower], dim=-1)
        
        # Project to desired output shape
        output = self.projection(hidden_last)
        
        # Reshape to spatial feature map: [B, output_channels*output_height*output_width] -> [B, output_channels, output_height, output_width]
        output = output.reshape(batch_size, self.output_channels, self.output_height, self.output_width)
        # concatenate output with Gaussian kernel of positions
        device = output.device
        y_grid = torch.arange(self.output_height, device=device).view(1, -1, 1)
        x_grid = torch.arange(self.output_width, device=device).view(1, 1, -1)
        sigma = torch.exp(self.log_sigma)
        #import pdb; pdb.set_trace()
        kernel = torch.exp(-((x_grid - (x/8.0).view(-1, 1, 1))**2 + (y_grid - (y/8.0).view(-1, 1, 1))**2) / (2 * sigma**2)).unsqueeze(1)
        output = torch.cat([output[:, :-1], kernel], dim=1)
        #print ('cheating')
        #output[:, 0, :, :] = 0
        #output[torch.arange(batch_size), 0, (y//8).long(), (x//8).long()] = 1
        
        return output

    def encode(self, x):
        """Alias for forward() to match the encoder interface"""
        return self.forward(x) 

    def forward_step(self, input):
        """
        Args:
            input: a dictionary containing the following keys:
                'image_features': Tensor of shape [B, C, H, W]
                'is_padding': Tensor of shape [B]
                'x': Tensor of shape [B, 1]
                'y': Tensor of shape [B, 1]
                'is_leftclick': Tensor of shape [B, 1]
                'is_rightclick': Tensor of shape [B, 1]
                'key_events': Tensor of shape [B, len(self.itos)]
                'hidden_states': dictionary of previous hidden states
        Returns:
            output: Tensor of shape [B, output_channels, output_height, output_width]
        """
        # initial RNN state: if starts with padding, then use padding state, otherwise use unknown state
        #import pdb; pdb.set_trace()
        if not hasattr(self, 'num_times'):
            self.num_times = 0
        self.num_times += 1

        batch_size = input['image_features'].shape[0]

        if 'hidden_states' in input:
            hidden_states_h_lower = input['hidden_states']['h_lower']
            hidden_states_h_upper = input['hidden_states']['h_upper']
            hidden_states_c_lower = input['hidden_states']['c_lower']
            hidden_states_c_upper = input['hidden_states']['c_upper']
            feedback = input['hidden_states']['feedback']
        else:
            hidden_states_h_lower = self.initial_state_unknown_h_lower.repeat(1, batch_size, 1) # bsz, hidden_size
            hidden_states_h_upper = self.initial_state_unknown_h_upper.repeat(1, batch_size, 1) # bsz, hidden_size
            hidden_states_c_lower = self.initial_state_unknown_c_lower.repeat(1, batch_size, 1) # bsz, hidden_size
            hidden_states_c_upper = self.initial_state_unknown_c_upper.repeat(1, batch_size, 1) # bsz, hidden_size
            feedback = self.initial_feedback_unknown.repeat(batch_size, 1) # bsz, hidden_size
        #import pdb; pdb.set_trace()

        inputs_t = input
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
        embedding_key_events_baseline = self.embedding_key_events(offset.unsqueeze(0)) # bsz, num_keys, hidden_size
        embedding_key_events_baseline = embedding_key_events_baseline.sum(dim=1) # bsz, hidden_size
        embedding_key_events = embedding_key_events - embedding_key_events_baseline

        embedding_all = embedding_is_leftclick + embedding_is_rightclick + embedding_key_events
        embedding_input = torch.cat([embedding_x, embedding_y, embedding_all, feedback], dim=-1) # bsz, hidden_size*4
        #embedding_input = embedding_x + embedding_y + embedding_all + feedback
        #embedding_input = self.input_projection(embedding_input) # bsz, hidden_size*4
        embedding_input = embedding_input.unsqueeze(1) # bsz, 1, hidden_size*4

        
        lstm_out_lower, (hidden_states_h_lower, hidden_states_c_lower) = self.lstm_lower(embedding_input, (hidden_states_h_lower, hidden_states_c_lower))
        image_features = inputs_t['image_features'] # bsz, num_channels, height, width
        assert image_features.shape[1] == self.input_channels, f"image_features.shape[1] = {image_features.shape[-1]} != self.input_channels = {self.input_channels}"
        image_features = torch.einsum('bchw->bhwc', image_features).reshape(batch_size, -1, self.input_channels)
        #image_features_with_position = image_features + self.image_position_embeddings
        image_features_with_position = image_features
        image_features_with_position = self.image_feature_projection(image_features_with_position)
        image_features_with_position = image_features_with_position + self.image_position_embeddings
        # apply multi-headed attention to attend lstm_out_lower to image_features_with_position
        context, attention_weights = self.multi_head_attention(self.lstm_projection_pre(lstm_out_lower), image_features_with_position, image_features_with_position, need_weights=False, average_attn_weights=False)
        context = self.lstm_projection_post(context) + lstm_out_lower

        # visualize attention weights and also x and y positions in the same image, but only for the first element in the batch
        if self.num_times % 1000 == 0 and False:
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
        
        hidden_last = torch.cat([lstm_out_upper, lstm_out_lower], dim=-1)
        
        # Project to desired output shape
        output = self.projection(hidden_last)
        
        # Reshape to spatial feature map: [B, output_channels*output_height*output_width] -> [B, output_channels, output_height, output_width]
        output = output.reshape(batch_size, self.output_channels, self.output_height, self.output_width)
        device = output.device
        y_grid = torch.arange(self.output_height, device=device).view(1, -1, 1)
        x_grid = torch.arange(self.output_width, device=device).view(1, 1, -1)
        sigma = torch.exp(self.log_sigma)
        #import pdb; pdb.set_trace()
        kernel = torch.exp(-((x_grid - (x/8.0).view(-1, 1, 1))**2 + (y_grid - (y/8.0).view(-1, 1, 1))**2) / (2 * sigma**2)).unsqueeze(1)
        output = torch.cat([output[:, :-1], kernel], dim=1)

        hidden_states = {
            'h_lower': hidden_states_h_lower,
            'h_upper': hidden_states_h_upper,
            'c_lower': hidden_states_c_lower,
            'c_upper': hidden_states_c_upper,
            'feedback': feedback
        }
        
        return output, hidden_states
