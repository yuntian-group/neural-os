import torch
import numpy as np
from PIL import Image, ImageDraw
from utils import initialize_model
from ldm.models.diffusion.ddpm import DDIMSampler
import json
import os
import random
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
# ----- Configuration -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCREEN_WIDTH, SCREEN_HEIGHT = 512, 384
LATENT_DIMS = (16, SCREEN_HEIGHT // 8, SCREEN_WIDTH // 8)
NUM_SAMPLING_STEPS = 32
NUM_SAMPLING_STEPS = 1000
NUM_SAMPLING_STEPS = 2

# Icons' positions
ICONS = {
    'home': {'center': (66, 175)},
    'close_btn': {'center': (499, 11)},
    'doom': {'center': (442, 48)},
}

# Load latent normalization stats
with open('../latent_stats.json', 'r') as f:
    latent_stats = json.load(f)
DATA_NORMALIZATION = {
    'mean': torch.tensor(latent_stats['mean']).to(device),
    'std': torch.tensor(latent_stats['std']).to(device)
}

# Load trained model
#model = initialize_model("config_final_model.yaml", "yuntian-deng/computer-model").to(device)
model = initialize_model("config_final_model_origunet_nospatial_x0.yaml", "yuntian-deng/computer-model-adddoom_play-8gpus-classic-fix-010000").to(device)
#model = initialize_model("config_final_model_origunet_nospatial_x0.yaml", "yuntian-deng/computer-model-adddoom_play-8gpus-classic-fix2-014000").to(device)
sampler = DDIMSampler(model)

# Padding image
padding_image = torch.zeros(*LATENT_DIMS).unsqueeze(0).to(device)
padding_image = (padding_image - DATA_NORMALIZATION['mean'].view(1, -1, 1, 1)) / DATA_NORMALIZATION['std'].view(1, -1, 1, 1)
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
INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                            'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
VALID_KEYS = [key for key in KEYS if key not in INVALID_KEYS]
itos = VALID_KEYS
stoi = {}
for i in range(len(itos)):
    stoi[itos[i]] = i
# Prepare model inputs
def prepare_input(prev_frame, hidden_states, x, y, left_click, timestep, keydown=None):
    key_events = torch.zeros(179, dtype=torch.long).to(device)
    if keydown is not None:
        key_events[stoi[keydown]] = 1
    inputs = {
        'image_features': prev_frame.to(device),
        'is_padding': torch.BoolTensor([timestep == 0]).to(device),
        'x': torch.LongTensor([x]).unsqueeze(0).to(device),
        'y': torch.LongTensor([y]).unsqueeze(0).to(device),
        'is_leftclick': torch.BoolTensor([left_click]).unsqueeze(0).to(device),
        'is_rightclick': torch.BoolTensor([False]).unsqueeze(0).to(device),
        'key_events': key_events,
    }
    if hidden_states is not None:
        inputs['hidden_states'] = hidden_states
    return inputs

# Single frame inference
@torch.no_grad()
def generate_frame(inputs):
    output_from_rnn, hidden_states = model.temporal_encoder.forward_step(inputs)
    if NUM_SAMPLING_STEPS >= 1000:
        sample_latent = model.p_sample_loop(cond={'c_concat': output_from_rnn}, shape=[1, *LATENT_DIMS], return_intermediates=False, verbose=False)
    else:
        sample_latent, _ = sampler.sample(
            S=NUM_SAMPLING_STEPS,
            conditioning={'c_concat': output_from_rnn},
            batch_size=1,
            shape=LATENT_DIMS,
            verbose=False
        )
    sample = sample_latent * DATA_NORMALIZATION['std'].view(1, -1, 1, 1) + DATA_NORMALIZATION['mean'].view(1, -1, 1, 1)
    sample = model.decode_first_stage(sample).squeeze(0).clamp(-1, 1)
    sample_img = ((sample[:3].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    return sample_latent, hidden_states, Image.fromarray(sample_img)

# Draw cursor and click indicators
def draw_cursor(img, x, y, clicked):
    x = x + 3
    y = y + 6
    draw = ImageDraw.Draw(img)
    radius = 12
    color = 'red' if clicked else 'black'
    color = 'red'
    draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], outline=color, width=3)
    return img

# Action sequence generator
def create_action_sequence():
    actions = []
    home_pos = ICONS['home']['center']
    doom_pos = ICONS['doom']['center']
    close_pos = ICONS['close_btn']['center']

    # Move to Home (a-b)
    #actions += [{'x': home_pos[0]-20, 'y': home_pos[1]-20, 'left_click': False}] * 3
    actions += [{'x': close_pos[0], 'y': close_pos[1], 'left_click': False}] * 3
    actions += [{'x': 512//2, 'y': 384//2, 'left_click': False}] * 1
    actions += [{'x': home_pos[0], 'y': home_pos[1], 'left_click': False}] * 3

    # Double click Home (c-f)
    actions += [{'x': home_pos[0], 'y': home_pos[1], 'left_click': True}]
    actions += [{'x': home_pos[0], 'y': home_pos[1], 'left_click': False}]
    actions += [{'x': home_pos[0], 'y': home_pos[1], 'left_click': True}]
    actions += [{'x': home_pos[0], 'y': home_pos[1], 'left_click': False}] * 5  # wait frames

    # Move cursor to close button (g-h)
    actions += [{'x': home_pos[0]+50, 'y': home_pos[1]-50, 'left_click': False}] * 1
    actions += [{'x': close_pos[0], 'y': close_pos[1], 'left_click': False}] * 1

    # Click close button (i-j)
    actions += [{'x': close_pos[0], 'y': close_pos[1], 'left_click': True}] * 2
    actions += [{'x': close_pos[0], 'y': close_pos[1], 'left_click': False}] * 5


    # Move cursor to close button (g-h)
    actions += [{'x': doom_pos[0], 'y': doom_pos[1], 'left_click': False}] * 2
    actions += [{'x': doom_pos[0], 'y': doom_pos[1], 'left_click': True}] * 2
    actions += [{'x': doom_pos[0], 'y': doom_pos[1], 'left_click': False}] * 5
    actions += [{'x': doom_pos[0], 'y': doom_pos[1], 'left_click': False, 'keydown': 'escape'}]
    #actions += [{'x': doom_pos[0], 'y': doom_pos[1], 'left_click': False}]
    #actions += [{'x': doom_pos[0], 'y': doom_pos[1], 'left_click': False, 'keydown': 'escape'}]
    actions += [{'x': doom_pos[0], 'y': doom_pos[1], 'left_click': False}] * 15
    #actions += [{'x': close_pos[0], 'y': close_pos[1], 'left_click': False}] * 1

    return actions

# Generate frames and visualize cursor
def main():
    output_dir = f"paper_neuralos_sequence_{NUM_SAMPLING_STEPS}_s{SEED}_3"
    os.makedirs(output_dir, exist_ok=True)

    prev_frame, hidden_states = padding_image, None
    actions = create_action_sequence()

    for t, action in enumerate(actions):
        inputs = prepare_input(
            prev_frame, hidden_states,
            x=action['x'],
            y=action['y'],
            left_click=action['left_click'],
            timestep=t,
            keydown=action['keydown'] if 'keydown' in action else None
        )
        prev_frame, hidden_states, img = generate_frame(inputs)

        # Visualize mouse
        img = draw_cursor(img, action['x'], action['y'], action['left_click'])

        # Save image
        filename = os.path.join(output_dir, f"frame_{t:02d}.png")
        img.save(filename)
        print(f'Saved: {filename}')

if __name__ == "__main__":
    main()
