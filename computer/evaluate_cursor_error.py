import os
import json
import random
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from latent_diffusion.ldm.models.diffusion.ddpm import DDIMSampler
from omegaconf import OmegaConf

from utils import initialize_model
from train_coordinate_predictor import CoordinateTrainer

# ------------------------------------------------------------------------------
# Config / hyperparameters
# ------------------------------------------------------------------------------
SEED = 1234
NUM_SAMPLES = 730

# Screen / latent dims
SCREEN_W, SCREEN_H = 512, 384
LATENT_DIMS = (16, SCREEN_H // 8, SCREEN_W // 8)
NUM_SAMPLING_STEPS = 32

# ------------------------------------------------------------------------------
# Utilities: prepare inputs / generate a single frame
# ------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
torch.manual_seed(SEED)

# load latent normalization
with open('../latent_stats.json','r') as f:
    stats = json.load(f)
MEAN = torch.tensor(stats['mean'], device=device)
STD  = torch.tensor(stats['std'], device=device)

# load NeuralOS model
def load_neuralos(config_path, ckpt):
    model = initialize_model(config_path, ckpt).to(device)
    sampler = DDIMSampler(model)
    # padding (empty) frame
    pad = torch.zeros(1, *LATENT_DIMS, device=device)
    pad = (pad - MEAN.view(1,-1,1,1)) / STD.view(1,-1,1,1)
    return model, sampler, pad

def prepare_input(prev_frame, hidden, x, y):
    inp = {
        'image_features': prev_frame,
        'is_padding': torch.BoolTensor([False]).to(device),
        'x': torch.LongTensor([x]).unsqueeze(0).to(device),
        'y': torch.LongTensor([y]).unsqueeze(0).to(device),
        'is_leftclick': torch.BoolTensor([False]).unsqueeze(0).to(device),
        'is_rightclick': torch.BoolTensor([False]).unsqueeze(0).to(device),
        'key_events': torch.zeros(179, dtype=torch.long, device=device),
    }
    if hidden is not None:
        inp['hidden_states'] = hidden
    return inp

@torch.no_grad()
def generate_frame(model, sampler, inputs):
    out_rnn, hidden = model.temporal_encoder.forward_step(inputs)
    if NUM_SAMPLING_STEPS >= 1000:
        latent = model.p_sample_loop(cond={'c_concat': out_rnn},
                                     shape=[1, *LATENT_DIMS], verbose=False)
    else:
        latent, _ = sampler.sample(
            S=NUM_SAMPLING_STEPS,
            conditioning={'c_concat': out_rnn},
            batch_size=1, shape=LATENT_DIMS, verbose=False
        )
    # decode
    img_lat = latent * STD.view(1,-1,1,1) + MEAN.view(1,-1,1,1)
    img = model.decode_first_stage(img_lat).squeeze(0).clamp(-1,1)
    arr = ((img[:3].permute(1,2,0).cpu().numpy()+1)*127.5).astype(np.uint8)
    return Image.fromarray(arr), hidden

def draw_cursor(img, x, y):
    draw = ImageDraw.Draw(img)
    r = 6
    draw.ellipse([(x-r,y-r),(x+r,y+r)], outline='red', width=2)
    return img

# ------------------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--neuralos-config',   type=str, default='config_final_model.yaml')
    p.add_argument('--neuralos-checkpoint', type=str, default='yuntian-deng/computer-model-ss005-cont-lr2e5-384k')
    p.add_argument('--coord-config',      type=str, default='DEBUG.yaml')
    p.add_argument('--coord-checkpoint',  type=str, default='cursor_position_model.ckpt')
    p.add_argument('--output-dir',        type=str, default='cursor_eval')
    p.add_argument('--num-samples',       type=int, default=NUM_SAMPLES)
    opts = p.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    # load NeuralOS
    neuralos, sampler, pad = load_neuralos(opts.neuralos_config,
                                          opts.neuralos_checkpoint)

    # load coordinate predictor
    cconf = OmegaConf.load(opts.coord_config)
    coord = CoordinateTrainer.load_from_checkpoint(opts.coord_checkpoint,
                image_predictor_config=cconf.image_predictor_model)
    coord = coord.to(device)
    coord.eval()

    errors = []
    prev_frame, hidden = pad, None

    for i in tqdm(range(opts.num_samples), desc="eval cursor"):
        # sample random position
        x = random.randint(0, SCREEN_W-1)
        y = random.randint(0, SCREEN_H-1)

        # prep & generate
        inp = prepare_input(prev_frame, hidden, x, y)
        frame, hidden = generate_frame(neuralos, sampler, inp)
        # draw actual cursor position
        #frame = draw_cursor(frame, x, y)

        # save for debug
        frame_path = os.path.join(opts.output_dir, f'frame_{i:04d}.png')
        frame.save(frame_path)

        # predict
        #import pdb; pdb.set_trace()
        res = coord.predict(frame_path, x, y, return_overlay=False)
        px, py = res['predicted_x'], res['predicted_y']

        dx = abs(px - x)
        dy = abs(py - y)
        dr = (dx*dx + dy*dy)**0.5
        errors.append((dx, dy, dr))
        errs = np.array(errors)
        mx, my, mr = errs.mean(axis=0)
        print(f"Δx = {mx:.3f}, Δy = {my:.3f}, Δr = {mr:.3f}")

        # next iteration uses last generated frame
        #prev_frame = (torch.from_numpy(np.transpose(np.array(frame), (2,0,1))).unsqueeze(0).float() / 127.5 -1.0).to(device)
        # actually convert back into latent normalized format by encoding through autoencoder is costly,
        # so for simplicity we keep pad frames for evaluation (no recurrence).
        prev_frame = pad
        hidden = None

    # compute means
    errs = np.array(errors)
    mx, my, mr = errs.mean(axis=0)
    print(f"Δx = {mx:.3f}, Δy = {my:.3f}, Δr = {mr:.3f}")

if __name__ == '__main__':
    main()

