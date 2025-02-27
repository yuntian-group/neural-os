#save_path = 'test_15_no_deltas_1000_paths'
from PIL import Image
import random
import torch
import numpy as np
import argparse
from omegaconf import OmegaConf
import time, os
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler
from computer.util import load_cond_from_config, load_first_stage_from_config, load_model, load_model_from_config, get_ground_truths, init_model, load_autoencoder_from_ckpt, load_cond_from_ckpt

# Set up GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

##Parse args here

def data_sample():


    x = random.randint(0, 512-1)
    y = random.randint(0, 384-1)
    right_click = random.randint(0, 1)
    left_click = random.randint(0, 1)


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

    i = random.randint(0, len(KEYS)-1)
    key = KEYS[i]

    itos = KEYS
    stoi = {key: i for i, key in enumerate(KEYS)}

    return x, y, right_click, left_click, key, stoi, itos

def create_sample(previous_frame, hidden_states, x, y, right_click, left_click, key, stoi, itos, t):

    inputs_t = {}
    inputs_t['image_features'] = previous_frame.to(device)
    inputs_t['is_padding'] = torch.BoolTensor([True if t == 0 else False]).to(device)
    inputs_t['x'] = torch.LongTensor([x if x is not None else 0]).unsqueeze(0).to(device)
    inputs_t['y'] = torch.LongTensor([y if y is not None else 0]).unsqueeze(0).to(device)
    inputs_t['is_leftclick'] = torch.BoolTensor([left_click]).unsqueeze(0).to(device)
    inputs_t['is_rightclick'] = torch.BoolTensor([right_click]).unsqueeze(0).to(device)
    inputs_t['key_events'] = torch.LongTensor([0 for _ in itos]).to(device)
    inputs_t['key_events'][stoi[key]] = 1
    if hidden_states is not None:
        if isinstance(hidden_states, tuple):
            inputs_t['hidden_states'] = tuple(h.to(device) for h in hidden_states)
        else:
            inputs_t['hidden_states'] = hidden_states

    return inputs_t

@torch.no_grad()
def sample_model(model, config, inputs_t):
    # Time temporal encoder
    start_temporal = time.time()
    output_from_rnn, hidden_states = model.temporal_encoder.forward_step(inputs_t)
    temporal_time = time.time() - start_temporal

    c_i = {}
    c_i['c_concat'] = output_from_rnn


    sampler = DDIMSampler(model)

    
    # Time p_sample_loop
    start_unet = time.time()
    sample_latent, _ = sampler.sample(S=8,  conditioning=c_i,  batch_size=1,  shape=[4, 48, 64],  verbose=False)
    unet_time = time.time() - start_unet
    
    # Time decode stage
    start_decode = time.time()
    sample_i = sample_latent * data_std + data_mean
    sample_i = model.decode_first_stage(sample_i)
    sample_i = sample_i.squeeze(0).clamp(-1, 1)
    decode_time = time.time() - start_decode

    sample_img = ((sample_i[:3].transpose(0,1).transpose(1,2).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)

    timing_info = {
        'temporal_encoder': temporal_time,
        'unet': unet_time,
        'decode': decode_time,
        'total': temporal_time + unet_time + decode_time
    }

    return sample_latent, sample_img, hidden_states, timing_info



if __name__ == "__main__":

    """
    Trains a model and samples it.
    """

    parser = argparse.ArgumentParser(description='Train and sample a model using a config file')
    parser.add_argument('--config', type=str, default="configs/standard_challenging_context32_nocond_all_rnn.eval.yaml",
                       help='Path to the configuration file (default: config_csllm.yaml)')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    save_path = config.save_path
    print ('='*10)
    print (save_path)
    print (args.config)
    #print (config.model.scheduler_sampling_rate)
    print ('='*10)
    #import pdb; pdb.set_trace()
    #from_autoencoder = True
    from_scratch = False
    #from_autoencoder = False # TODO: fix
    if from_scratch:
        model = init_model(config)
        #model = load_first_stage_from_config(model, './autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt')
        model = load_first_stage_from_config(model, '../autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384/model-354000.ckpt')
    else:
        #model = load_model_from_config(config, './saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_eval/model_saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_eval.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond/model-step=720000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all/model-step=308000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2/model-step=024000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn/model-step=004000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont/model-step=006000.ckpt')
        model = load_model_from_config(config, 'saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont_clusters/model-step=030000.ckpt')

    # Move model to GPU
    model = model.to(device)
    
    num_frames = 2000
    previous_frame = torch.zeros(1, 48, 64, 4).to(device)  # Move to GPU
    hidden_states = None
    
    # For normalization in sample_model function
    data_mean = -0.54
    data_std = 6.78
    data_min = -27.681446075439453
    data_max = 30.854148864746094
    
    # Lists to store timing statistics
    all_timings = []
    
    for i in range(num_frames):
        print(f"Frame {i}")
        x, y, right_click, left_click, key, stoi, itos = data_sample()
        
        start_frame = time.time()
        inputs_t = create_sample(previous_frame, hidden_states, x, y, right_click, left_click, key, stoi, itos, i)
        previous_frame, sample_img, hidden_states, timing_info = sample_model(model, config, inputs_t)
        frame_time = time.time() - start_frame
        
        timing_info['full_frame'] = frame_time
        all_timings.append(timing_info)
        
        os.makedirs('config_exp', exist_ok=True)
        Image.fromarray(sample_img).save(f'config_exp/sample_img_{i}.png')
        
        # Print timing for this frame
        print(f"Frame {i} timing (seconds):")
        print(f"  Temporal encoder: {timing_info['temporal_encoder']:.4f}")
        print(f"  UNet sampling: {timing_info['unet']:.4f}")
        print(f"  Decode stage: {timing_info['decode']:.4f}")
        print(f"  Total processing: {timing_info['total']:.4f}")
        print(f"  Full frame time: {timing_info['full_frame']:.4f}")
        print(f"  FPS: {1.0/timing_info['full_frame']:.2f}")
        
    # Calculate and print summary statistics
    timings_array = np.array([[t[k] for k in ['temporal_encoder', 'unet', 'decode', 'total', 'full_frame']] 
                             for t in all_timings])
    mean_timings = np.mean(timings_array, axis=0)
    std_timings = np.std(timings_array, axis=0)
    
    print("\nSummary Statistics:")
    print(f"Average times (seconds):")
    print(f"  Temporal encoder: {mean_timings[0]:.4f} ± {std_timings[0]:.4f}")
    print(f"  UNet sampling: {mean_timings[1]:.4f} ± {std_timings[1]:.4f}")
    print(f"  Decode stage: {mean_timings[2]:.4f} ± {std_timings[2]:.4f}")
    print(f"  Total processing: {mean_timings[3]:.4f} ± {std_timings[3]:.4f}")
    print(f"  Full frame time: {mean_timings[4]:.4f} ± {std_timings[4]:.4f}")
    print(f"Average FPS: {1.0/mean_timings[4]:.2f}")


