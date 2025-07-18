from computer.util import load_cond_from_config, load_first_stage_from_config, load_model, load_model_from_config, get_ground_truths, init_model, load_autoencoder_from_ckpt, load_cond_from_ckpt
from computer.train import train_model
from computer.sample import sample_model
from data.data_processing.datasets import DataModule
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import torch
import os
import re
import argparse
#torch.set_float32_matmul_precision('highest')
torch.set_float32_matmul_precision('high')

#save_path = 'test_15_no_deltas_1000_paths'

##Parse args here

if __name__ == "__main__":

    """
    Trains a model and samples it.
    """

    parser = argparse.ArgumentParser(description='Train and sample a model using a config file')
    parser.add_argument('--config', type=str, default="config_csllm.yaml",
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
    #from_autoencoder = False
    from_autoencoder = True
    #from_autoencoder = False
    if 'eval' in args.config:
        from_autoencoder = False
    if not from_autoencoder:
        assert 'eval' in args.config
    #from_autoencoder = True # TODO: fix
    from_autoencoder = False # TODO: fix
    #from_autoencoder = True # TODO: fix
    if from_autoencoder:
        model = init_model(config)
        #model = load_first_stage_from_config(model, './autoencoder_saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_model-603000.ckpt')
        #model = load_first_stage_from_config(model, '../autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384/model-354000.ckpt')
        model = load_first_stage_from_config(model, '../autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_cont_mar15_acc1_cont_1e6_cont_2e7_cont/model-2076000.ckpt')
    else:
        #model = load_model_from_config(config, './saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_eval/model_saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_eval.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond/model-step=720000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all/model-step=308000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2/model-step=024000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn/model-step=004000.ckpt')
        #model = load_model_from_config(config, './saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont_clusters_all/model-step=014000.ckpt')
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k_maskprev0/model-step=010000.ckpt')
        #model = load_model_from_config(config, './saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384/model-step=045000.ckpt')
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered_with_desktop_1.5k/model-step=530000.ckpt')
        #model = load_model_from_config(config, './saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_debug_fixed//model-step=110000.ckpt')
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2/model-step=762000.ckpt')
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without/model-step=990000.ckpt')
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without/model-step=740000.ckpt')
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_8192_1layer/model-step=740000.ckpt')
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_8192_1layer_trim/model-step=720000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50/model-step=130000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all/model-step=038000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu/model-step=008000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu_simplifyinput/model-step=116000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu_simplifyinput_debug_pretrain/model-step=001000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu_simplifyinput_debug_fullreinit_addattn/model-step=002000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu_simplifyinput_debug_fullreinit/model-step=011000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu_simplifyinput_debug_fullreinit_reinitnone_cheat/model-step=002000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu_simplifyinput_debug_fullreinit_reinitnone/model-step=011000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80/model-step=002000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_freezernn_context8/model-step=008000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrain2_context8/model-step=040000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32/model-step=152000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_cont_cont_cont/model-step=032000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_cont_cont_cont_filtered/model-step=020000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered/model-step=096000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze/model-step=048000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_cont_cont_cont_filtered_all/model-step=004000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_cont_cont_cont_filtered/model-step=020000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_challenging/model-step=056000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging/model-step=136000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded5/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging/model-step=020000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded5/s_f_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc/model-step=020000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded5/s_f_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew/model-step=024000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata/model-step=056000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata/model-step=076000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005/model-step=108000.ckpt')
        #model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont/model-step=376000.ckpt')
        ##model = load_model_from_config(config, '/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5/model-step=432000.ckpt')
        #model = load_model_from_config(config, 'sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16/model-step=116000.ckpt')
        # Find latest checkpoint in the folder
        ckpt_folder = 'sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample_pretrainrnn'
        ckpt_folder = 'sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample_pretrainrnn'
        ckpt_folder = 'sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd'
        #ckpt_folder = save_path.rstrip('2')
        #ckpt_folder = save_path.replace('_online', '')
        if save_path.endswith('_online'):
            ckpt_folder = save_path[:-len('_online')]
        if save_path.endswith('_online_x0'):
            ckpt_folder = save_path[:-len('_online_x0')]
        if save_path.endswith('_pretrainrnn'):
            ckpt_folder = save_path[:-len('_pretrainrnn')]
        if save_path.endswith('_unfreeze'):
            ckpt_folder = save_path[:-len('_unfreeze')]
        if save_path.endswith('_online_x0_unfreeze'):
            ckpt_folder = save_path[:-len('_online_x0_unfreeze')]
        if save_path.endswith('_online_x0_noorig'):
            ckpt_folder = save_path[:-len('_online_x0_noorig')]
        if save_path.endswith('_ddpm32'):
            ckpt_folder = save_path[:-len('_ddpm32')]
        if save_path.endswith('_ddpm32_eps'):
            ckpt_folder = save_path[:-len('_ddpm32_eps')]
        if save_path.endswith('_joint_onlineonly'):
            ckpt_folder = save_path[:-len('_joint_onlineonly')]
        if save_path.endswith('_joint_onlineonly_eps'):
            ckpt_folder = save_path[:-len('_joint_onlineonly_eps')]
        if save_path.endswith('2'):
            ckpt_folder = save_path[:-len('2')]
        if save_path.endswith('7'):
            ckpt_folder = save_path[:-len('7')]

        #ckpt_folder = 'sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2'
        ckpt_files = [f for f in os.listdir(ckpt_folder) if f.startswith('model-step=') and f.endswith('.ckpt')]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_folder}")
        step_pattern = re.compile(r'model-step=(\d+)\.ckpt')
        ckpt_steps = [(f, int(step_pattern.search(f).group(1))) for f in ckpt_files if step_pattern.search(f)]
        ckpt_steps.sort(key=lambda x: x[1])
        latest_ckpt_file, latest_step = ckpt_steps[-1]
        latest_ckpt_path = os.path.join(ckpt_folder, latest_ckpt_file)
        print(f"Found latest checkpoint: {latest_ckpt_path} (step {latest_step})")
        model = load_model_from_config(config, latest_ckpt_path)
        #### REPLACEMENT_LINE
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_standard_filtered/model-step=010000.ckpt')
        pass
        #model = load_model_from_config(config, 'saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_ddd_difficult_only_withlstmencoder_without_minmax/model-step=740000.ckpt')
    # model = load_cond_from_config(model, "model_bert.ckpt")

    #model = load_model_from_config(config, 'oct29_fixcursor_test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt')
    #model = load_model_from_config(config, 'saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256/model-step=010500.ckpt')
    #model = load_model_from_config(config, 'saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256_cont/model-step=003500.ckpt')
    #model = load_model_from_config(config, 'oct29_fixcursor_test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt')
    #model = load_model_from_config(config, 'saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256/model_saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256.ckpt')
    #import pdb; pdb.set_trace()

    #model = load_model_from_config(config, 'test_12_600_epoch_no_deltas/model_test_12_600_epoch_no_deltas.ckpt')
    #model = init_model(config) #initializes the all model modules.

    #model = load_autoencoder_from_ckpt(model, 'autoencoder/train_0/model_ae_epoch=00.ckpt') #loads autoencoder weights.
    #model = load_cond_from_ckpt(model, 'model_bert.ckpt') #loads encoder weights.
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model = model.to(device)
    data: DataModule = instantiate_from_config(config.data)
    data.setup()

    print("---------------------------------"); print("\u2705 Model loaded with ae and cond."); print("---------------------------------")

    

    # for name, child in model.model.diffusion_model.named_children():
    #     print(name)
    #     if name == 'input_blocks': print(child)

    #prompts, image_sequences, targets = get_ground_truths(data.datasets['train'], idxs=[i for i in range(173)])

    os.makedirs(save_path, exist_ok=True)

    # sample_model(model, prompts, image_sequences, save_path, create_video=True)

    model = train_model(model, data, save_path, config)
    #model = model.to(device)

    #sample_model(model, prompts, image_sequences, save_path, True)
    #prompts, image_sequences, targets = get_ground_truths(data.datasets['train'], idxs=[i for i in range(173)])
