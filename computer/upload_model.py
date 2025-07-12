import torch
from huggingface_hub import HfApi, create_repo
import os
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config
import json

# Set your Hugging Face token here
# Set your model details
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-432k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-54k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-challenging-42k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-lr5e6-40k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-lr8e5-8k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-pretrainrnn-48k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-pretrainrnn-challenging-46k"



MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-challenging-166k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-pretrainrnn-challenging-184k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-pretrainrnn-challenging-balanced-120k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-pretrainrnn-challenging-balanced-lr5e6-124k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-pretrainrnn-challenging-balanced-lr1.25e6-110k"


MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-370k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-368k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-unfreezernn-160k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-unfreezernn-origunet-nospatial-144k"


MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-674k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-unfreezernn-198k"


MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-online-74k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-online-70k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-online-x0-46k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-338k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-ddpm32-eps-144k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-ddpm32-x0-142k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-70k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-22-38k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-joint-onlineonly-eps22-40k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222-42k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-2222-70k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222-48k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-06k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-136k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-184k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-224k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-272k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-oo-eps222222k7-270k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-oo-eps222222k72-270k" # wrong name, should be 050k
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k72-108k"

LOCAL_CHECKPOINT_PATH = "test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "oct27_test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "checkpoints/model-step=007500.ckpt"
LOCAL_CHECKPOINT_PATH = "test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "test_15_no_deltas_1000_paths/model_test_15_no_deltas_1000_paths.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5/model_saved_fixcursor_lr2e5.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug/model-step=004000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe/model-step=002000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe/model-step=005000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_identity/model_saved_fixcursor_lr2e5_debug_gpt_firstframe_identity.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model-step=001000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model-step=001000-v1.ckpt"
#LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model-step=005000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap/model_saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity/model-step=009000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_identity/model-step=005000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_debugidentity_256_cont/model-step=003000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_fixcursor_lr2e5_debug_gpt_firstframe_posmap_longtrainh200/model-step=007500.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5/model-step=012500.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick/model-step=246000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos/model-step=043000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2/model-step=393000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_debug_fixed/model-step=100000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_debug_fixed//model-step=110000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2/model-step=762000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384_cont2_debug_fixed//model-step=110000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_standard_challenging_context32_nocond_cont_cont_all_cont/model-step=040000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_standard_challenging_context32_nocond_fixnorm/model-step=002000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_standard_challenging_context32_nocond_fixnorm_all/model-step=300000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2/model-step=018000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz/model-step=030000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1/model-step=020000.ckpt"
LOCAL_CHECKPOINT_PATH = "saved_standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_0.2_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont_clusters_all_realall/model-step=072000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging/model-step=048000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging/model-step=072000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging/model-step=096000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging/model-step=136000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded2/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging/model-step=136000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata/model-step=556000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb/model-step=056000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005/model-step=104000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb/model-step=064000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont/model-step=048000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb/model-step=064000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont/model-step=160000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont/model-step=236000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont/model-step=372000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5/model-step=256000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5/model-step=384000.ckpt"
LOCAL_CHECKPOINT_PATH = "/root/computer/computer/train_dataset_encoded6/sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5/model-step=432000.ckpt"
LOCAL_CHECKPOINT_PATH = "sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema/model-step=010000.ckpt"
LOCAL_CHECKPOINT_PATH = "sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample/model-step=042000.ckpt"
LOCAL_CHECKPOINT_PATH = "sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample/model-step=042000.ckpt"
LOCAL_CHECKPOINT_PATH = "sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_lr8e5/model-step=008000.ckpt"
LOCAL_CHECKPOINT_PATH = "sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_pretrainrnn/model-step=048000.ckpt"
LOCAL_CHECKPOINT_PATH = "sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample_pretrainrnn/model-step=046000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample/model-step=166000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample_pretrainrnn/model-step=184000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced/model-step=120000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_diffusion_freezernn_contfiltered_unfreeze_afterchallenging_newdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6/model-step=124000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_context64_b16_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr1.25e6/model-step=110000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd2/model-step=370000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2/model-step=368000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contunfreezernn_newnewd2/model-step=160000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contunfreezernn_newnewd_origunet_nospatial2/model-step=144000.ckpt"


LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2/model-step=674000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contunfreezernn_newnewd2/model-step=198000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online/model-step=074000.ckpt"

LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online/model-step=070000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0/model-step=338000.ckpt"

LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_ddpm32_eps/model-step=144000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_ddpm32/model-step=142000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly/model-step=070000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly22/model-step=038000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly_eps22/model-step=040000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly222/model-step=042000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly2222/model-step=070000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly222222/model-step=048000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly222222k7/model-step=006000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly222222k7/model-step=136000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly222222k7/model-step=224000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly222222k7/model-step=272000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly_eps222222k7/model-step=270000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly_eps222222k72/model-step=050000.ckpt"
LOCAL_CHECKPOINT_PATH = "./sb_computecanada_fsdp_noema_challengingandsample_pretrainrnn_balanced_lr5e6_contbest_samplercover_newd_contfreezernn_newnewd_origunet_nospatial2_ONLINE_online_online_x0_joint_onlineonly222222k72/model-step=108000.ckpt"
CONFIG_PATH = "config_csllm.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe_identity.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe_posmap_debugidentity.yaml"
CONFIG_PATH = "configs/2e5_debug_gpt_firstframe_posmap_longtrainh200.yaml"
CONFIG_PATH = "configs/pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos.yaml"
CONFIG_PATH = "configs/pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384.yaml"
CONFIG_PATH = "configs/standard_challenging_context32_nocond_all.yaml"
CONFIG_PATH = "configs/standard_challenging_context32_nocond_all_rnn.eval.yaml"

def upload_model_to_hub():
    # Load the configuration
    config = OmegaConf.load(CONFIG_PATH)
    
    # Load the local checkpoint
    #checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    # Extract only the state_dict
    #state_dict = checkpoint['state_dict']
    state_dict = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    
    # Create or get the repo
    api = HfApi()
    repo_url = create_repo(MODEL_NAME, private=True)
    
    # Save the state_dict to a file
    torch.save(state_dict, "model.safetensors")
    
    # Convert OmegaConf to a regular dict and save as JSON
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open("config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Create a README file
    with open("README.md", "w") as f:
        f.write(f"# {MODEL_NAME}\n\nThis is a LatentDiffusion model for frame prediction.")
    
    # Push the files to the hub
    api.upload_file(
        path_or_fileobj="model.safetensors",
        path_in_repo="model.safetensors",
        repo_id=MODEL_NAME,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="config.json",
        path_in_repo="config.json",
        repo_id=MODEL_NAME,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=MODEL_NAME,
        repo_type="model",
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{MODEL_NAME}")
    
    # Clean up the temporary files
    os.remove("model.safetensors")
    os.remove("config.json")
    os.remove("README.md")

if __name__ == "__main__":
    upload_model_to_hub()
