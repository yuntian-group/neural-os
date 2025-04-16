



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_cont.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont 2>&1 &



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.cont 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.bsz40 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.bsz50 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64_gpu8_filtered.largeimg.lr4e5 2>&1 &



CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_gpu1.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu1 2>&1 &
CUDA_VISIBLE_DEVICES=1,2 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_gpu2.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu2 2>&1 &
CUDA_VISIBLE_DEVICES=3,4,5,6 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_gpu4.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu4 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.cont2 2>&1 &



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.cont3 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest 2>&1 &




CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context4.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context4 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8 2>&1 &



CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_freezernn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.freezernn 2>&1 &
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_freezernn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.freezernn.context8 2>&1 &



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.contfreezernn 2>&1 &



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput 2>&1 &

stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.rerun 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.fixed 2>&1 &




CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.fixed.cont 2>&1 &


CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_addattn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.addattn.fixed 2>&1 &
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_reinitrnn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitrnn.fixed 2>&1 &
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_reinitcnn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitcnn.fixed 2>&1 &
CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_reinitnone.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitnone.fixed.cont 2>&1 &

CUDA_VISIBLE_DEVICES=5 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_reinitnone_overfit.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitnone.fixed.overfit 2>&1 &

CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_reinitnone_highprecision.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitnone.highprecision.fixed 2>&1 &


CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_addattn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.addattn.fixed.cont 2>&1 &


CUDA_VISIBLE_DEVICES=7 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_reinitnone_cheat.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitnone_cheat.fixed.cont 2>&1 &


CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_fullreinit_noimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.fixed.noimg 2>&1 &


CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain 2>&1 &
CUDA_VISIBLE_DEVICES=5 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain_posttrain.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain.posttrain 2>&1 &


CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain2.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2 2>&1 &
CUDA_VISIBLE_DEVICES=7 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain3.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain3 2>&1 &
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain4.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain4 2>&1 &
CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain5.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain5 2>&1 &


CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain21.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain21 2>&1 &
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain22.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain22 2>&1 &

CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain3_posttrain.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain3.posttrain 2>&1 &
CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrainnone_posttrain.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrainnone.posttrain 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain228 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain228.lr6.4e4 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.lr3e4 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.lr6.4e4 2>&1 &

stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting2 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228_nopretrain_posttrain.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting2_nopretrain_posttrain 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_context8_all_debug_pretrain228_real_contdebug.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting2.real.contdebug 2>&1 &

stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrain2_context8.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8 2>&1 &



1. finetune part 1 on real self.pretrain but not self.pretrain2, not self.pretrain3, real
CUDA_VISIBLE_DEVICES=0,1 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrain2_context8_finetunerealpart1.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.finetunerealpart1 2>&1 &


2. finetune part 2 but on real: self.pretrain2
CUDA_VISIBLE_DEVICES=2,3 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrain2_context8_finetunerealpart2.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.finetunerealpart2 2>&1 &

self.pretrain, not self.pretrain2,  self.pretrain3
3. finetune part 1 on real, but also part 2 on synt
CUDA_VISIBLE_DEVICES=4,5 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrain2_context8_finetunerealpart1synpart2.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.finetunerealpart1synpart2 2>&1 &

4. pretrain part 1 and part 2
CUDA_VISIBLE_DEVICES=6,7 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrain2_context8_pretrainrealpart1synpart2.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.pretrainrealpart1synpart2 2>&1 &



CUDA_VISIBLE_DEVICES=2,3 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrain2_context8_posttrain_orig.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.posttrain_orig 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrain2_context8_posttrain_finetunerealpart1.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.posttrain_finetunerealpart1 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_2Xdata_4Xb.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.2Xdata.4Xb 2>&1 &

stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_3Xdata_4Xb.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.3Xdata.4Xb 2>&1 &

#stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_6Xb.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.6Xb 2>&1 &

stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb 2>&1 &




# try training diffusion, freeze rnn
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn 2>&1 &

stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.cont.cont 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.cont.cont.cont 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_filtered.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.cont.cont.cont.filtered 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.cont.cont.cont.filtered.all 2>&1 &



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_unfreeze.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze 2>&1 &


# eval old ckpt
(py311) root@shy-tree-rests-ice-01:~/computer/computer# ls -lh train_dataset_encoded/saved_final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg_cont4_lr8e5_b50_context8_b80_all_fixrelu_simplifyinput
total 26G
-rw------- 1 root root 26G Apr  6 16:16 'model-step=116000.ckpt'


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_unfreeze.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze 2>&1 &



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_unfreeze_filtered.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze.filtered 2>&1 &


# freezernn, but on all data
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_contall_challenging.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.contall.challenging 2>&1 &



# freezernn, but on challenging data only
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_challenging.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.challenging 2>&1 &


stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_pretrainreal_context32_cont_4Xdata_4Xb_diffusion_freezernn_unfreeze_afterchallenging.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging 2>&1 &
