



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
