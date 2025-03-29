



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_cont.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont 2>&1 &



stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.bsz40 2>&1 &
stdbuf -oL -eL python main.py --config configs/final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered_largeimg.yaml > log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.bsz50 2>&1 &
