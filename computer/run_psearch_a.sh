#!/bin/bash

CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320.yaml > log.a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320 2>&1 &
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/psearch_a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192.yaml > log.a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192 2>&1 &
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320.yaml > log.a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320 2>&1 &
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_cm1_2_mc320.yaml > log.a_hs4096_oc32_nl48_ar2_cm1_2_mc320 2>&1 &
CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320.yaml > log.a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320 2>&1 &
CUDA_VISIBLE_DEVICES=5 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_3_mc320.yaml > log.a_hs4096_oc32_nl48_ar_cm1_2_3_mc320 2>&1 &
CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc320.yaml > log.a_hs4096_oc32_nl48_ar_cm1_2_mc320 2>&1 &
CUDA_VISIBLE_DEVICES=7 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc384.yaml > log.a_hs4096_oc32_nl48_ar_cm1_2_mc384 2>&1 &

CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc192.yaml > log.a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc192 2>&1 &
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc448.yaml > log.a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc448 2>&1 &



CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc448.yaml > log.a_hs4096_oc32_nl48_ar_cm1_2_mc448 2>&1 &



CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64.yaml > log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64 2>&1 &
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr4e5_b64.yaml > log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr4e5_b64 2>&1 &
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr1.6e4_b64.yaml > log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr1.6e4_b64 2>&1 &
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b128.yaml > log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b128 2>&1 &

CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64 2>&1 &
CUDA_VISIBLE_DEVICES=5 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr4e5_b64.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr4e5_b64 2>&1 &
CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr1.6e4_b64.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr1.6e4_b64 2>&1 &
CUDA_VISIBLE_DEVICES=7 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b100.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b100 2>&1 &













CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu1.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu1 2>&1 &
CUDA_VISIBLE_DEVICES=1,2 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu2.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu2 2>&1 &
CUDA_VISIBLE_DEVICES=3,4,5,6 stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu4.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu4 2>&1 &



stdbuf -oL -eL python main.py --config configs/psearch_a_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8.yaml > log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8 2>&1 &
