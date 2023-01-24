#!/usr/bin/env bash

set -ex

touch $(ls xformers/csrc/attention/cuda/fmha/kernels/*.cu)
touch $(ls xformers/csrc/attention/cuda/fmha/*.cu | grep generic)

# touch $(ls xformers/components/attention/csrc/cuda/mem_eff_attention/kernels/*.cu)
# touch $(ls xformers/components/attention/csrc/cuda/mem_eff_attention/*.cu | grep generic | grep forward)


# export CUDA_HOME="/usr/local/cuda-11.6/"
# TORCH_CUDA_ARCH_LIST="7.5" \
TORCH_CUDA_ARCH_LIST="8.0" \
FORCE_CUDA=1 \
XFORMERS_DISABLE_FLASH_ATTN=1 \
NVCC_THREADS=0 \
python3 setup.py develop

