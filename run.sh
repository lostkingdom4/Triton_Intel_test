TORCH_COMPILE_DEBUG=1 \
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=Triton \
TORCHINDUCTOR_MAX_AUTOTUNE=1 \
TORCHINDUCTOR_AUTOTUNE_FALLBACK_TO_ATEN=0 \
TRITON_PRINT_AUTOTUNING=1 \
/home/shukai/anaconda3/envs/py310/bin/python /home/shukai/torch_trition_test/carc_triton_test.py