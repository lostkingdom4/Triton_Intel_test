
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()
_frozen_param0 = None  # device(type='xpu', index=0) torch.float32 (30522, 768) (768, 1) 7f60a21a5e90
_frozen_param3 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c72bdd0
_frozen_param4 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c72bd30
_frozen_param10 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a217a750
_frozen_param12 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a219d760
_frozen_param13 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a219ec50
_frozen_param14 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a219f790
_frozen_param16 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21a5fd0
_frozen_param18 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a219eb10
_frozen_param19 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad300
_frozen_param20 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad1c0
_frozen_param26 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad4e0
_frozen_param28 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad580
_frozen_param29 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad620
_frozen_param30 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad5d0
_frozen_param32 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21ad3f0
_frozen_param34 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad2b0
_frozen_param35 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad800
_frozen_param36 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad760
_frozen_param42 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad9e0
_frozen_param44 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ada80
_frozen_param45 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21adb20
_frozen_param46 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21adad0
_frozen_param48 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21ad8f0
_frozen_param50 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ad7b0
_frozen_param51 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21add00
_frozen_param52 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21adc60
_frozen_param58 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21adee0
_frozen_param60 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21adf80
_frozen_param61 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae020
_frozen_param62 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21adfd0
_frozen_param64 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21addf0
_frozen_param66 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21adcb0
_frozen_param67 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae200
_frozen_param68 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae160
_frozen_param74 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae3e0
_frozen_param76 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae480
_frozen_param77 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae520
_frozen_param78 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae4d0
_frozen_param80 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21ae2f0
_frozen_param82 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae1b0
_frozen_param83 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae700
_frozen_param84 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae660
_frozen_param90 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae8e0
_frozen_param92 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae980
_frozen_param93 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aea20
_frozen_param94 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae9d0
_frozen_param96 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21ae7f0
_frozen_param98 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21ae6b0
_frozen_param99 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aec00
_frozen_param100 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aeb60
_frozen_param106 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aede0
_frozen_param108 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aee80
_frozen_param109 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aef20
_frozen_param110 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aeed0
_frozen_param112 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21aecf0
_frozen_param114 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aebb0
_frozen_param115 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af100
_frozen_param116 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af060
_frozen_param122 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af2e0
_frozen_param124 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af380
_frozen_param125 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af420
_frozen_param126 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af3d0
_frozen_param128 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21af1f0
_frozen_param130 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af0b0
_frozen_param131 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af600
_frozen_param132 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af560
_frozen_param138 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af7e0
_frozen_param140 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af880
_frozen_param141 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af920
_frozen_param142 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af8d0
_frozen_param144 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21af6f0
_frozen_param146 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21af5b0
_frozen_param147 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21afb00
_frozen_param148 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21afa60
_frozen_param154 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21afce0
_frozen_param156 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21afd80
_frozen_param157 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21afe20
_frozen_param158 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21afdd0
_frozen_param160 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f60a21afbf0
_frozen_param162 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21afab0
_frozen_param163 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8040
_frozen_param164 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f60a21aff60
_frozen_param170 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8220
_frozen_param172 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c82c0
_frozen_param173 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8360
_frozen_param174 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8310
_frozen_param176 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f609c6c8450
_frozen_param178 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8130
_frozen_param179 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8540
_frozen_param180 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c84a0
_frozen_param186 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8720
_frozen_param188 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c87c0
_frozen_param189 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8860
_frozen_param190 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8810
_frozen_param192 = None  # device(type='xpu', index=0) torch.float32 (3072,) (1,) 7f609c6c8950
_frozen_param194 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c84f0
_frozen_param195 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c8a40
_frozen_param196 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f609c6c89a0
_frozen_param198 = None  # device(type='xpu', index=0) torch.float32 (768,) (1,) 7f61dd3339c0
_frozen_param201 = None  # device(type='xpu', index=0) torch.float32 (1, 512, 768) (393216, 768, 1) 7f61d3104270
_frozen_param202 = None  # device(type='xpu', index=0) torch.float32 (1, 512, 768) (393216, 768, 1) 7f61d3149b70
_frozen_param203 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d8b33100
_frozen_param204 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d31c9210
_frozen_param205 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d308c900
_frozen_param206 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d308fab0
_frozen_param207 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d308ffb0
_frozen_param208 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d308df30
_frozen_param209 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d8b33dd0
_frozen_param210 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d3012890
_frozen_param211 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d8c34680
_frozen_param212 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d31ae660
_frozen_param213 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d31aeac0
_frozen_param214 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d308dad0
_frozen_param215 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d3012430
_frozen_param216 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c4630
_frozen_param217 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c4400
_frozen_param218 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c4ef0
_frozen_param219 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c42c0
_frozen_param220 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6520
_frozen_param221 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c51c0
_frozen_param222 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c62a0
_frozen_param223 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c4cc0
_frozen_param224 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6bb0
_frozen_param225 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c6b10
_frozen_param226 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c6c00
_frozen_param227 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6c50
_frozen_param228 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d87ab560
_frozen_param229 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c6cf0
_frozen_param230 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c6ca0
_frozen_param231 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6d40
_frozen_param232 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6d90
_frozen_param233 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c6de0
_frozen_param234 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c6e30
_frozen_param235 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6e80
_frozen_param236 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6ed0
_frozen_param237 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c6f20
_frozen_param238 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c6f70
_frozen_param239 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c6fc0
_frozen_param240 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c7010
_frozen_param241 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c7060
_frozen_param242 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c70b0
_frozen_param243 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c7100
_frozen_param244 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c7150
_frozen_param245 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c71a0
_frozen_param246 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c71f0
_frozen_param247 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c7240
_frozen_param248 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c7290
_frozen_param249 = None  # device(type='xpu', index=0) torch.float32 (768, 3072) (1, 768) 7f61d30c72e0
_frozen_param250 = None  # device(type='xpu', index=0) torch.float32 (3072, 768) (1, 3072) 7f61d30c7330
_frozen_param251 = None  # device(type='xpu', index=0) torch.float32 (768, 768) (1, 768) 7f61d30c73d0


# kernel path: /tmp/torchinductor_root/4g/c4gmhbikkqxjcjymeyu7cuokoj563v5noxjvrmzo37j25cpqsbur.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
# inputs_embeds => embedding
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.triton_heuristics import grid, split_scan_grid

@triton_heuristics.persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '9710e3cf217aec8cc88e960f6514e3b610ce8beb47bd134c8ae0cebe0c49264f', 'kernel_num_gb': 0.006301696}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-12
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)


def get_args():
    arg_0 = rand_strided((1, 512), (512, 1), device='xpu:0', dtype=torch.int64)
    arg_1 = rand_strided((30522, 768), (768, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 512, 768), (393216, 768, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((768,), (1,), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((1, 512, 768), (393216, 768, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(*args, 512, 768, grid=grid(512), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused_add_embedding_native_layer_norm_0.benchmark_all_configs(*args, 512, 768, grid=grid(512))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.006301696
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='xpu')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _xpu_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/xh/cxhkg27tnht44uwicaioahdqp2su4cg2as6ykg2k24jixaldrk2f.py
# Source Nodes: [attn_output], Original ATen: [aten.view]
# attn_output => full_default
triton_poi_fused_view_1 = async_compile.triton('triton_poi_fused_view_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '9710e3cf217aec8cc88e960f6514e3b610ce8beb47bd134c8ae0cebe0c49264f', 'kernel_num_gb': 0.012582912},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.001953125
    tl.store(out_ptr0 + (x0), tmp0, None)


def get_args():
    arg_0 = rand_strided((12, 512, 512), (262144, 512, 1), device='xpu:0', dtype=torch.float32)
    return arg_0,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_1.run(*args, 3145728, grid=grid(3145728), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_view_1.benchmark_all_configs(*args, 3145728, grid=grid(3145728))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.012582912
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='xpu')


# kernel path: /tmp/torchinductor_root/qf/cqfbrtdefa2s6ututwjhrsj2fqbxyv2hssingq45yw7u2mw4j23g.py
# Source Nodes: [attn_output_2], Original ATen: [aten.clone]
# attn_output_2 => clone_1
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '9710e3cf217aec8cc88e960f6514e3b610ce8beb47bd134c8ae0cebe0c49264f', 'kernel_num_gb': 0.003145728},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (32768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)


def get_args():
    arg_0 = rand_strided((12, 512, 64), (32768, 64, 1), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 512, 12, 64), (393216, 768, 64, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(*args, 393216, grid=grid(393216), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_clone_2.benchmark_all_configs(*args, 393216, grid=grid(393216))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.003145728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='xpu')


# kernel path: /tmp/torchinductor_root/5s/c5s26rzikabpsfx4eq35jjojp6i7trsbktmlguiddtgzf6mmi6pz.py
# Source Nodes: [add_1, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_4
# attention_output => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_3, var_mean_1
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_per_fused_add_native_layer_norm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.triton_heuristics import grid, split_scan_grid

@triton_heuristics.persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '9710e3cf217aec8cc88e960f6514e3b610ce8beb47bd134c8ae0cebe0c49264f', 'kernel_num_gb': 0.004724736}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)


def get_args():
    arg_0 = rand_strided((512, 768), (768, 1), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((768,), (1,), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_3.run(*args, 512, 768, grid=grid(512), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused_add_native_layer_norm_3.benchmark_all_configs(*args, 512, 768, grid=grid(512))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.004724736
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='xpu')


# kernel path: /tmp/torchinductor_root/li/clitbgrk26cxmmsmcaws6wlj2d7ugqivlgeaggkyxhnhne2wwkqc.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_7, erf, mul_4, mul_5, mul_6
triton_poi_fused_gelu_4 = async_compile.triton('triton_poi_fused_gelu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '9710e3cf217aec8cc88e960f6514e3b610ce8beb47bd134c8ae0cebe0c49264f', 'kernel_num_gb': 0.012582912},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_4(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)


def get_args():
    arg_0 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='xpu:0', dtype=torch.float32)
    return arg_0,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_4.run(*args, 1572864, grid=grid(1572864), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_gelu_4.benchmark_all_configs(*args, 1572864, grid=grid(1572864))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.012582912
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='xpu')


# kernel path: /tmp/torchinductor_root/2s/c2sjjk7dyhm2o33ddpmjcjdfksn7mkquzx2mbw7zufhowmp5bm3g.py
# Source Nodes: [pooled_output_2], Original ATen: [aten.tanh]
# pooled_output_2 => tanh
triton_poi_fused_tanh_5 = async_compile.triton('triton_poi_fused_tanh_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '9710e3cf217aec8cc88e960f6514e3b610ce8beb47bd134c8ae0cebe0c49264f', 'kernel_num_gb': 6.144e-06},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = libdevice.tanh(tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, xmask)


def get_args():
    arg_0 = rand_strided((1, 768), (768, 1), device='xpu:0', dtype=torch.float32)
    return arg_0,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_5.run(*args, 768, grid=grid(768), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_tanh_5.benchmark_all_configs(*args, 768, grid=grid(768))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 6.144e-06
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='xpu')


async_compile.wait(globals())
del async_compile

def call(args):
    arg201_1, = args
    args.clear()
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        buf3 = empty_strided((1, 512, 768), (393216, 768, 1), device='xpu', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg201_1, _frozen_param0, _frozen_param201, _frozen_param202, _frozen_param3, _frozen_param4, buf3, 512, 768, grid=grid(512), stream=stream0)
        del arg201_1
        buf4 = empty_strided((512, 768), (768, 1), device='xpu', dtype=torch.float32)
        # Source Nodes: [l__self___encoder_layer_0_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param10, reinterpret_tensor(buf3, (512, 768), (768, 1), 0), _frozen_param203, alpha=1, beta=1, out=buf4)
        buf5 = empty_strided((12, 512, 512), (262144, 512, 1), device='xpu', dtype=torch.float32)
        # Source Nodes: [attn_output], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf5, 3145728, grid=grid(3145728), stream=stream0)
        buf6 = empty_strided((12, 512, 64), (32768, 64, 1), device='xpu', dtype=torch.float32)
        # Source Nodes: [attn_output], Original ATen: [aten.bmm, aten.view]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf4, (12, 512, 64), (64, 768, 1), 0), out=buf6)
        buf7 = reinterpret_tensor(buf4, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [attn_output_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf6, buf7, 393216, grid=grid(393216), stream=stream0)
        buf8 = reinterpret_tensor(buf6, (512, 768), (768, 1), 0); del buf6  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param12, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), _frozen_param204, alpha=1, beta=1, out=buf8)
        buf12 = reinterpret_tensor(buf7, (1, 512, 768), (393216, 768, 1), 0); del buf7  # reuse
        # Source Nodes: [add_1, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf8, buf3, _frozen_param13, _frozen_param14, buf12, 512, 768, grid=grid(512), stream=stream0)
        buf13 = empty_strided((512, 3072), (3072, 1), device='xpu', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param16, reinterpret_tensor(buf12, (512, 768), (768, 1), 0), _frozen_param205, alpha=1, beta=1, out=buf13)
        buf14 = reinterpret_tensor(buf13, (1, 512, 3072), (1572864, 3072, 1), 0); del buf13  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf14, 1572864, grid=grid(1572864), stream=stream0)
        buf15 = buf8; del buf8  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param18, reinterpret_tensor(buf14, (512, 3072), (3072, 1), 0), _frozen_param206, alpha=1, beta=1, out=buf15)
        buf19 = buf3; del buf3  # reuse
        # Source Nodes: [add_2, current_states_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf15, buf12, _frozen_param19, _frozen_param20, buf19, 512, 768, grid=grid(512), stream=stream0)
        buf20 = buf15; del buf15  # reuse
        # Source Nodes: [l__self___encoder_layer_1_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param26, reinterpret_tensor(buf19, (512, 768), (768, 1), 0), _frozen_param207, alpha=1, beta=1, out=buf20)
        buf21 = reinterpret_tensor(buf12, (12, 512, 64), (32768, 64, 1), 0); del buf12  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf20, (12, 512, 64), (64, 768, 1), 0), out=buf21)
        buf22 = reinterpret_tensor(buf20, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf21, buf22, 393216, grid=grid(393216), stream=stream0)
        buf23 = reinterpret_tensor(buf21, (512, 768), (768, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param28, reinterpret_tensor(buf22, (512, 768), (768, 1), 0), _frozen_param208, alpha=1, beta=1, out=buf23)
        buf27 = reinterpret_tensor(buf22, (1, 512, 768), (393216, 768, 1), 0); del buf22  # reuse
        # Source Nodes: [add_3, attention_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf23, buf19, _frozen_param29, _frozen_param30, buf27, 512, 768, grid=grid(512), stream=stream0)
        buf28 = reinterpret_tensor(buf14, (512, 3072), (3072, 1), 0); del buf14  # reuse
        # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param32, reinterpret_tensor(buf27, (512, 768), (768, 1), 0), _frozen_param209, alpha=1, beta=1, out=buf28)
        buf29 = reinterpret_tensor(buf28, (1, 512, 3072), (1572864, 3072, 1), 0); del buf28  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf29, 1572864, grid=grid(1572864), stream=stream0)
        buf30 = buf23; del buf23  # reuse
        # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param34, reinterpret_tensor(buf29, (512, 3072), (3072, 1), 0), _frozen_param210, alpha=1, beta=1, out=buf30)
        buf34 = buf19; del buf19  # reuse
        # Source Nodes: [add_4, current_states_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf30, buf27, _frozen_param35, _frozen_param36, buf34, 512, 768, grid=grid(512), stream=stream0)
        buf35 = buf30; del buf30  # reuse
        # Source Nodes: [l__self___encoder_layer_2_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param42, reinterpret_tensor(buf34, (512, 768), (768, 1), 0), _frozen_param211, alpha=1, beta=1, out=buf35)
        buf36 = reinterpret_tensor(buf27, (12, 512, 64), (32768, 64, 1), 0); del buf27  # reuse
        # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf35, (12, 512, 64), (64, 768, 1), 0), out=buf36)
        buf37 = reinterpret_tensor(buf35, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf36, buf37, 393216, grid=grid(393216), stream=stream0)
        buf38 = reinterpret_tensor(buf36, (512, 768), (768, 1), 0); del buf36  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param44, reinterpret_tensor(buf37, (512, 768), (768, 1), 0), _frozen_param212, alpha=1, beta=1, out=buf38)
        buf42 = reinterpret_tensor(buf37, (1, 512, 768), (393216, 768, 1), 0); del buf37  # reuse
        # Source Nodes: [add_5, attention_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf38, buf34, _frozen_param45, _frozen_param46, buf42, 512, 768, grid=grid(512), stream=stream0)
        buf43 = reinterpret_tensor(buf29, (512, 3072), (3072, 1), 0); del buf29  # reuse
        # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param48, reinterpret_tensor(buf42, (512, 768), (768, 1), 0), _frozen_param213, alpha=1, beta=1, out=buf43)
        buf44 = reinterpret_tensor(buf43, (1, 512, 3072), (1572864, 3072, 1), 0); del buf43  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf44, 1572864, grid=grid(1572864), stream=stream0)
        buf45 = buf38; del buf38  # reuse
        # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param50, reinterpret_tensor(buf44, (512, 3072), (3072, 1), 0), _frozen_param214, alpha=1, beta=1, out=buf45)
        buf49 = buf34; del buf34  # reuse
        # Source Nodes: [add_6, current_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf45, buf42, _frozen_param51, _frozen_param52, buf49, 512, 768, grid=grid(512), stream=stream0)
        buf50 = buf45; del buf45  # reuse
        # Source Nodes: [l__self___encoder_layer_3_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param58, reinterpret_tensor(buf49, (512, 768), (768, 1), 0), _frozen_param215, alpha=1, beta=1, out=buf50)
        buf51 = reinterpret_tensor(buf42, (12, 512, 64), (32768, 64, 1), 0); del buf42  # reuse
        # Source Nodes: [attn_output_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf50, (12, 512, 64), (64, 768, 1), 0), out=buf51)
        buf52 = reinterpret_tensor(buf50, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf50  # reuse
        # Source Nodes: [attn_output_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf51, buf52, 393216, grid=grid(393216), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (512, 768), (768, 1), 0); del buf51  # reuse
        # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param60, reinterpret_tensor(buf52, (512, 768), (768, 1), 0), _frozen_param216, alpha=1, beta=1, out=buf53)
        buf57 = reinterpret_tensor(buf52, (1, 512, 768), (393216, 768, 1), 0); del buf52  # reuse
        # Source Nodes: [add_7, attention_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf53, buf49, _frozen_param61, _frozen_param62, buf57, 512, 768, grid=grid(512), stream=stream0)
        buf58 = reinterpret_tensor(buf44, (512, 3072), (3072, 1), 0); del buf44  # reuse
        # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param64, reinterpret_tensor(buf57, (512, 768), (768, 1), 0), _frozen_param217, alpha=1, beta=1, out=buf58)
        buf59 = reinterpret_tensor(buf58, (1, 512, 3072), (1572864, 3072, 1), 0); del buf58  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf59, 1572864, grid=grid(1572864), stream=stream0)
        buf60 = buf53; del buf53  # reuse
        # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param66, reinterpret_tensor(buf59, (512, 3072), (3072, 1), 0), _frozen_param218, alpha=1, beta=1, out=buf60)
        buf64 = buf49; del buf49  # reuse
        # Source Nodes: [add_8, current_states_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf60, buf57, _frozen_param67, _frozen_param68, buf64, 512, 768, grid=grid(512), stream=stream0)
        buf65 = buf60; del buf60  # reuse
        # Source Nodes: [l__self___encoder_layer_4_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param74, reinterpret_tensor(buf64, (512, 768), (768, 1), 0), _frozen_param219, alpha=1, beta=1, out=buf65)
        buf66 = reinterpret_tensor(buf57, (12, 512, 64), (32768, 64, 1), 0); del buf57  # reuse
        # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf65, (12, 512, 64), (64, 768, 1), 0), out=buf66)
        buf67 = reinterpret_tensor(buf65, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf65  # reuse
        # Source Nodes: [attn_output_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf66, buf67, 393216, grid=grid(393216), stream=stream0)
        buf68 = reinterpret_tensor(buf66, (512, 768), (768, 1), 0); del buf66  # reuse
        # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param76, reinterpret_tensor(buf67, (512, 768), (768, 1), 0), _frozen_param220, alpha=1, beta=1, out=buf68)
        buf72 = reinterpret_tensor(buf67, (1, 512, 768), (393216, 768, 1), 0); del buf67  # reuse
        # Source Nodes: [add_9, attention_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf68, buf64, _frozen_param77, _frozen_param78, buf72, 512, 768, grid=grid(512), stream=stream0)
        buf73 = reinterpret_tensor(buf59, (512, 3072), (3072, 1), 0); del buf59  # reuse
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param80, reinterpret_tensor(buf72, (512, 768), (768, 1), 0), _frozen_param221, alpha=1, beta=1, out=buf73)
        buf74 = reinterpret_tensor(buf73, (1, 512, 3072), (1572864, 3072, 1), 0); del buf73  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf74, 1572864, grid=grid(1572864), stream=stream0)
        buf75 = buf68; del buf68  # reuse
        # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param82, reinterpret_tensor(buf74, (512, 3072), (3072, 1), 0), _frozen_param222, alpha=1, beta=1, out=buf75)
        buf79 = buf64; del buf64  # reuse
        # Source Nodes: [add_10, current_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf75, buf72, _frozen_param83, _frozen_param84, buf79, 512, 768, grid=grid(512), stream=stream0)
        buf80 = buf75; del buf75  # reuse
        # Source Nodes: [l__self___encoder_layer_5_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param90, reinterpret_tensor(buf79, (512, 768), (768, 1), 0), _frozen_param223, alpha=1, beta=1, out=buf80)
        buf81 = reinterpret_tensor(buf72, (12, 512, 64), (32768, 64, 1), 0); del buf72  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf80, (12, 512, 64), (64, 768, 1), 0), out=buf81)
        buf82 = reinterpret_tensor(buf80, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf80  # reuse
        # Source Nodes: [attn_output_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf81, buf82, 393216, grid=grid(393216), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (512, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param92, reinterpret_tensor(buf82, (512, 768), (768, 1), 0), _frozen_param224, alpha=1, beta=1, out=buf83)
        buf87 = reinterpret_tensor(buf82, (1, 512, 768), (393216, 768, 1), 0); del buf82  # reuse
        # Source Nodes: [add_11, attention_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf83, buf79, _frozen_param93, _frozen_param94, buf87, 512, 768, grid=grid(512), stream=stream0)
        buf88 = reinterpret_tensor(buf74, (512, 3072), (3072, 1), 0); del buf74  # reuse
        # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param96, reinterpret_tensor(buf87, (512, 768), (768, 1), 0), _frozen_param225, alpha=1, beta=1, out=buf88)
        buf89 = reinterpret_tensor(buf88, (1, 512, 3072), (1572864, 3072, 1), 0); del buf88  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf89, 1572864, grid=grid(1572864), stream=stream0)
        buf90 = buf83; del buf83  # reuse
        # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param98, reinterpret_tensor(buf89, (512, 3072), (3072, 1), 0), _frozen_param226, alpha=1, beta=1, out=buf90)
        buf94 = buf79; del buf79  # reuse
        # Source Nodes: [add_12, current_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf90, buf87, _frozen_param99, _frozen_param100, buf94, 512, 768, grid=grid(512), stream=stream0)
        buf95 = buf90; del buf90  # reuse
        # Source Nodes: [l__self___encoder_layer_6_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param106, reinterpret_tensor(buf94, (512, 768), (768, 1), 0), _frozen_param227, alpha=1, beta=1, out=buf95)
        buf96 = reinterpret_tensor(buf87, (12, 512, 64), (32768, 64, 1), 0); del buf87  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf95, (12, 512, 64), (64, 768, 1), 0), out=buf96)
        buf97 = reinterpret_tensor(buf95, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf95  # reuse
        # Source Nodes: [attn_output_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf96, buf97, 393216, grid=grid(393216), stream=stream0)
        buf98 = reinterpret_tensor(buf96, (512, 768), (768, 1), 0); del buf96  # reuse
        # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param108, reinterpret_tensor(buf97, (512, 768), (768, 1), 0), _frozen_param228, alpha=1, beta=1, out=buf98)
        buf102 = reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0); del buf97  # reuse
        # Source Nodes: [add_13, attention_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf98, buf94, _frozen_param109, _frozen_param110, buf102, 512, 768, grid=grid(512), stream=stream0)
        buf103 = reinterpret_tensor(buf89, (512, 3072), (3072, 1), 0); del buf89  # reuse
        # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param112, reinterpret_tensor(buf102, (512, 768), (768, 1), 0), _frozen_param229, alpha=1, beta=1, out=buf103)
        buf104 = reinterpret_tensor(buf103, (1, 512, 3072), (1572864, 3072, 1), 0); del buf103  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf104, 1572864, grid=grid(1572864), stream=stream0)
        buf105 = buf98; del buf98  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param114, reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0), _frozen_param230, alpha=1, beta=1, out=buf105)
        buf109 = buf94; del buf94  # reuse
        # Source Nodes: [add_14, current_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf105, buf102, _frozen_param115, _frozen_param116, buf109, 512, 768, grid=grid(512), stream=stream0)
        buf110 = buf105; del buf105  # reuse
        # Source Nodes: [l__self___encoder_layer_7_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param122, reinterpret_tensor(buf109, (512, 768), (768, 1), 0), _frozen_param231, alpha=1, beta=1, out=buf110)
        buf111 = reinterpret_tensor(buf102, (12, 512, 64), (32768, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [attn_output_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf110, (12, 512, 64), (64, 768, 1), 0), out=buf111)
        buf112 = reinterpret_tensor(buf110, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf110  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf111, buf112, 393216, grid=grid(393216), stream=stream0)
        buf113 = reinterpret_tensor(buf111, (512, 768), (768, 1), 0); del buf111  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param124, reinterpret_tensor(buf112, (512, 768), (768, 1), 0), _frozen_param232, alpha=1, beta=1, out=buf113)
        buf117 = reinterpret_tensor(buf112, (1, 512, 768), (393216, 768, 1), 0); del buf112  # reuse
        # Source Nodes: [add_15, attention_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf113, buf109, _frozen_param125, _frozen_param126, buf117, 512, 768, grid=grid(512), stream=stream0)
        buf118 = reinterpret_tensor(buf104, (512, 3072), (3072, 1), 0); del buf104  # reuse
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param128, reinterpret_tensor(buf117, (512, 768), (768, 1), 0), _frozen_param233, alpha=1, beta=1, out=buf118)
        buf119 = reinterpret_tensor(buf118, (1, 512, 3072), (1572864, 3072, 1), 0); del buf118  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf119, 1572864, grid=grid(1572864), stream=stream0)
        buf120 = buf113; del buf113  # reuse
        # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param130, reinterpret_tensor(buf119, (512, 3072), (3072, 1), 0), _frozen_param234, alpha=1, beta=1, out=buf120)
        buf124 = buf109; del buf109  # reuse
        # Source Nodes: [add_16, current_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf120, buf117, _frozen_param131, _frozen_param132, buf124, 512, 768, grid=grid(512), stream=stream0)
        buf125 = buf120; del buf120  # reuse
        # Source Nodes: [l__self___encoder_layer_8_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param138, reinterpret_tensor(buf124, (512, 768), (768, 1), 0), _frozen_param235, alpha=1, beta=1, out=buf125)
        buf126 = reinterpret_tensor(buf117, (12, 512, 64), (32768, 64, 1), 0); del buf117  # reuse
        # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf125, (12, 512, 64), (64, 768, 1), 0), out=buf126)
        buf127 = reinterpret_tensor(buf125, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf125  # reuse
        # Source Nodes: [attn_output_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf126, buf127, 393216, grid=grid(393216), stream=stream0)
        buf128 = reinterpret_tensor(buf126, (512, 768), (768, 1), 0); del buf126  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param140, reinterpret_tensor(buf127, (512, 768), (768, 1), 0), _frozen_param236, alpha=1, beta=1, out=buf128)
        buf132 = reinterpret_tensor(buf127, (1, 512, 768), (393216, 768, 1), 0); del buf127  # reuse
        # Source Nodes: [add_17, attention_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf128, buf124, _frozen_param141, _frozen_param142, buf132, 512, 768, grid=grid(512), stream=stream0)
        buf133 = reinterpret_tensor(buf119, (512, 3072), (3072, 1), 0); del buf119  # reuse
        # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param144, reinterpret_tensor(buf132, (512, 768), (768, 1), 0), _frozen_param237, alpha=1, beta=1, out=buf133)
        buf134 = reinterpret_tensor(buf133, (1, 512, 3072), (1572864, 3072, 1), 0); del buf133  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf134, 1572864, grid=grid(1572864), stream=stream0)
        buf135 = buf128; del buf128  # reuse
        # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param146, reinterpret_tensor(buf134, (512, 3072), (3072, 1), 0), _frozen_param238, alpha=1, beta=1, out=buf135)
        buf139 = buf124; del buf124  # reuse
        # Source Nodes: [add_18, current_states_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf135, buf132, _frozen_param147, _frozen_param148, buf139, 512, 768, grid=grid(512), stream=stream0)
        buf140 = buf135; del buf135  # reuse
        # Source Nodes: [l__self___encoder_layer_9_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param154, reinterpret_tensor(buf139, (512, 768), (768, 1), 0), _frozen_param239, alpha=1, beta=1, out=buf140)
        buf141 = reinterpret_tensor(buf132, (12, 512, 64), (32768, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [attn_output_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf140, (12, 512, 64), (64, 768, 1), 0), out=buf141)
        buf142 = reinterpret_tensor(buf140, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf140  # reuse
        # Source Nodes: [attn_output_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf141, buf142, 393216, grid=grid(393216), stream=stream0)
        buf143 = reinterpret_tensor(buf141, (512, 768), (768, 1), 0); del buf141  # reuse
        # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param156, reinterpret_tensor(buf142, (512, 768), (768, 1), 0), _frozen_param240, alpha=1, beta=1, out=buf143)
        buf147 = reinterpret_tensor(buf142, (1, 512, 768), (393216, 768, 1), 0); del buf142  # reuse
        # Source Nodes: [add_19, attention_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf143, buf139, _frozen_param157, _frozen_param158, buf147, 512, 768, grid=grid(512), stream=stream0)
        buf148 = reinterpret_tensor(buf134, (512, 3072), (3072, 1), 0); del buf134  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param160, reinterpret_tensor(buf147, (512, 768), (768, 1), 0), _frozen_param241, alpha=1, beta=1, out=buf148)
        buf149 = reinterpret_tensor(buf148, (1, 512, 3072), (1572864, 3072, 1), 0); del buf148  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf149, 1572864, grid=grid(1572864), stream=stream0)
        buf150 = buf143; del buf143  # reuse
        # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param162, reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0), _frozen_param242, alpha=1, beta=1, out=buf150)
        buf154 = buf139; del buf139  # reuse
        # Source Nodes: [add_20, current_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf150, buf147, _frozen_param163, _frozen_param164, buf154, 512, 768, grid=grid(512), stream=stream0)
        buf155 = buf150; del buf150  # reuse
        # Source Nodes: [l__self___encoder_layer_10_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param170, reinterpret_tensor(buf154, (512, 768), (768, 1), 0), _frozen_param243, alpha=1, beta=1, out=buf155)
        buf156 = reinterpret_tensor(buf147, (12, 512, 64), (32768, 64, 1), 0); del buf147  # reuse
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf155, (12, 512, 64), (64, 768, 1), 0), out=buf156)
        buf157 = reinterpret_tensor(buf155, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf155  # reuse
        # Source Nodes: [attn_output_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf156, buf157, 393216, grid=grid(393216), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (512, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [hidden_states_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param172, reinterpret_tensor(buf157, (512, 768), (768, 1), 0), _frozen_param244, alpha=1, beta=1, out=buf158)
        buf162 = reinterpret_tensor(buf157, (1, 512, 768), (393216, 768, 1), 0); del buf157  # reuse
        # Source Nodes: [add_21, attention_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf158, buf154, _frozen_param173, _frozen_param174, buf162, 512, 768, grid=grid(512), stream=stream0)
        buf163 = reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0); del buf149  # reuse
        # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param176, reinterpret_tensor(buf162, (512, 768), (768, 1), 0), _frozen_param245, alpha=1, beta=1, out=buf163)
        buf164 = reinterpret_tensor(buf163, (1, 512, 3072), (1572864, 3072, 1), 0); del buf163  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf164, 1572864, grid=grid(1572864), stream=stream0)
        buf165 = buf158; del buf158  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param178, reinterpret_tensor(buf164, (512, 3072), (3072, 1), 0), _frozen_param246, alpha=1, beta=1, out=buf165)
        buf169 = buf154; del buf154  # reuse
        # Source Nodes: [add_22, current_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf165, buf162, _frozen_param179, _frozen_param180, buf169, 512, 768, grid=grid(512), stream=stream0)
        buf170 = buf165; del buf165  # reuse
        # Source Nodes: [l__self___encoder_layer_11_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param186, reinterpret_tensor(buf169, (512, 768), (768, 1), 0), _frozen_param247, alpha=1, beta=1, out=buf170)
        buf171 = reinterpret_tensor(buf162, (12, 512, 64), (32768, 64, 1), 0); del buf162  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, reinterpret_tensor(buf170, (12, 512, 64), (64, 768, 1), 0), out=buf171)
        del buf5
        buf172 = reinterpret_tensor(buf170, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf171, buf172, 393216, grid=grid(393216), stream=stream0)
        buf173 = reinterpret_tensor(buf171, (512, 768), (768, 1), 0); del buf171  # reuse
        # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param188, reinterpret_tensor(buf172, (512, 768), (768, 1), 0), _frozen_param248, alpha=1, beta=1, out=buf173)
        buf177 = reinterpret_tensor(buf172, (1, 512, 768), (393216, 768, 1), 0); del buf172  # reuse
        # Source Nodes: [add_23, attention_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf173, buf169, _frozen_param189, _frozen_param190, buf177, 512, 768, grid=grid(512), stream=stream0)
        buf178 = reinterpret_tensor(buf164, (512, 3072), (3072, 1), 0); del buf164  # reuse
        # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param192, reinterpret_tensor(buf177, (512, 768), (768, 1), 0), _frozen_param249, alpha=1, beta=1, out=buf178)
        buf179 = reinterpret_tensor(buf178, (1, 512, 3072), (1572864, 3072, 1), 0); del buf178  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf179, 1572864, grid=grid(1572864), stream=stream0)
        buf180 = buf173; del buf173  # reuse
        # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param194, reinterpret_tensor(buf179, (512, 3072), (3072, 1), 0), _frozen_param250, alpha=1, beta=1, out=buf180)
        del buf179
        buf184 = buf169; del buf169  # reuse
        # Source Nodes: [add_24, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf180, buf177, _frozen_param195, _frozen_param196, buf184, 512, 768, grid=grid(512), stream=stream0)
        del buf177
        del buf180
        buf185 = empty_strided((1, 768), (768, 1), device='xpu', dtype=torch.float32)
        # Source Nodes: [pooled_output], Original ATen: [aten.addmm]
        extern_kernels.addmm(_frozen_param198, reinterpret_tensor(buf184, (1, 768), (768, 1), 0), _frozen_param251, alpha=1, beta=1, out=buf185)
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [pooled_output_2], Original ATen: [aten.tanh]
        triton_poi_fused_tanh_5.run(buf186, 768, grid=grid(768), stream=stream0)
    return (buf184, buf186, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _frozen_param0
    _frozen_param0 = rand_strided((30522, 768), (768, 1), device='xpu:0', dtype=torch.float32)
    global _frozen_param3
    _frozen_param3 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param4
    _frozen_param4 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param10
    _frozen_param10 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param12
    _frozen_param12 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param13
    _frozen_param13 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param14
    _frozen_param14 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param16
    _frozen_param16 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param18
    _frozen_param18 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param19
    _frozen_param19 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param20
    _frozen_param20 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param26
    _frozen_param26 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param28
    _frozen_param28 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param29
    _frozen_param29 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param30
    _frozen_param30 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param32
    _frozen_param32 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param34
    _frozen_param34 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param35
    _frozen_param35 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param36
    _frozen_param36 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param42
    _frozen_param42 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param44
    _frozen_param44 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param45
    _frozen_param45 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param46
    _frozen_param46 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param48
    _frozen_param48 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param50
    _frozen_param50 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param51
    _frozen_param51 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param52
    _frozen_param52 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param58
    _frozen_param58 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param60
    _frozen_param60 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param61
    _frozen_param61 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param62
    _frozen_param62 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param64
    _frozen_param64 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param66
    _frozen_param66 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param67
    _frozen_param67 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param68
    _frozen_param68 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param74
    _frozen_param74 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param76
    _frozen_param76 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param77
    _frozen_param77 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param78
    _frozen_param78 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param80
    _frozen_param80 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param82
    _frozen_param82 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param83
    _frozen_param83 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param84
    _frozen_param84 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param90
    _frozen_param90 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param92
    _frozen_param92 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param93
    _frozen_param93 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param94
    _frozen_param94 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param96
    _frozen_param96 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param98
    _frozen_param98 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param99
    _frozen_param99 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param100
    _frozen_param100 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param106
    _frozen_param106 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param108
    _frozen_param108 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param109
    _frozen_param109 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param110
    _frozen_param110 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param112
    _frozen_param112 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param114
    _frozen_param114 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param115
    _frozen_param115 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param116
    _frozen_param116 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param122
    _frozen_param122 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param124
    _frozen_param124 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param125
    _frozen_param125 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param126
    _frozen_param126 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param128
    _frozen_param128 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param130
    _frozen_param130 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param131
    _frozen_param131 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param132
    _frozen_param132 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param138
    _frozen_param138 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param140
    _frozen_param140 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param141
    _frozen_param141 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param142
    _frozen_param142 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param144
    _frozen_param144 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param146
    _frozen_param146 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param147
    _frozen_param147 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param148
    _frozen_param148 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param154
    _frozen_param154 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param156
    _frozen_param156 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param157
    _frozen_param157 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param158
    _frozen_param158 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param160
    _frozen_param160 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param162
    _frozen_param162 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param163
    _frozen_param163 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param164
    _frozen_param164 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param170
    _frozen_param170 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param172
    _frozen_param172 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param173
    _frozen_param173 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param174
    _frozen_param174 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param176
    _frozen_param176 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param178
    _frozen_param178 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param179
    _frozen_param179 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param180
    _frozen_param180 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param186
    _frozen_param186 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param188
    _frozen_param188 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param189
    _frozen_param189 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param190
    _frozen_param190 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param192
    _frozen_param192 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param194
    _frozen_param194 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param195
    _frozen_param195 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param196
    _frozen_param196 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param198
    _frozen_param198 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global _frozen_param201
    _frozen_param201 = rand_strided((1, 512, 768), (393216, 768, 1), device='xpu:0', dtype=torch.float32)
    global _frozen_param202
    _frozen_param202 = rand_strided((1, 512, 768), (393216, 768, 1), device='xpu:0', dtype=torch.float32)
    global _frozen_param203
    _frozen_param203 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param204
    _frozen_param204 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param205
    _frozen_param205 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param206
    _frozen_param206 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param207
    _frozen_param207 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param208
    _frozen_param208 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param209
    _frozen_param209 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param210
    _frozen_param210 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param211
    _frozen_param211 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param212
    _frozen_param212 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param213
    _frozen_param213 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param214
    _frozen_param214 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param215
    _frozen_param215 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param216
    _frozen_param216 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param217
    _frozen_param217 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param218
    _frozen_param218 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param219
    _frozen_param219 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param220
    _frozen_param220 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param221
    _frozen_param221 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param222
    _frozen_param222 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param223
    _frozen_param223 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param224
    _frozen_param224 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param225
    _frozen_param225 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param226
    _frozen_param226 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param227
    _frozen_param227 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param228
    _frozen_param228 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param229
    _frozen_param229 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param230
    _frozen_param230 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param231
    _frozen_param231 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param232
    _frozen_param232 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param233
    _frozen_param233 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param234
    _frozen_param234 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param235
    _frozen_param235 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param236
    _frozen_param236 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param237
    _frozen_param237 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param238
    _frozen_param238 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param239
    _frozen_param239 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param240
    _frozen_param240 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param241
    _frozen_param241 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param242
    _frozen_param242 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param243
    _frozen_param243 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param244
    _frozen_param244 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param245
    _frozen_param245 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param246
    _frozen_param246 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param247
    _frozen_param247 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param248
    _frozen_param248 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param249
    _frozen_param249 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.float32)
    global _frozen_param250
    _frozen_param250 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.float32)
    global _frozen_param251
    _frozen_param251 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.float32)
    arg201_1 = rand_strided((1, 512), (512, 1), device='xpu:0', dtype=torch.int64)
    fn = lambda: call([arg201_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
