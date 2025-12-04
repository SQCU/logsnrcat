# AOT ID: ['1_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: ./.inductor_cache\pf\cpfvgaeqmjiabws2hpxv3kl2wyv2xegvvry25ykbenrihrfmycw5.py
# Topologically Sorted Source Nodes: [x, args, sin, cos, spins, x_pad, unfold, patches, permute, patches_1, unsqueeze_1, spins_expanded, cat_1], Original ATen: [aten.unsqueeze, aten.mul, aten.sin, aten.cos, aten.cat, aten.reflection_pad2d, aten.unfold, aten.permute, aten.clone, aten._unsafe_view, aten.expand]
# Source node to ATen node mapping:
#   args => mul
#   cat_1 => cat_1
#   cos => cos
#   patches => unfold_1
#   patches_1 => clone, view
#   permute => permute
#   sin => sin
#   spins => cat
#   spins_expanded => expand
#   unfold => unfold
#   unsqueeze_1 => unsqueeze_1
#   x => unsqueeze
#   x_pad => _unsafe_index, _unsafe_index_1, abs_1, abs_2, iota, sub, sub_1
# Graph fragment:
#   %primals_3 : Tensor "f32[256, 3, 16, 16][768, 256, 16, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=primals_2]
#   %unsqueeze : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_1, -1), kwargs = {})
#   %mul : Tensor "f32[256, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %primals_2), kwargs = {})
#   %sin : Tensor "f32[256, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul,), kwargs = {})
#   %cos : Tensor "f32[256, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul,), kwargs = {})
#   %cat : Tensor "f32[256, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sin, %cos], -1), kwargs = {})
#   %iota : Tensor "i64[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (18,), kwargs = {start: -1, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %abs_1 : Tensor "i64[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%iota,), kwargs = {})
#   %sub : Tensor "i64[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (15, %abs_1), kwargs = {})
#   %abs_2 : Tensor "i64[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %sub_1 : Tensor "i64[18][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (15, %abs_2), kwargs = {})
#   %_unsafe_index : Tensor "f32[256, 3, 18, 16][864, 288, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_3, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[256, 3, 18, 18][972, 324, 18, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
#   %unfold : Tensor "f32[256, 3, 8, 18, 4][972, 324, 36, 1, 18]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unfold.default](args = (%_unsafe_index_1, 2, 4, 2), kwargs = {})
#   %unfold_1 : Tensor "f32[256, 3, 8, 8, 4, 4][972, 324, 36, 2, 18, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unfold.default](args = (%unfold, 3, 4, 2), kwargs = {})
#   %permute : Tensor "f32[256, 8, 8, 3, 4, 4][972, 36, 2, 324, 18, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unfold_1, [0, 2, 3, 1, 4, 5]), kwargs = {})
#   %clone : Tensor "f32[256, 8, 8, 3, 4, 4][3072, 384, 48, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[256, 64, 48][3072, 48, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [256, 64, 48]), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[256, 1, 8][8, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 1), kwargs = {})
#   %expand : Tensor "f32[256, 64, 8][8, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [-1, 64, -1]), kwargs = {})
#   %cat_1 : Tensor "f32[256, 64, 56][3584, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %expand], -1), kwargs = {})
#   return %cat_1
triton_poi_fused__unsafe_view_cat_clone_cos_expand_mul_permute_reflection_pad2d_sin_unfold_unsqueeze_0 = async_compile.triton('triton_poi_fused__unsafe_view_cat_clone_cos_expand_mul_permute_reflection_pad2d_sin_unfold_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_cat_clone_cos_expand_mul_permute_reflection_pad2d_sin_unfold_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7340256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_cat_clone_cos_expand_mul_permute_reflection_pad2d_sin_unfold_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 917504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 56)
    x1 = ((xindex // 56) % 64)
    x2 = xindex // 3584
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 48, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (255 + ((-1)*tl_math.abs((-15) + tl_math.abs((-1) + 2*((x1 % 8)) + (((x0) % 4))))) + ((-16)*tl_math.abs((-15) + tl_math.abs((-1) + 2*(x1 // 8) + ((((x0) // 4) % 4))))) + 256*((((x0) // 16) % 3)) + 768*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 56, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = (-48) + x0
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tmp13 & tmp6
    tmp15 = tl.load(in_ptr1 + (x2), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr2 + ((-48) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl_math.sin(tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tmp9 >= tmp12
    tmp22 = tl.full([1], 8, tl.int64)
    tmp23 = tmp9 < tmp22
    tmp24 = tmp21 & tmp6
    tmp25 = tl.load(in_ptr1 + (x2), tmp24, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + ((-4) + ((-48) + x0)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl_math.cos(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp24, tmp28, tmp29)
    tmp31 = tl.where(tmp13, tmp20, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp6, tmp31, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tl.store(out_ptr0 + (x4), tmp34, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\sq\csqei4ernhpyv4jvueeixhlr23p4u6itdditdjjgfbzkrcolg2j6.py
# Topologically Sorted Source Nodes: [x_1, input_1], Original ATen: [aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   input_1 => add, add_1, mul_1, mul_2, rsqrt, sub_4, var_mean
#   x_1 => view_2
# Graph fragment:
#   %addmm : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm]
#   %buf3 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf3]
#   %getitem_1 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_6 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_7 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_7]
#   %view_2 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [256, 64, 256]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %add : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub_4 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %getitem_1), kwargs = {})
#   %mul_1 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt), kwargs = {})
#   %mul_2 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_6), kwargs = {})
#   %add_1 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_7), kwargs = {})
#   return %getitem_1,%buf3,%rsqrt,%add_1
triton_per_fused_native_layer_norm_view_1 = async_compile.triton('triton_per_fused_native_layer_norm_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 50333696}}
)
@triton.jit
def triton_per_fused_native_layer_norm_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp21 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = (tmp5 / tmp7)
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = 256.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp0 - tmp8
    tmp20 = tmp19 * tmp18
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (r0_1 + 256*x0), tmp24, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\s3\cs3vz25z3slks6vkobselajtgnt5mvgfd4hzd3ubx7b4lxb5nsf4.py
# Topologically Sorted Source Nodes: [input_2, x_norm], Original ATen: [aten.addmm, aten.view, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   input_2 => add_tensor_1, view_4
#   x_norm => add_2, mean, mul_3, pow_1, rsqrt_1
# Graph fragment:
#   %mm_default_1 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_9 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_9]
#   %buf8 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf8]
#   %rsqrt_1 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %add_tensor_1 : Tensor "f32[16384, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_9), kwargs = {})
#   %view_4 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [256, 64, 256]), kwargs = {})
#   %pow_1 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_4, 2), kwargs = {})
#   %mean : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_1 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_3 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %rsqrt_1), kwargs = {})
#   return %buf8,%rsqrt_1,%mul_3
triton_per_fused_add_addmm_mean_mul_pow_rsqrt_view_2 = async_compile.triton('triton_per_fused_add_addmm_mean_mul_pow_rsqrt_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_mean_mul_pow_rsqrt_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 50332672}}
)
@triton.jit
def triton_per_fused_add_addmm_mean_mul_pow_rsqrt_view_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = 256.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1.1920928955078125e-07
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp2 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp12, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\7q\c7qiuue3ue4btyvzsg3sps7tahqzabmobbyn56zkhjvci7t4ixe6.py
# Topologically Sorted Source Nodes: [Q], Original ATen: [aten.eye]
# Source node to ATen node mapping:
#   Q => eq, full_default, full_default_1, iota_2, unsqueeze_2, where
# Graph fragment:
#   %iota_2 : Tensor "i64[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_2, -1), kwargs = {})
#   %eq : Tensor "b8[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_2, %iota_2), kwargs = {})
#   %full_default : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[64, 64][64, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %full_default_1), kwargs = {})
#   return %where
triton_poi_fused_eye_3 = async_compile.triton('triton_poi_fused_eye_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eye_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eye_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp0 = x1
    tmp1 = x0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ft\cftfsw72aqxepumeqvxnum5yj6mhtlrjjmt37ycg7l226ynpygvi.py
# Topologically Sorted Source Nodes: [getitem_3, v_2, pow_1, sum_1, v_norm_sq, truediv, mul_1], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_3 => select
#   mul_1 => mul_5
#   pow_1 => pow_2
#   sum_1 => sum_1
#   truediv => mul_4, reciprocal
#   v_2 => unsqueeze_3
#   v_norm_sq => add_3
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_1]
#   %reciprocal : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal]
#   %select : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 1), kwargs = {})
#   %pow_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_3, 2), kwargs = {})
#   %sum_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_2,), kwargs = {})
#   %add_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-08), kwargs = {})
#   %reciprocal : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_3,), kwargs = {})
#   %mul_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 2.0), kwargs = {})
#   %mul_5 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_3), kwargs = {})
#   return %sum_1,%reciprocal,%mul_5
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\45\c45lndmjkbilse2nyouudyvmioubjd47fvsz6nqrhlc6n6iplgnb.py
# Topologically Sorted Source Nodes: [Q_1], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   Q_1 => sub_5
# Graph fragment:
#   %mm_2 : Tensor "f32[64, 64][64, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %sub_5 : Tensor "f32[64, 64][64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %mm_2), kwargs = {})
#   return %sub_5
triton_poi_fused_sub_5 = async_compile.triton('triton_poi_fused_sub_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sub_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp6 = tl.load(in_out_ptr0 + (x2), None)
    tmp0 = x1
    tmp1 = x0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 - tmp6
    tl.store(in_out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ns\cnsa52c264aur36b2qh7a36zezyr6n7flwvti4zyalh2b4hx6cxw.py
# Topologically Sorted Source Nodes: [getitem_4, v_3, pow_2, sum_2, v_norm_sq_1, truediv_1, mul_2], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_4 => select_1
#   mul_2 => mul_7
#   pow_2 => pow_3
#   sum_2 => sum_2
#   truediv_1 => mul_6, reciprocal_1
#   v_3 => unsqueeze_4
#   v_norm_sq_1 => add_4
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_2 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_2]
#   %reciprocal_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_1]
#   %select_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 1), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 1), kwargs = {})
#   %pow_3 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_4, 2), kwargs = {})
#   %sum_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_3,), kwargs = {})
#   %add_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 1e-08), kwargs = {})
#   %reciprocal_1 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_4,), kwargs = {})
#   %mul_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 2.0), kwargs = {})
#   %mul_7 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_4), kwargs = {})
#   return %sum_2,%reciprocal_1,%mul_7
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (64 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\iv\civ5mnulfu2f536pezavfzslbua2oysqqtldmpikfimph22hwwzz.py
# Topologically Sorted Source Nodes: [Q_2], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   Q_2 => sub_6
# Graph fragment:
#   %sub_5 : Tensor "f32[64, 64][64, 1]cuda:0" = PlaceHolder[target=sub_5]
#   %mm_4 : Tensor "f32[64, 64][64, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %sub_6 : Tensor "f32[64, 64][64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, %mm_4), kwargs = {})
#   return %sub_6
triton_poi_fused_sub_7 = async_compile.triton('triton_poi_fused_sub_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sub_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 - tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\tq\ctqesg5jgssd7lewnroxgmlpof24b6erqb3lxgtep2il3onq4lfu.py
# Topologically Sorted Source Nodes: [getitem_5, v_4, pow_3, sum_3, v_norm_sq_2, truediv_2, mul_3], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_5 => select_2
#   mul_3 => mul_9
#   pow_3 => pow_4
#   sum_3 => sum_3
#   truediv_2 => mul_8, reciprocal_2
#   v_4 => unsqueeze_5
#   v_norm_sq_2 => add_5
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_3 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_3]
#   %reciprocal_2 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_2]
#   %select_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 2), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_2, 1), kwargs = {})
#   %pow_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_5, 2), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_4,), kwargs = {})
#   %add_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 1e-08), kwargs = {})
#   %reciprocal_2 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_5,), kwargs = {})
#   %mul_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 2.0), kwargs = {})
#   %mul_9 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %unsqueeze_5), kwargs = {})
#   return %sum_3,%reciprocal_2,%mul_9
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (128 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\26\c26z2mvereducu6zmzj4i27n57fqnq6v77awcpehbyrczrvp4d3i.py
# Topologically Sorted Source Nodes: [getitem_6, v_5, pow_4, sum_4, v_norm_sq_3, truediv_3, mul_4], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_6 => select_3
#   mul_4 => mul_11
#   pow_4 => pow_5
#   sum_4 => sum_4
#   truediv_3 => mul_10, reciprocal_3
#   v_5 => unsqueeze_6
#   v_norm_sq_3 => add_6
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_4 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_4]
#   %reciprocal_3 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_3]
#   %select_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 3), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_3, 1), kwargs = {})
#   %pow_5 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_6, 2), kwargs = {})
#   %sum_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_5,), kwargs = {})
#   %add_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_4, 1e-08), kwargs = {})
#   %reciprocal_3 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_6,), kwargs = {})
#   %mul_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 2.0), kwargs = {})
#   %mul_11 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_6), kwargs = {})
#   return %sum_4,%reciprocal_3,%mul_11
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (192 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\gs\cgsh6zycaovl3x4x22muskhzfxrgnh7fvf2itthpqxhszmrvbkat.py
# Topologically Sorted Source Nodes: [getitem_7, v_6, pow_5, sum_5, v_norm_sq_4, truediv_4, mul_5], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_7 => select_4
#   mul_5 => mul_13
#   pow_5 => pow_6
#   sum_5 => sum_5
#   truediv_4 => mul_12, reciprocal_4
#   v_6 => unsqueeze_7
#   v_norm_sq_4 => add_7
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_5 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_5]
#   %reciprocal_4 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_4]
#   %select_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 4), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_4, 1), kwargs = {})
#   %pow_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_7, 2), kwargs = {})
#   %sum_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_6,), kwargs = {})
#   %add_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_5, 1e-08), kwargs = {})
#   %reciprocal_4 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_7,), kwargs = {})
#   %mul_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 2.0), kwargs = {})
#   %mul_13 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_7), kwargs = {})
#   return %sum_5,%reciprocal_4,%mul_13
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (256 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\7z\c7zejvwnh3bhogbcevqg3duh3gv5bp2jlmb6auut3dfv6pi4dbvc.py
# Topologically Sorted Source Nodes: [getitem_8, v_7, pow_6, sum_6, v_norm_sq_5, truediv_5, mul_6], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_8 => select_5
#   mul_6 => mul_15
#   pow_6 => pow_7
#   sum_6 => sum_6
#   truediv_5 => mul_14, reciprocal_5
#   v_7 => unsqueeze_8
#   v_norm_sq_5 => add_8
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_6 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_6]
#   %reciprocal_5 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_5]
#   %select_5 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 5), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_5, 1), kwargs = {})
#   %pow_7 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_8, 2), kwargs = {})
#   %sum_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_7,), kwargs = {})
#   %add_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_6, 1e-08), kwargs = {})
#   %reciprocal_5 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_8,), kwargs = {})
#   %mul_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 2.0), kwargs = {})
#   %mul_15 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_8), kwargs = {})
#   return %sum_6,%reciprocal_5,%mul_15
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (320 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\os\cosmzsqad3wpjqb2gc4wcholxnoivo5jf5hz66izjwkw6c2br772.py
# Topologically Sorted Source Nodes: [getitem_9, v_8, pow_7, sum_7, v_norm_sq_6, truediv_6, mul_7], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_9 => select_6
#   mul_7 => mul_17
#   pow_7 => pow_8
#   sum_7 => sum_7
#   truediv_6 => mul_16, reciprocal_6
#   v_8 => unsqueeze_9
#   v_norm_sq_6 => add_9
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_7 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_7]
#   %reciprocal_6 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_6]
#   %select_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 6), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_6, 1), kwargs = {})
#   %pow_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_9, 2), kwargs = {})
#   %sum_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_8,), kwargs = {})
#   %add_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_7, 1e-08), kwargs = {})
#   %reciprocal_6 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_9,), kwargs = {})
#   %mul_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 2.0), kwargs = {})
#   %mul_17 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_9), kwargs = {})
#   return %sum_7,%reciprocal_6,%mul_17
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (384 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\cj\ccj2cfu7foprx2yfvlrv2f3v5p5hvwwdvbojjhfrxgwddkoyivw2.py
# Topologically Sorted Source Nodes: [getitem_10, v_9, pow_8, sum_8, v_norm_sq_7, truediv_7, mul_8], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_10 => select_7
#   mul_8 => mul_19
#   pow_8 => pow_9
#   sum_8 => sum_8
#   truediv_7 => mul_18, reciprocal_7
#   v_9 => unsqueeze_10
#   v_norm_sq_7 => add_10
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_8 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_8]
#   %reciprocal_7 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_7]
#   %select_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 7), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_7, 1), kwargs = {})
#   %pow_9 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_10, 2), kwargs = {})
#   %sum_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_9,), kwargs = {})
#   %add_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_8, 1e-08), kwargs = {})
#   %reciprocal_7 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_10,), kwargs = {})
#   %mul_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 2.0), kwargs = {})
#   %mul_19 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_10), kwargs = {})
#   return %sum_8,%reciprocal_7,%mul_19
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (448 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\yy\cyyrwbbwmav4p6d4pw4ezb5ttqqnc5heqfy5wfjys7e3lay42gct.py
# Topologically Sorted Source Nodes: [getitem_11, v_10, pow_9, sum_9, v_norm_sq_8, truediv_8, mul_9], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_11 => select_8
#   mul_9 => mul_21
#   pow_9 => pow_10
#   sum_9 => sum_9
#   truediv_8 => mul_20, reciprocal_8
#   v_10 => unsqueeze_11
#   v_norm_sq_8 => add_11
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_9 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_9]
#   %reciprocal_8 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_8]
#   %select_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 8), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_8, 1), kwargs = {})
#   %pow_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_11, 2), kwargs = {})
#   %sum_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_10,), kwargs = {})
#   %add_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_9, 1e-08), kwargs = {})
#   %reciprocal_8 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_11,), kwargs = {})
#   %mul_20 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 2.0), kwargs = {})
#   %mul_21 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_11), kwargs = {})
#   return %sum_9,%reciprocal_8,%mul_21
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (512 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\we\cwe6fjf2bukvuiabs3qn5mujvghxwu3y7z3gbvw2fcngynqybujr.py
# Topologically Sorted Source Nodes: [getitem_12, v_11, pow_10, sum_10, v_norm_sq_9, truediv_9, mul_10], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_12 => select_9
#   mul_10 => mul_23
#   pow_10 => pow_11
#   sum_10 => sum_10
#   truediv_9 => mul_22, reciprocal_9
#   v_11 => unsqueeze_12
#   v_norm_sq_9 => add_12
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_10 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_10]
#   %reciprocal_9 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_9]
#   %select_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 9), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_9, 1), kwargs = {})
#   %pow_11 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_12, 2), kwargs = {})
#   %sum_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_11,), kwargs = {})
#   %add_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_10, 1e-08), kwargs = {})
#   %reciprocal_9 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_12,), kwargs = {})
#   %mul_22 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 2.0), kwargs = {})
#   %mul_23 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_12), kwargs = {})
#   return %sum_10,%reciprocal_9,%mul_23
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (576 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\io\ciole64b5f24c5qch7ic4542slgqazsq27of7ni7o5ljnzkiaowj.py
# Topologically Sorted Source Nodes: [getitem_13, v_12, pow_11, sum_11, v_norm_sq_10, truediv_10, mul_11], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_13 => select_10
#   mul_11 => mul_25
#   pow_11 => pow_12
#   sum_11 => sum_11
#   truediv_10 => mul_24, reciprocal_10
#   v_12 => unsqueeze_13
#   v_norm_sq_10 => add_13
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_11 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_11]
#   %reciprocal_10 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_10]
#   %select_10 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 10), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_10, 1), kwargs = {})
#   %pow_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_13, 2), kwargs = {})
#   %sum_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_12,), kwargs = {})
#   %add_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_11, 1e-08), kwargs = {})
#   %reciprocal_10 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_13,), kwargs = {})
#   %mul_24 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 2.0), kwargs = {})
#   %mul_25 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %unsqueeze_13), kwargs = {})
#   return %sum_11,%reciprocal_10,%mul_25
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (640 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ko\ckojzcexf54vctbgshrv3kmrmxncsvo4kdd72y3mt7mgzdjajkjm.py
# Topologically Sorted Source Nodes: [getitem_14, v_13, pow_12, sum_12, v_norm_sq_11, truediv_11, mul_12], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_14 => select_11
#   mul_12 => mul_27
#   pow_12 => pow_13
#   sum_12 => sum_12
#   truediv_11 => mul_26, reciprocal_11
#   v_13 => unsqueeze_14
#   v_norm_sq_11 => add_14
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_12 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_12]
#   %reciprocal_11 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_11]
#   %select_11 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 11), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_11, 1), kwargs = {})
#   %pow_13 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_14, 2), kwargs = {})
#   %sum_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_13,), kwargs = {})
#   %add_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_12, 1e-08), kwargs = {})
#   %reciprocal_11 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_14,), kwargs = {})
#   %mul_26 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 2.0), kwargs = {})
#   %mul_27 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_14), kwargs = {})
#   return %sum_12,%reciprocal_11,%mul_27
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (704 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\cx\ccxxbjxtxbyauhxhr3px4fdds67oxfm5ojung346qyshod7xqkc2.py
# Topologically Sorted Source Nodes: [getitem_15, v_14, pow_13, sum_13, v_norm_sq_12, truediv_12, mul_13], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_15 => select_12
#   mul_13 => mul_29
#   pow_13 => pow_14
#   sum_13 => sum_13
#   truediv_12 => mul_28, reciprocal_12
#   v_14 => unsqueeze_15
#   v_norm_sq_12 => add_15
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_13 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_13]
#   %reciprocal_12 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_12]
#   %select_12 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 12), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_12, 1), kwargs = {})
#   %pow_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_15, 2), kwargs = {})
#   %sum_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_14,), kwargs = {})
#   %add_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_13, 1e-08), kwargs = {})
#   %reciprocal_12 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_15,), kwargs = {})
#   %mul_28 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 2.0), kwargs = {})
#   %mul_29 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_15), kwargs = {})
#   return %sum_13,%reciprocal_12,%mul_29
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (768 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\gj\cgj7lzbremn3vh7vdyu4pihrima4553lrvxr35eiomsp2l4gc6fx.py
# Topologically Sorted Source Nodes: [getitem_16, v_15, pow_14, sum_14, v_norm_sq_13, truediv_13, mul_14], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_16 => select_13
#   mul_14 => mul_31
#   pow_14 => pow_15
#   sum_14 => sum_14
#   truediv_13 => mul_30, reciprocal_13
#   v_15 => unsqueeze_16
#   v_norm_sq_13 => add_16
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_14 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_14]
#   %reciprocal_13 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_13]
#   %select_13 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 13), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_13, 1), kwargs = {})
#   %pow_15 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_16, 2), kwargs = {})
#   %sum_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_15,), kwargs = {})
#   %add_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_14, 1e-08), kwargs = {})
#   %reciprocal_13 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_16,), kwargs = {})
#   %mul_30 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 2.0), kwargs = {})
#   %mul_31 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %unsqueeze_16), kwargs = {})
#   return %sum_14,%reciprocal_13,%mul_31
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (832 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ly\clyy5xkryt5dzlfiwj73jjb37ttozsxcq4xe6aayhzvd5ineklgi.py
# Topologically Sorted Source Nodes: [getitem_17, v_16, pow_15, sum_15, v_norm_sq_14, truediv_14, mul_15], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_17 => select_14
#   mul_15 => mul_33
#   pow_15 => pow_16
#   sum_15 => sum_15
#   truediv_14 => mul_32, reciprocal_14
#   v_16 => unsqueeze_17
#   v_norm_sq_14 => add_17
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_15 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_15]
#   %reciprocal_14 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_14]
#   %select_14 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 14), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_14, 1), kwargs = {})
#   %pow_16 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_17, 2), kwargs = {})
#   %sum_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_16,), kwargs = {})
#   %add_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_15, 1e-08), kwargs = {})
#   %reciprocal_14 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_17,), kwargs = {})
#   %mul_32 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_14, 2.0), kwargs = {})
#   %mul_33 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %unsqueeze_17), kwargs = {})
#   return %sum_15,%reciprocal_14,%mul_33
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (896 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\pq\cpqdqqid6rowlbgbkliidkg2n6l53jgiz2guwlxmeudqrvy2uvpa.py
# Topologically Sorted Source Nodes: [getitem_18, v_17, pow_16, sum_16, v_norm_sq_15, truediv_15, mul_16], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_18 => select_15
#   mul_16 => mul_35
#   pow_16 => pow_17
#   sum_16 => sum_16
#   truediv_15 => mul_34, reciprocal_15
#   v_17 => unsqueeze_18
#   v_norm_sq_15 => add_18
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_16 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_16]
#   %reciprocal_15 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_15]
#   %select_15 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 15), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_15, 1), kwargs = {})
#   %pow_17 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_18, 2), kwargs = {})
#   %sum_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_17,), kwargs = {})
#   %add_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_16, 1e-08), kwargs = {})
#   %reciprocal_15 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_18,), kwargs = {})
#   %mul_34 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_15, 2.0), kwargs = {})
#   %mul_35 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_18), kwargs = {})
#   return %sum_16,%reciprocal_15,%mul_35
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (960 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\e2\ce2c6szun42pm4mbg2ozc4u6ufcomdesopr7aj4n4l57iknt2vb4.py
# Topologically Sorted Source Nodes: [getitem_19, v_18, pow_17, sum_17, v_norm_sq_16, truediv_16, mul_17], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_19 => select_16
#   mul_17 => mul_37
#   pow_17 => pow_18
#   sum_17 => sum_17
#   truediv_16 => mul_36, reciprocal_16
#   v_18 => unsqueeze_19
#   v_norm_sq_16 => add_19
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_17 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_17]
#   %reciprocal_16 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_16]
#   %select_16 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 16), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_16, 1), kwargs = {})
#   %pow_18 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_19, 2), kwargs = {})
#   %sum_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_18,), kwargs = {})
#   %add_19 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_17, 1e-08), kwargs = {})
#   %reciprocal_16 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_19,), kwargs = {})
#   %mul_36 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 2.0), kwargs = {})
#   %mul_37 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %unsqueeze_19), kwargs = {})
#   return %sum_17,%reciprocal_16,%mul_37
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1024 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\iv\civwm3duado5ywixridpy5xdq3bdency2ze7dwgapm62gple3wfi.py
# Topologically Sorted Source Nodes: [getitem_20, v_19, pow_18, sum_18, v_norm_sq_17, truediv_17, mul_18], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_20 => select_17
#   mul_18 => mul_39
#   pow_18 => pow_19
#   sum_18 => sum_18
#   truediv_17 => mul_38, reciprocal_17
#   v_19 => unsqueeze_20
#   v_norm_sq_17 => add_20
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_18 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_18]
#   %reciprocal_17 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_17]
#   %select_17 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 17), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_17, 1), kwargs = {})
#   %pow_19 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_20, 2), kwargs = {})
#   %sum_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_19,), kwargs = {})
#   %add_20 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_18, 1e-08), kwargs = {})
#   %reciprocal_17 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_20,), kwargs = {})
#   %mul_38 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 2.0), kwargs = {})
#   %mul_39 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %unsqueeze_20), kwargs = {})
#   return %sum_18,%reciprocal_17,%mul_39
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1088 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\cp\ccplxnjhjewuplmsv5ffqppf5io5nkkyonc3ikibiiaeqqrljxnz.py
# Topologically Sorted Source Nodes: [getitem_21, v_20, pow_19, sum_19, v_norm_sq_18, truediv_18, mul_19], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_21 => select_18
#   mul_19 => mul_41
#   pow_19 => pow_20
#   sum_19 => sum_19
#   truediv_18 => mul_40, reciprocal_18
#   v_20 => unsqueeze_21
#   v_norm_sq_18 => add_21
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_19 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_19]
#   %reciprocal_18 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_18]
#   %select_18 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 18), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_18, 1), kwargs = {})
#   %pow_20 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_21, 2), kwargs = {})
#   %sum_19 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_20,), kwargs = {})
#   %add_21 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_19, 1e-08), kwargs = {})
#   %reciprocal_18 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_21,), kwargs = {})
#   %mul_40 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_18, 2.0), kwargs = {})
#   %mul_41 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_21), kwargs = {})
#   return %sum_19,%reciprocal_18,%mul_41
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1152 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ln\clnshxxzhssrygf3mnovorjw5wbduooagnwehycz7pgmedz34npe.py
# Topologically Sorted Source Nodes: [getitem_22, v_21, pow_20, sum_20, v_norm_sq_19, truediv_19, mul_20], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_22 => select_19
#   mul_20 => mul_43
#   pow_20 => pow_21
#   sum_20 => sum_20
#   truediv_19 => mul_42, reciprocal_19
#   v_21 => unsqueeze_22
#   v_norm_sq_19 => add_22
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_20 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_20]
#   %reciprocal_19 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_19]
#   %select_19 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 19), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_19, 1), kwargs = {})
#   %pow_21 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_22, 2), kwargs = {})
#   %sum_20 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_21,), kwargs = {})
#   %add_22 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_20, 1e-08), kwargs = {})
#   %reciprocal_19 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_22,), kwargs = {})
#   %mul_42 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_19, 2.0), kwargs = {})
#   %mul_43 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %unsqueeze_22), kwargs = {})
#   return %sum_20,%reciprocal_19,%mul_43
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1216 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\32\c32h7okg7dakqd4chm2z2silx3fcfjqxnu3mkxbmjervujqdebil.py
# Topologically Sorted Source Nodes: [getitem_23, v_22, pow_21, sum_21, v_norm_sq_20, truediv_20, mul_21], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_23 => select_20
#   mul_21 => mul_45
#   pow_21 => pow_22
#   sum_21 => sum_21
#   truediv_20 => mul_44, reciprocal_20
#   v_22 => unsqueeze_23
#   v_norm_sq_20 => add_23
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_21 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_21]
#   %reciprocal_20 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_20]
#   %select_20 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 20), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_20, 1), kwargs = {})
#   %pow_22 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_23, 2), kwargs = {})
#   %sum_21 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_22,), kwargs = {})
#   %add_23 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_21, 1e-08), kwargs = {})
#   %reciprocal_20 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_23,), kwargs = {})
#   %mul_44 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_20, 2.0), kwargs = {})
#   %mul_45 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %unsqueeze_23), kwargs = {})
#   return %sum_21,%reciprocal_20,%mul_45
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1280 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\q7\cq7obomp6lscxxame53i7qg6lkh4fae5cogibmlgf6tizd4d4scz.py
# Topologically Sorted Source Nodes: [getitem_24, v_23, pow_22, sum_22, v_norm_sq_21, truediv_21, mul_22], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_24 => select_21
#   mul_22 => mul_47
#   pow_22 => pow_23
#   sum_22 => sum_22
#   truediv_21 => mul_46, reciprocal_21
#   v_23 => unsqueeze_24
#   v_norm_sq_21 => add_24
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_22 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_22]
#   %reciprocal_21 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_21]
#   %select_21 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 21), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_21, 1), kwargs = {})
#   %pow_23 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_24, 2), kwargs = {})
#   %sum_22 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_23,), kwargs = {})
#   %add_24 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_22, 1e-08), kwargs = {})
#   %reciprocal_21 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_24,), kwargs = {})
#   %mul_46 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_21, 2.0), kwargs = {})
#   %mul_47 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_24), kwargs = {})
#   return %sum_22,%reciprocal_21,%mul_47
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1344 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\w6\cw67wt5e4zj3r5wxlikfucm5ksvhytwmtgfaersy7x4462pisdgm.py
# Topologically Sorted Source Nodes: [getitem_25, v_24, pow_23, sum_23, v_norm_sq_22, truediv_22, mul_23], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_25 => select_22
#   mul_23 => mul_49
#   pow_23 => pow_24
#   sum_23 => sum_23
#   truediv_22 => mul_48, reciprocal_22
#   v_24 => unsqueeze_25
#   v_norm_sq_22 => add_25
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_23 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_23]
#   %reciprocal_22 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_22]
#   %select_22 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 22), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_22, 1), kwargs = {})
#   %pow_24 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_25, 2), kwargs = {})
#   %sum_23 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_24,), kwargs = {})
#   %add_25 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_23, 1e-08), kwargs = {})
#   %reciprocal_22 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_25,), kwargs = {})
#   %mul_48 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_22, 2.0), kwargs = {})
#   %mul_49 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %unsqueeze_25), kwargs = {})
#   return %sum_23,%reciprocal_22,%mul_49
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1408 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\bu\cbuczunvxztn2buhtuusyzgezluwc7eazxsc5jawh55iqn4m7me5.py
# Topologically Sorted Source Nodes: [getitem_26, v_25, pow_24, sum_24, v_norm_sq_23, truediv_23, mul_24], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_26 => select_23
#   mul_24 => mul_51
#   pow_24 => pow_25
#   sum_24 => sum_24
#   truediv_23 => mul_50, reciprocal_23
#   v_25 => unsqueeze_26
#   v_norm_sq_23 => add_26
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_24 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_24]
#   %reciprocal_23 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_23]
#   %select_23 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 23), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_23, 1), kwargs = {})
#   %pow_25 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_26, 2), kwargs = {})
#   %sum_24 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_25,), kwargs = {})
#   %add_26 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_24, 1e-08), kwargs = {})
#   %reciprocal_23 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_26,), kwargs = {})
#   %mul_50 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_23, 2.0), kwargs = {})
#   %mul_51 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %unsqueeze_26), kwargs = {})
#   return %sum_24,%reciprocal_23,%mul_51
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1472 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\7z\c7zrl4577wmsiymfy4smns32pcbezxlc7glvpfn32gllpvibaf4t.py
# Topologically Sorted Source Nodes: [getitem_27, v_26, pow_25, sum_25, v_norm_sq_24, truediv_24, mul_25], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_27 => select_24
#   mul_25 => mul_53
#   pow_25 => pow_26
#   sum_25 => sum_25
#   truediv_24 => mul_52, reciprocal_24
#   v_26 => unsqueeze_27
#   v_norm_sq_24 => add_27
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_25 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_25]
#   %reciprocal_24 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_24]
#   %select_24 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 24), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_24, 1), kwargs = {})
#   %pow_26 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_27, 2), kwargs = {})
#   %sum_25 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_26,), kwargs = {})
#   %add_27 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_25, 1e-08), kwargs = {})
#   %reciprocal_24 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_27,), kwargs = {})
#   %mul_52 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_24, 2.0), kwargs = {})
#   %mul_53 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_27), kwargs = {})
#   return %sum_25,%reciprocal_24,%mul_53
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1536 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\5q\c5qzwmi7te4d5ueigy36ptvaj3xvi2wtjairmahdc6yycd4e67q3.py
# Topologically Sorted Source Nodes: [getitem_28, v_27, pow_26, sum_26, v_norm_sq_25, truediv_25, mul_26], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_28 => select_25
#   mul_26 => mul_55
#   pow_26 => pow_27
#   sum_26 => sum_26
#   truediv_25 => mul_54, reciprocal_25
#   v_27 => unsqueeze_28
#   v_norm_sq_25 => add_28
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_26 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_26]
#   %reciprocal_25 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_25]
#   %select_25 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 25), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_25, 1), kwargs = {})
#   %pow_27 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_28, 2), kwargs = {})
#   %sum_26 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_27,), kwargs = {})
#   %add_28 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_26, 1e-08), kwargs = {})
#   %reciprocal_25 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_28,), kwargs = {})
#   %mul_54 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_25, 2.0), kwargs = {})
#   %mul_55 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %unsqueeze_28), kwargs = {})
#   return %sum_26,%reciprocal_25,%mul_55
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1600 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\re\cre534hxh5di6glhwytvvhfqn4dbgvovxusast7ril3oybvd3ncb.py
# Topologically Sorted Source Nodes: [getitem_29, v_28, pow_27, sum_27, v_norm_sq_26, truediv_26, mul_27], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_29 => select_26
#   mul_27 => mul_57
#   pow_27 => pow_28
#   sum_27 => sum_27
#   truediv_26 => mul_56, reciprocal_26
#   v_28 => unsqueeze_29
#   v_norm_sq_26 => add_29
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_27 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_27]
#   %reciprocal_26 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_26]
#   %select_26 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 26), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_26, 1), kwargs = {})
#   %pow_28 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_29, 2), kwargs = {})
#   %sum_27 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_28,), kwargs = {})
#   %add_29 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_27, 1e-08), kwargs = {})
#   %reciprocal_26 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_29,), kwargs = {})
#   %mul_56 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_26, 2.0), kwargs = {})
#   %mul_57 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %unsqueeze_29), kwargs = {})
#   return %sum_27,%reciprocal_26,%mul_57
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1664 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ep\cepw44rt4r64pnlffa5gv4o45kkawrbvu3aclkzwstxn4b2lwmrd.py
# Topologically Sorted Source Nodes: [getitem_30, v_29, pow_28, sum_28, v_norm_sq_27, truediv_27, mul_28], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_30 => select_27
#   mul_28 => mul_59
#   pow_28 => pow_29
#   sum_28 => sum_28
#   truediv_27 => mul_58, reciprocal_27
#   v_29 => unsqueeze_30
#   v_norm_sq_27 => add_30
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_28 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_28]
#   %reciprocal_27 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_27]
#   %select_27 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 27), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_27, 1), kwargs = {})
#   %pow_29 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_30, 2), kwargs = {})
#   %sum_28 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_29,), kwargs = {})
#   %add_30 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_28, 1e-08), kwargs = {})
#   %reciprocal_27 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_30,), kwargs = {})
#   %mul_58 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_27, 2.0), kwargs = {})
#   %mul_59 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_30), kwargs = {})
#   return %sum_28,%reciprocal_27,%mul_59
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1728 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\xe\cxeocjd4kmv6724uhntixgtgfhhbfm7aul4livvc6t4bqz4mvyz4.py
# Topologically Sorted Source Nodes: [getitem_31, v_30, pow_29, sum_29, v_norm_sq_28, truediv_28, mul_29], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_31 => select_28
#   mul_29 => mul_61
#   pow_29 => pow_30
#   sum_29 => sum_29
#   truediv_28 => mul_60, reciprocal_28
#   v_30 => unsqueeze_31
#   v_norm_sq_28 => add_31
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_29 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_29]
#   %reciprocal_28 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_28]
#   %select_28 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 28), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_28, 1), kwargs = {})
#   %pow_30 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_31, 2), kwargs = {})
#   %sum_29 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_30,), kwargs = {})
#   %add_31 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_29, 1e-08), kwargs = {})
#   %reciprocal_28 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_31,), kwargs = {})
#   %mul_60 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_28, 2.0), kwargs = {})
#   %mul_61 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %unsqueeze_31), kwargs = {})
#   return %sum_29,%reciprocal_28,%mul_61
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1792 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\5q\c5qh6gnx44ffr3wijpl4pxy2pz7yehjbfoi7zdorki2ebseyjsky.py
# Topologically Sorted Source Nodes: [getitem_32, v_31, pow_30, sum_30, v_norm_sq_29, truediv_29, mul_30], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_32 => select_29
#   mul_30 => mul_63
#   pow_30 => pow_31
#   sum_30 => sum_30
#   truediv_29 => mul_62, reciprocal_29
#   v_31 => unsqueeze_32
#   v_norm_sq_29 => add_32
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_30 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_30]
#   %reciprocal_29 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_29]
#   %select_29 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 29), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_29, 1), kwargs = {})
#   %pow_31 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_32, 2), kwargs = {})
#   %sum_30 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_31,), kwargs = {})
#   %add_32 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_30, 1e-08), kwargs = {})
#   %reciprocal_29 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_32,), kwargs = {})
#   %mul_62 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_29, 2.0), kwargs = {})
#   %mul_63 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_62, %unsqueeze_32), kwargs = {})
#   return %sum_30,%reciprocal_29,%mul_63
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1856 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\t6\ct6to2c5ynen7zqhpsbp5judl3hxf5ao27hc377agq3e5trhhfi7.py
# Topologically Sorted Source Nodes: [getitem_33, v_32, pow_31, sum_31, v_norm_sq_30, truediv_30, mul_31], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_33 => select_30
#   mul_31 => mul_65
#   pow_31 => pow_32
#   sum_31 => sum_31
#   truediv_30 => mul_64, reciprocal_30
#   v_32 => unsqueeze_33
#   v_norm_sq_30 => add_33
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_31 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_31]
#   %reciprocal_30 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_30]
#   %select_30 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 30), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_30, 1), kwargs = {})
#   %pow_32 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_33, 2), kwargs = {})
#   %sum_31 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_32,), kwargs = {})
#   %add_33 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_31, 1e-08), kwargs = {})
#   %reciprocal_30 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_33,), kwargs = {})
#   %mul_64 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_30, 2.0), kwargs = {})
#   %mul_65 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_33), kwargs = {})
#   return %sum_31,%reciprocal_30,%mul_65
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1920 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\7a\c7aur54qw3tz3yc7facbj5fzxxrg2emecihsmdczxfo2mzjxylli.py
# Topologically Sorted Source Nodes: [getitem_34, v_33, pow_32, sum_32, v_norm_sq_31, truediv_31, mul_32], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   getitem_34 => select_31
#   mul_32 => mul_67
#   pow_32 => pow_33
#   sum_32 => sum_32
#   truediv_31 => mul_66, reciprocal_31
#   v_33 => unsqueeze_34
#   v_norm_sq_31 => add_34
# Graph fragment:
#   %primals_11 : Tensor "f32[32, 64][64, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %sum_32 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_32]
#   %reciprocal_31 : Tensor "f32[][]cuda:0" = PlaceHolder[target=reciprocal_31]
#   %select_31 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_11, 0, 31), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_31, 1), kwargs = {})
#   %pow_33 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%unsqueeze_34, 2), kwargs = {})
#   %sum_32 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_33,), kwargs = {})
#   %add_34 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_32, 1e-08), kwargs = {})
#   %reciprocal_31 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_34,), kwargs = {})
#   %mul_66 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_31, 2.0), kwargs = {})
#   %mul_67 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %unsqueeze_34), kwargs = {})
#   return %sum_32,%reciprocal_31,%mul_67
triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37 = async_compile.triton('triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 768}}
)
@triton.jit
def triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1984 + r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 * tmp0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp11, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\dh\cdhlu4fe72na2guvi7l3pwf2tizooo2ysyycqmotd7zhnfy3vlrd.py
# Topologically Sorted Source Nodes: [linear_2, chunk, view, q_1, q_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   linear_2 => view_6
#   q_1 => permute_4
#   q_2 => clone_1
#   view => view_7
# Graph fragment:
#   %mm : Tensor "f32[16384, 768][768, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_6 : Tensor "f32[256, 64, 768][49152, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [256, 64, 768]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_6, 256, -1), kwargs = {})
#   %view_7 : Tensor "f32[256, 64, 4, 64][49152, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [256, 64, 4, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[256, 4, 64, 64][49152, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_clone_split_transpose_view_38 = async_compile.triton('triton_poi_fused__unsafe_view_clone_split_transpose_view_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_split_transpose_view_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_split_transpose_view_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 4)
    x3 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 768*x1 + 49152*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\5x\c5xsmg2k6wnuj2z2g372hvhkkarfdvzyseb35sz2kuhbj4ein6oi.py
# Topologically Sorted Source Nodes: [linear_2, chunk, view_1, k_1, k_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   k_1 => permute_5
#   k_2 => clone_2
#   linear_2 => view_6
#   view_1 => view_8
# Graph fragment:
#   %mm : Tensor "f32[16384, 768][768, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_6 : Tensor "f32[256, 64, 768][49152, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [256, 64, 768]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_6, 256, -1), kwargs = {})
#   %view_8 : Tensor "f32[256, 64, 4, 64][49152, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [256, 64, 4, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[256, 4, 64, 64][49152, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused__unsafe_view_clone_split_transpose_view_39 = async_compile.triton('triton_poi_fused__unsafe_view_clone_split_transpose_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_split_transpose_view_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_split_transpose_view_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 4)
    x3 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 64*x2 + 768*x1 + 49152*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\xn\cxnvqbwbqazralcnykxq5rtlkgm2qypzevm3ulhtbhnvcmfqec4i.py
# Topologically Sorted Source Nodes: [inv_freq_scaled, vals, f, vals_1, f_1, vals_2, f_2, full_freqs, pad_1, full_freqs_1], Original ATen: [aten.div, aten.select, aten.view, aten.mul, aten.cat, aten.zeros]
# Source node to ATen node mapping:
#   f => mul_132, view_14
#   f_1 => mul_133, view_15
#   f_2 => mul_134, view_16
#   full_freqs => cat_2
#   full_freqs_1 => cat_3
#   inv_freq_scaled => div
#   pad_1 => full_default_4
#   vals => select_64
#   vals_1 => select_65
#   vals_2 => select_66
# Graph fragment:
#   %primals_13 : Tensor "f32[64, 3][3, 1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_12 : Tensor "f32[5][1]cuda:0" = PlaceHolder[target=primals_12]
#   %div : Tensor "f32[5][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_12, 1.0), kwargs = {})
#   %select_64 : Tensor "f32[64][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_13, 1, 0), kwargs = {})
#   %view_14 : Tensor "f32[64, 1][3, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_64, [64, 1]), kwargs = {})
#   %mul_132 : Tensor "f32[64, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %div), kwargs = {})
#   %select_65 : Tensor "f32[64][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_13, 1, 1), kwargs = {})
#   %view_15 : Tensor "f32[64, 1][3, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_65, [64, 1]), kwargs = {})
#   %mul_133 : Tensor "f32[64, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %div), kwargs = {})
#   %select_66 : Tensor "f32[64][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_13, 1, 2), kwargs = {})
#   %view_16 : Tensor "f32[64, 1][3, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_66, [64, 1]), kwargs = {})
#   %mul_134 : Tensor "f32[64, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, %div), kwargs = {})
#   %cat_2 : Tensor "f32[64, 15][15, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_132, %mul_133, %mul_134], -1), kwargs = {})
#   %full_default_4 : Tensor "f32[64, 17][17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([64, 17], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat_3 : Tensor "f32[64, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %full_default_4], -1), kwargs = {})
#   return %cat_3
triton_poi_fused_cat_div_mul_select_view_zeros_40 = async_compile.triton('triton_poi_fused_cat_div_mul_select_view_zeros_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_div_mul_select_view_zeros_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_div_mul_select_view_zeros_40(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 15, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 5, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (3*x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x0), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tmp5 >= tmp8
    tmp19 = tl.full([1], 10, tl.int64)
    tmp20 = tmp5 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr0 + (1 + 3*x1), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + ((-5) + (x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp22, tmp27, tmp28)
    tmp30 = tmp5 >= tmp19
    tmp31 = tl.full([1], 15, tl.int64)
    tmp32 = tmp5 < tmp31
    tmp33 = tmp30 & tmp4
    tmp34 = tl.load(in_ptr0 + (2 + 3*x1), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr1 + ((-10) + (x0)), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp34 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp33, tmp38, tmp39)
    tmp41 = tl.where(tmp21, tmp29, tmp40)
    tmp42 = tl.where(tmp9, tmp17, tmp41)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp4, tmp42, tmp43)
    tmp45 = tmp0 >= tmp3
    tmp46 = tl.full([1], 32, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = 0.0
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.where(tmp4, tmp44, tmp50)
    tl.store(out_ptr0 + (x2), tmp51, xmask)
''', device_str='cuda')


# kernel path: ./.inductor_cache\it\citpavozrngqkvzfm7i2zssqwghcg3p4u5derciogiofoxj7qjrj.py
# Topologically Sorted Source Nodes: [q_2, emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, chunk_1, mul_65, neg, cat_5, mul_66, q_rot], Original ATen: [aten._unsafe_view, aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten.split, aten.mul, aten.neg, aten.add]
# Source node to ATen node mapping:
#   cat_5 => cat_4
#   chunk_1 => split_1
#   cos_1 => cos_1
#   cos_2 => unsqueeze_70
#   emb => clone_3, expand_1, unsqueeze_68, view_17
#   mul_65 => mul_135
#   mul_66 => mul_136
#   neg => neg
#   q_2 => view_11
#   q_rot => add_67
#   sin_1 => sin_1
#   sin_2 => unsqueeze_72
#   unsqueeze_66 => unsqueeze_69
#   unsqueeze_68 => unsqueeze_71
# Graph fragment:
#   %mm_65 : Tensor "f32[65536, 64][64, 1]cuda:0" = PlaceHolder[target=mm_65]
#   %cat_3 : Tensor "f32[64, 32][32, 1]cuda:0" = PlaceHolder[target=cat_3]
#   %view_11 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_65, [256, 4, 64, 64]), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[64, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_3, 1), kwargs = {})
#   %expand_1 : Tensor "f32[64, 2, 32][32, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_68, [64, 2, 32]), kwargs = {})
#   %clone_3 : Tensor "f32[64, 2, 32][64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_17 : Tensor "f32[64, 64][64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [64, 64]), kwargs = {})
#   %cos_1 : Tensor "f32[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_17,), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[1, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cos_1, 0), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_69, 1), kwargs = {})
#   %sin_1 : Tensor "f32[64, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_17,), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[1, 64, 64][4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sin_1, 0), kwargs = {})
#   %unsqueeze_72 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_71, 1), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_11, 32, -1), kwargs = {})
#   %mul_135 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %unsqueeze_70), kwargs = {})
#   %neg : Tensor "f32[256, 4, 64, 32][8192, 2048, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%getitem_6,), kwargs = {})
#   %cat_4 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %getitem_5], -1), kwargs = {})
#   %mul_136 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_4, %unsqueeze_72), kwargs = {})
#   %add_67 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_135, %mul_136), kwargs = {})
#   return %add_67
triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41 = async_compile.triton('triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83902464}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x4 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (32*x1 + ((x0 % 32))), None, eviction_policy='evict_last')
    tmp2 = tl_math.cos(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 32, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tl.load(in_ptr0 + (32 + 64*x4 + (x0)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp8, tmp10, tmp11)
    tmp13 = tmp4 >= tmp7
    tmp14 = tl.full([1], 64, tl.int64)
    tmp15 = tmp4 < tmp14
    tmp16 = tl.load(in_ptr0 + (64*x4 + ((-32) + x0)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp8, tmp12, tmp16)
    tmp18 = tl_math.sin(tmp1)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp3 + tmp19
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\em\cemzzq4ezjpikg4bedcjta7grhnrpwkg7hrmbdm2snbac6ttwmrn.py
# Topologically Sorted Source Nodes: [linear_2, chunk, view_2, v_1, q_3, k_3, flex_attention], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   chunk => split
#   flex_attention => flex_attention
#   k_3 => view_21
#   linear_2 => view_6
#   q_3 => view_19
#   v_1 => permute_6
#   view_2 => view_9
# Graph fragment:
#   %mm_195 : Tensor "f32[65536, 64][64, 1]cuda:0" = PlaceHolder[target=mm_195]
#   %mm_260 : Tensor "f32[65536, 64][64, 1]cuda:0" = PlaceHolder[target=mm_260]
#   %mm : Tensor "f32[16384, 768][768, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf214 : Tensor "f32[256, 1, 4, 64][256, 256, 64, 1]cuda:0" = PlaceHolder[target=buf214]
#   %buf215 : Tensor "f32[256, 1, 4, 64][256, 256, 64, 1]cuda:0" = PlaceHolder[target=buf215]
#   %primals_15 : Tensor "i32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=primals_15]
#   %primals_14 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=primals_14]
#   %primals_20 : Tensor "i32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=primals_20]
#   %primals_21 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=primals_21]
#   %primals_16 : Tensor "i32[16384][1]cuda:0" = PlaceHolder[target=primals_16]
#   %primals_17 : Tensor "b8[16384][1]cuda:0" = PlaceHolder[target=primals_17]
#   %primals_18 : Tensor "f32[16384, 2][2, 1]cuda:0" = PlaceHolder[target=primals_18]
#   %primals_19 : Tensor "f32[][]cuda:0" = PlaceHolder[target=primals_19]
#   %view_6 : Tensor "f32[256, 64, 768][49152, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [256, 64, 768]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_6, 256, -1), kwargs = {})
#   %view_9 : Tensor "f32[256, 64, 4, 64][49152, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [256, 64, 4, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[256, 4, 64, 64][49152, 64, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 2, 1, 3]), kwargs = {})
#   %view_19 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_195, [256, 4, 64, 64]), kwargs = {})
#   %view_21 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_260, [256, 4, 64, 64]), kwargs = {})
#   %flex_attention : [num_users=2] = call_function[target=torch.ops.higher_order.flex_attention](args = (%view_19, %view_21, %permute_6, %sdpa_score0, (64, 64, %primals_15, %primals_14, %primals_20, %primals_21, %primals_22, %primals_23, %primals_24, %primals_25, 128, 128, %sdpa_mask0), 0.125, {PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_16, %primals_17, %primals_18, %primals_19)), kwargs = {})
#   return %buf216
triton_tem_fused__unsafe_view_split_transpose_view_42 = async_compile.triton('triton_tem_fused__unsafe_view_split_transpose_view_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=1,
num_warps=2,
triton_meta={'signature': {'arg_Q': '*fp32', 'arg_K': '*fp32', 'arg_V': '*fp32', 'arg_M': '*fp32', 'arg_L': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'in_ptr9': '*i32', 'in_ptr10': '*i1', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_split_transpose_view_42', 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False, 'FLOAT32_PRECISION': "'tf32'", 'IS_DIVISIBLE': False, 'GQA_SHARED_HEADS': 1, 'HAS_FULL_BLOCKS': True, 'SM_SCALE': 0.125, 'SPLIT_KV': 1, 'QK_HEAD_DIM': 64, 'QK_HEAD_DIM_ROUNDED': 64, 'V_HEAD_DIM': 64, 'V_HEAD_DIM_ROUNDED': 64, 'SAFE_HEAD_DIM': True, 'BLOCK_M': 64, 'SAFE_M_BOUNDARY': True, 'SAFE_N_BOUNDARY': True, 'BLOCK_N': 64, 'SPARSE_KV_BLOCK_SIZE': 128, 'USE_TMA': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_split_transpose_view_42(arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0):
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    SPLIT_KV : tl.constexpr = 1
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    SAFE_M_BOUNDARY : tl.constexpr = True
    SAFE_N_BOUNDARY : tl.constexpr = True
    BLOCK_N : tl.constexpr = 64
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    USE_TMA : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    Q = arg_Q
    K = arg_K
    V = arg_V + 512
    M = arg_M
    L = arg_L
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX

    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # reduction buffers: M rowmax across local KV split, L local sumexp across local KV split
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # BLOCK_M, QK_HEAD_DIM: M, and D dimemsion are always assigned to the same block
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head t: Number of kv splits
    # (Modifiable) Config options:
    # SPLIT_KV: number of blocks K & V are split into
    # TILE_KV: length of each local KV split
    # BLOCK_M: block size that Q is padded along seqlen dim.
    # BLOCK_N: block size of K & V along N dimension.
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # SAFE_M_BOUNDARY: Is Q seqlen a multiple of BLOCK_M? If so, we can skip an extra boundary check for loading query.
    # SAFE_N_BOUNDARY: Is KV seqlen a multiple of BLOCK_N? If so, we can skip an extra boundary check for loading key/value.

    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base.
    #
    # SPARSE_KV_BLOCK_SIZE: sparse mask block size along KV seqlen dim.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    #
    #
    # Output: ACC output accumulated across local KV split.

    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define Q Strides
    stride_qz, stride_qh, stride_qg, stride_qm, stride_qk = 16384, 4096, 4096, 64, 1
    stride_kz, stride_kh, stride_kn, stride_kk = 16384, 4096, 64, 1
    stride_vz, stride_vh, stride_vn, stride_vk = 49152, 64, 768, 1
    stride_mz, stride_mt, stride_mh, stride_mm = 256, 256, 64, 1
    stride_lz, stride_lt, stride_lh, stride_lm = 256, 256, 64, 1


    Z = 256
    ZKV = 256
    HKV = 4
    G: tl.constexpr = GQA_SHARED_HEADS
    HQ = HKV * G
    Q_LEN = 64
    KV_LEN = 64

    MATMUL_PRECISION = Q.dtype.element_ty

    # Make sure each split is a multiple of BLOCK_N
    TILE_KV_OG = tl.cdiv(KV_LEN, SPLIT_KV)
    TILE_KV = tl.cdiv(TILE_KV_OG, BLOCK_N) * BLOCK_N
    TILE_KV_MULTIPLE: tl.constexpr = (TILE_KV // BLOCK_N)

    off_z = tl.program_id(0).to(INDEX_DTYPE) // HKV
    off_zkv = off_z % ZKV
    off_hkv = tl.program_id(0).to(INDEX_DTYPE) % HKV
    off_t = tl.program_id(1).to(INDEX_DTYPE)

    q_offset = off_z * stride_qz + off_hkv * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    K = K + k_offset
    V = V + v_offset

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_z % SPARSE_Z
    sparse_idx_h = off_hkv % SPARSE_HQ

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    SPARSE_KV_BLOCK_CNT = tl.cdiv(KV_LEN, SPARSE_KV_BLOCK_SIZE)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    # initialize offsets
    tl.device_assert(BLOCK_M % G == 0)
    BLOCK_M_PER_HQ: tl.constexpr = BLOCK_M // G
    off_g = tl.arange(0, G)                                                 # [G]
    offs_g = tl.ravel(tl.broadcast_to(off_g[:, None], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_hq = offs_g + off_hkv * G
    off_m = tl.arange(0, BLOCK_M_PER_HQ)                                    # [BLOCK_M_PER_HQ]
    offs_m = tl.ravel(tl.broadcast_to(off_m[None, :], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_d = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_vd = tl.arange(0, V_HEAD_DIM_ROUNDED)

    # Get HZ offsets for KV_NUM_BLKS and KV_IDX
    stride_block_z, stride_block_h, stride_block_row = 1, 1, 1
    sparse_block_hz_offset = sparse_idx_z * stride_block_z + sparse_idx_h * stride_block_h
    stride_kv_z, stride_kv_h, stride_kv_row, stride_kv_col = 1, 1, 1, 1
    sparse_idx_hz_offset = sparse_idx_z * stride_kv_z + sparse_idx_h * stride_kv_h

    # Calculate KV blocks that belong this CTA.
    block_n_start = off_t * TILE_KV_MULTIPLE                        # n_offset inside sparse block
    block_n_end = block_n_start + TILE_KV_MULTIPLE                  # end BLOCK_N

    q_range = stride_qg * off_g[:, None, None] + stride_qm * off_m[None, :, None] + stride_qk * offs_d[None, None, :]

    if not SAFE_M_BOUNDARY and not SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=(offs_d[None, None, :] < QK_HEAD_DIM) & (off_m[None, :, None] < Q_LEN))
    elif SAFE_M_BOUNDARY and not SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=offs_d[None, None, :] < QK_HEAD_DIM)
    elif not SAFE_M_BOUNDARY and SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=off_m[None, :, None] < Q_LEN)
    else:
        q = tl.load(Q + q_offset + q_range)

    q = tl.reshape(q, [BLOCK_M, QK_HEAD_DIM_ROUNDED])


    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # find first kv block we are loading and the number of blocks we are loading
    # Offset the kv_indices tensor by the correct batch and head
    kv_indices = KV_IDX + sparse_idx_hz_offset
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_block_hz_offset)
    MAX_KV_IDX = 1
    indices_idx = (block_n_start // SPARSE_KV_MULTIPLE) % (MAX_KV_IDX)
    off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
    off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N
    # first kv block we're loading

    # last valid block according to sparse mask
    block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

    offs_n = tl.arange(0, BLOCK_N) + off_n

    desc_k = None
    desc_v = None

    acc, l_i, m_i = forward_inner(
        arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0,
        q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
        # accumulatd values
        acc, l_i, m_i,
        #offsets
        off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
        off_n,
        #block sparse data
        kv_indices, kv_num_blocks,
        block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
        MATMUL_PRECISION,
        stride_kk, stride_kn, stride_vn, stride_vk,
        IS_FULL_BLOCKS=False,
    )


    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        kv_indices = FULL_KV_IDX + sparse_idx_hz_offset
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_block_hz_offset)
        # Assign full block in a reverse order for off_t. Prioritize the last CTA.
        block_n_start = (SPLIT_KV - off_t - 1) * TILE_KV_MULTIPLE
        block_n_end = block_n_start + TILE_KV_MULTIPLE
        indices_idx = (block_n_start // SPARSE_KV_MULTIPLE) % (MAX_KV_IDX)
        off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
        off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N

        # last valid block according to sparse mask
        block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

        offs_n = tl.arange(0, BLOCK_N) + off_n

        acc, l_i, m_i = forward_inner(
            arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0,
            q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
            # accumulatd values
            acc, l_i, m_i,
            #offsets
            off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
            off_n,
            #block sparse data
            kv_indices, kv_num_blocks,
            block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
            MATMUL_PRECISION,
            stride_kk, stride_kn, stride_vn, stride_vk,
            IS_FULL_BLOCKS=True,
        )

    m_offset = off_t * stride_mt + off_z * stride_mz
    l_offset = off_t * stride_lt + off_z * stride_lz

    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_mh, stride_mm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + l_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_lh, stride_lm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )

    # Store output, logsumexp and rowmax for cross CTA reduction. (all in float32, even when input data are in fp16)
    m_i = m_i.reshape(G, BLOCK_M_PER_HQ)
    l_i = l_i.reshape(G, BLOCK_M_PER_HQ)
    if SAFE_M_BOUNDARY:
        tl.store(M_block_ptr, m_i)
        tl.store(L_block_ptr, l_i)
    else:
        tl.store(M_block_ptr, m_i, boundary_check=(1,))
        tl.store(L_block_ptr, l_i, boundary_check=(1,))

    # -- store output
    idx_z = off_z
    idx_t = off_t
    idx_hq = off_hkv*G + off_g[:, None, None]
    idx_m = off_m[None, :, None]
    idx_d = offs_vd[None, None, :]

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)
    acc = acc.reshape(G, BLOCK_M_PER_HQ, V_HEAD_DIM)
    xindex = idx_d + 64*idx_m + 4096*idx_hq + 16384*idx_t + 16384*idx_z
    tl.store(out_ptr0 + (tl.broadcast_to(idx_d + 64*idx_m + 4096*idx_hq + 16384*idx_z, acc.shape)), acc, mask)


# Utility triton funcs
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_LEN: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_LEN), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_LEN), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)


# Common Imports
@triton.jit
def forward_block_mn(
    arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0,
    q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    kv_offset,
    MATMUL_PRECISION, RCP_LN2,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    SPLIT_KV : tl.constexpr = 1
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    SAFE_M_BOUNDARY : tl.constexpr = True
    SAFE_N_BOUNDARY : tl.constexpr = True
    BLOCK_N : tl.constexpr = 64
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    USE_TMA : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32


    # -- load k --
    # NB reversed order to since K is transposed
    kv_base_offset = kv_start + kv_offset

    # Load K as [BLOCK_N, QK_HEAD_DIM_ROUNDED] then transpose to [QK_HEAD_DIM_ROUNDED, BLOCK_N]
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_n_load = kv_base_offset + tl.arange(0, BLOCK_N)
    k = load_checked_2d(K, offs_n_load, offs_k, stride_kn, stride_kk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)

    k = tl.trans(k)
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
    # which is larger than the actual number of elements. To avoid access memory out of bound,
    # we need to mask out the elements that are out of Q_LEN & KV_LEN.
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    tmp0 = (qk)
    post_mod_scores = tmp0


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = (m)
        tmp2 = tl.load(in_ptr9 + tmp1)
        tmp3 = (n)
        tmp4 = tl.load(in_ptr9 + tmp3)
        tmp5 = tmp2 > tmp4
        tmp6 = tmp2 == tmp4
        tmp7 = tl.load(in_ptr10 + tmp1)
        tmp8 = tmp7 == 0
        tmp9 = tmp1 >= tmp3
        tmp10 = tmp8 | tmp9
        tmp11 = tmp6 & tmp10
        tmp12 = tl.load(in_ptr11 + 2*tmp1)
        tmp13 = tl.load(in_ptr11 + 2*tmp3)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp14 * tmp14
        tmp16 = tl.load(in_ptr11 + 1 + 2*tmp1)
        tmp17 = tl.load(in_ptr11 + 1 + 2*tmp3)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp18 * tmp18
        tmp20 = tmp15 + tmp19
        tmp21 = tl.load(in_ptr12 + 0)
        tmp22 = tmp20 < tmp21
        tmp23 = tmp11 & tmp22
        tmp24 = tmp5 | tmp23
        mask_mod_output = tmp24


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    # Calculate offsets for V loading - reuse kv_base_offset from K loading
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)
    v = load_checked_2d(V, offs_n_load, offs_v, stride_vn, stride_vk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

@triton.jit
def forward_inner(
    arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0,
    q, K, V,
    desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    SPLIT_KV : tl.constexpr = 1
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    SAFE_M_BOUNDARY : tl.constexpr = True
    SAFE_N_BOUNDARY : tl.constexpr = True
    BLOCK_N : tl.constexpr = 64
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    USE_TMA : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32


    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    kv_offset = 0

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        # Here IS_DIVISIBLE acts are the start_n = tl.multiple_of(start_n, BLOCK_N) from triton_fused_attention.
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )



        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        offs_n = offs_n + offset
        kv_offset += offset


    return acc, l_i, m_i
''', device_str='cuda')


# kernel path: ./.inductor_cache\sa\csaed3tkxmw7vfnisxwvq65v2yvnkekjjqwaxxnuzov6eedjaxzg.py
# Topologically Sorted Source Nodes: [linear_2, chunk, view_2, v_1, q_3, k_3, flex_attention], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   chunk => split
#   flex_attention => flex_attention, getitem_9
#   k_3 => view_21
#   linear_2 => view_6
#   q_3 => view_19
#   v_1 => permute_6
#   view_2 => view_9
# Graph fragment:
#   %buf216 : Tensor "f32[256, 1, 4, 64, 64][16384, 16384, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf216]
#   %buf217 : Tensor  = PlaceHolder[target=buf217]
#   %buf218 : Tensor  = PlaceHolder[target=buf218]
#   %view_6 : Tensor "f32[256, 64, 768][49152, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [256, 64, 768]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_6, 256, -1), kwargs = {})
#   %view_9 : Tensor "f32[256, 64, 4, 64][49152, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [256, 64, 4, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[256, 4, 64, 64][49152, 64, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 2, 1, 3]), kwargs = {})
#   %view_19 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_195, [256, 4, 64, 64]), kwargs = {})
#   %view_21 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_260, [256, 4, 64, 64]), kwargs = {})
#   %flex_attention : [num_users=2] = call_function[target=torch.ops.higher_order.flex_attention](args = (%view_19, %view_21, %permute_6, %sdpa_score0, (64, 64, %primals_15, %primals_14, %primals_20, %primals_21, %primals_22, %primals_23, %primals_24, %primals_25, 128, 128, %sdpa_mask0), 0.125, {PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_16, %primals_17, %primals_18, %primals_19)), kwargs = {})
#   %getitem_9 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=operator.getitem](args = (%flex_attention, 0), kwargs = {})
#   return %getitem_9
triton_poi_fused__unsafe_view_split_transpose_view_43 = async_compile.triton('triton_poi_fused__unsafe_view_split_transpose_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_split_transpose_view_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_split_transpose_view_43(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = float("-inf")
    tmp3 = tmp1 == tmp2
    tmp4 = tmp1 - tmp1
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp5, tmp4)
    tmp7 = libdevice.exp2(tmp6)
    tmp8 = tmp0 * tmp7
    tmp10 = tmp9 * tmp7
    tmp11 = 1.0
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp13 = (tmp8 / tmp12)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\py\cpyo4wuplzdewsq4k7jexjhncg6m2rf57gvb4gkdfi2bikwqluhu.py
# Topologically Sorted Source Nodes: [linear_2, chunk, view_2, v_1, q_3, k_3, flex_attention], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   chunk => split
#   flex_attention => flex_attention, getitem_10
#   k_3 => view_21
#   linear_2 => view_6
#   q_3 => view_19
#   v_1 => permute_6
#   view_2 => view_9
# Graph fragment:
#   %buf217 : Tensor  = PlaceHolder[target=buf217]
#   %buf218 : Tensor  = PlaceHolder[target=buf218]
#   %view_6 : Tensor "f32[256, 64, 768][49152, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [256, 64, 768]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_6, 256, -1), kwargs = {})
#   %view_9 : Tensor "f32[256, 64, 4, 64][49152, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [256, 64, 4, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[256, 4, 64, 64][49152, 64, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 2, 1, 3]), kwargs = {})
#   %view_19 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_195, [256, 4, 64, 64]), kwargs = {})
#   %view_21 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_260, [256, 4, 64, 64]), kwargs = {})
#   %flex_attention : [num_users=2] = call_function[target=torch.ops.higher_order.flex_attention](args = (%view_19, %view_21, %permute_6, %sdpa_score0, (64, 64, %primals_15, %primals_14, %primals_20, %primals_21, %primals_22, %primals_23, %primals_24, %primals_25, 128, 128, %sdpa_mask0), 0.125, {PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_16, %primals_17, %primals_18, %primals_19)), kwargs = {})
#   %getitem_10 : Tensor "f32[256, 4, 64][256, 64, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%flex_attention, 1), kwargs = {})
#   return %getitem_10
triton_poi_fused__unsafe_view_split_transpose_view_44 = async_compile.triton('triton_poi_fused__unsafe_view_split_transpose_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_split_transpose_view_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_split_transpose_view_44(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp1 = float("-inf")
    tmp2 = tmp0 == tmp1
    tmp4 = tmp0 - tmp0
    tmp5 = 0.0
    tmp6 = tl.where(tmp2, tmp5, tmp4)
    tmp7 = libdevice.exp2(tmp6)
    tmp8 = tmp3 * tmp7
    tmp9 = 1.0
    tmp10 = tl.where(tmp2, tmp9, tmp8)
    tmp11 = libdevice.log2(tmp10)
    tmp12 = tmp11 + tmp0
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\wm\cwmeq3tsngkbwqxen6vj2xbem6m22xu4d3nrx3zn4srzh52nmtki.py
# Topologically Sorted Source Nodes: [transpose_3, contiguous], Original ATen: [aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_5
#   transpose_3 => permute_137
# Graph fragment:
#   %getitem_9 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %permute_137 : Tensor "f32[256, 64, 4, 64][16384, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_9, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "f32[256, 64, 4, 64][16384, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_137,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_clone_transpose_45 = async_compile.triton('triton_poi_fused_clone_transpose_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 4)
    x2 = ((xindex // 256) % 64)
    x3 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 4096*x1 + 16384*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ge\cgesnqyoodsnqhkcz2xn3zos6fnaqf3jqku7gw74ynvqy7cbvkev.py
# Topologically Sorted Source Nodes: [input_2, attn_1, linear_4, gate_attn, mul_133, x_2, x_norm_1], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   attn_1 => view_24
#   gate_attn => sigmoid
#   input_2 => add_tensor_1, view_4
#   linear_4 => view_26
#   mul_133 => mul_267
#   x_2 => add_133
#   x_norm_1 => add_134, mean_1, mul_268, pow_130, rsqrt_2
# Graph fragment:
#   %mm_default_1 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_9 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_9]
#   %mm_261 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_261]
#   %addmm_2 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %buf224 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf224]
#   %rsqrt_2 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_2]
#   %add_tensor_1 : Tensor "f32[16384, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_9), kwargs = {})
#   %view_4 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [256, 64, 256]), kwargs = {})
#   %view_24 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_261, [256, 64, 256]), kwargs = {})
#   %view_26 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [256, 64, 256]), kwargs = {})
#   %sigmoid : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_26,), kwargs = {})
#   %mul_267 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_24, %sigmoid), kwargs = {})
#   %add_133 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %mul_267), kwargs = {})
#   %pow_130 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_133, 2), kwargs = {})
#   %mean_1 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_130, [2], True), kwargs = {})
#   %add_134 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_2 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_134,), kwargs = {})
#   %mul_268 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_133, %rsqrt_2), kwargs = {})
#   return %buf224,%rsqrt_2,%mul_268
triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_46 = async_compile.triton('triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 83887104}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r0_1 + 256*x0), None)
    tmp4 = tl.load(in_ptr3 + (r0_1 + 256*x0), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = 256.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1.1920928955078125e-07
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp7 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp17, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\wg\cwgjpja3acuizry2frlzq3mdy3p5ewozx6ddesklngkjw6cgpwaw.py
# Topologically Sorted Source Nodes: [x12, chunk_3, silu, mul_134], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   chunk_3 => split_3
#   mul_134 => mul_270
#   silu => mul_269, sigmoid_1
#   x12 => view_28
# Graph fragment:
#   %mm_262 : Tensor "f32[16384, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_262]
#   %view_28 : Tensor "f32[256, 64, 2048][131072, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_262, [256, 64, 2048]), kwargs = {})
#   %split_3 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_28, 1024, -1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[256, 64, 1024][65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem_12,), kwargs = {})
#   %mul_269 : Tensor "f32[256, 64, 1024][65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_12, %sigmoid_1), kwargs = {})
#   %mul_270 : Tensor "f32[256, 64, 1024][65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %getitem_13), kwargs = {})
#   return %mul_270
triton_poi_fused__unsafe_view_mul_silu_split_47 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_split_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_split_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 268435456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_split_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*x1), None)
    tmp3 = tl.load(in_ptr0 + (1024 + x0 + 2048*x1), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\yt\cythsklofdkxbtyxhjkpqbkiqzaw56wvevpon4vu3eue47j7tqve.py
# Topologically Sorted Source Nodes: [input_2, attn_1, linear_4, gate_attn, mul_133, x_2, ffn_out, x_3, x_norm_2], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   attn_1 => view_24
#   ffn_out => view_30
#   gate_attn => sigmoid
#   input_2 => add_tensor_1, view_4
#   linear_4 => view_26
#   mul_133 => mul_267
#   x_2 => add_133
#   x_3 => add_135
#   x_norm_2 => add_136, mean_2, mul_271, pow_131, rsqrt_3
# Graph fragment:
#   %mm_default_1 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_9 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_9]
#   %mm_261 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_261]
#   %addmm_2 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %mm_263 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_263]
#   %add_135 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_135]
#   %buf231 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf231]
#   %rsqrt_3 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %add_tensor_1 : Tensor "f32[16384, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_9), kwargs = {})
#   %view_4 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [256, 64, 256]), kwargs = {})
#   %view_24 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_261, [256, 64, 256]), kwargs = {})
#   %view_26 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [256, 64, 256]), kwargs = {})
#   %sigmoid : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_26,), kwargs = {})
#   %mul_267 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_24, %sigmoid), kwargs = {})
#   %add_133 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %mul_267), kwargs = {})
#   %view_30 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_263, [256, 64, 256]), kwargs = {})
#   %add_135 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %view_30), kwargs = {})
#   %pow_131 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_135, 2), kwargs = {})
#   %mean_2 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_131, [2], True), kwargs = {})
#   %add_136 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_3 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_136,), kwargs = {})
#   %mul_271 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_135, %rsqrt_3), kwargs = {})
#   return %add_135,%buf231,%rsqrt_3,%mul_271
triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_48 = async_compile.triton('triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_48', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 134218752}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_48(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r0_1 + 256*x0), None)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 256*x0), None)
    tmp8 = tl.load(in_ptr3 + (r0_1 + 256*x0), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = 256.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1.1920928955078125e-07
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp9 * tmp18
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp9, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp19, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\nx\cnxwdqqmt6dytbi5uon4bjtvmykysaex477v3xyll6voi233nlgi.py
# Topologically Sorted Source Nodes: [attn_3, linear_9, gate_attn_1, mul_267, x_4, x_norm_3], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   attn_3 => view_50
#   gate_attn_1 => sigmoid_2
#   linear_9 => view_52
#   mul_267 => mul_535
#   x_4 => add_267
#   x_norm_3 => add_268, mean_3, mul_536, pow_260, rsqrt_4
# Graph fragment:
#   %add_135 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_135]
#   %mm_525 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_525]
#   %addmm_3 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %buf445 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf445]
#   %rsqrt_4 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %view_50 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_525, [256, 64, 256]), kwargs = {})
#   %view_52 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [256, 64, 256]), kwargs = {})
#   %sigmoid_2 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_52,), kwargs = {})
#   %mul_535 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, %sigmoid_2), kwargs = {})
#   %add_267 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %mul_535), kwargs = {})
#   %pow_260 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_267, 2), kwargs = {})
#   %mean_3 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_260, [2], True), kwargs = {})
#   %add_268 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_3, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_4 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_268,), kwargs = {})
#   %mul_536 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_267, %rsqrt_4), kwargs = {})
#   return %buf445,%rsqrt_4,%mul_536
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_49 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 83886080}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), None)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 256*x0), None)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp10 = 256.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1.1920928955078125e-07
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp5 * tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp15, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\as\caspjcotfdhpocolz5z2qpbmitrfwnqoihbh6ynxxktufzrjcntl.py
# Topologically Sorted Source Nodes: [attn_3, linear_9, gate_attn_1, mul_267, x_4, ffn_out_1, x_5, x_norm_4], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   attn_3 => view_50
#   ffn_out_1 => view_56
#   gate_attn_1 => sigmoid_2
#   linear_9 => view_52
#   mul_267 => mul_535
#   x_4 => add_267
#   x_5 => add_269
#   x_norm_4 => add_270, mean_4, mul_539, pow_261, rsqrt_5
# Graph fragment:
#   %add_135 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_135]
#   %mm_525 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_525]
#   %addmm_3 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %mm_527 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_527]
#   %buf451 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf451]
#   %rsqrt_5 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_5]
#   %view_50 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_525, [256, 64, 256]), kwargs = {})
#   %view_52 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [256, 64, 256]), kwargs = {})
#   %sigmoid_2 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_52,), kwargs = {})
#   %mul_535 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, %sigmoid_2), kwargs = {})
#   %add_267 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %mul_535), kwargs = {})
#   %view_56 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_527, [256, 64, 256]), kwargs = {})
#   %add_269 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_267, %view_56), kwargs = {})
#   %pow_261 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_269, 2), kwargs = {})
#   %mean_4 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_261, [2], True), kwargs = {})
#   %add_270 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_5 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_270,), kwargs = {})
#   %mul_539 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_269, %rsqrt_5), kwargs = {})
#   return %buf451,%rsqrt_5,%mul_539
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_50 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 100663296}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), None)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 256*x0), None)
    tmp6 = tl.load(in_ptr3 + (r0_1 + 256*x0), None)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = 256.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1.1920928955078125e-07
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp7 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp17, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\c6\cc6b6tjfyxhsbiye7kstbyrz2j6ay75bw2mblca2lqasbsa3p2rt.py
# Topologically Sorted Source Nodes: [attn_3, linear_9, gate_attn_1, mul_267, x_4, ffn_out_1, x_5, attn_5, linear_14, gate_attn_2, mul_401, x_6, x_norm_5], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   attn_3 => view_50
#   attn_5 => view_76
#   ffn_out_1 => view_56
#   gate_attn_1 => sigmoid_2
#   gate_attn_2 => sigmoid_4
#   linear_14 => view_78
#   linear_9 => view_52
#   mul_267 => mul_535
#   mul_401 => mul_803
#   x_4 => add_267
#   x_5 => add_269
#   x_6 => add_401
#   x_norm_5 => add_402, mean_5, mul_804, pow_390, rsqrt_6
# Graph fragment:
#   %add_135 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_135]
#   %mm_525 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_525]
#   %addmm_3 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %mm_527 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_527]
#   %mm_789 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_789]
#   %addmm_4 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %add_401 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_401]
#   %buf666 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf666]
#   %rsqrt_6 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %view_50 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_525, [256, 64, 256]), kwargs = {})
#   %view_52 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [256, 64, 256]), kwargs = {})
#   %sigmoid_2 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_52,), kwargs = {})
#   %mul_535 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, %sigmoid_2), kwargs = {})
#   %add_267 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %mul_535), kwargs = {})
#   %view_56 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_527, [256, 64, 256]), kwargs = {})
#   %add_269 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_267, %view_56), kwargs = {})
#   %view_76 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_789, [256, 64, 256]), kwargs = {})
#   %view_78 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [256, 64, 256]), kwargs = {})
#   %sigmoid_4 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_78,), kwargs = {})
#   %mul_803 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_76, %sigmoid_4), kwargs = {})
#   %add_401 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_269, %mul_803), kwargs = {})
#   %pow_390 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_401, 2), kwargs = {})
#   %mean_5 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_390, [2], True), kwargs = {})
#   %add_402 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_5, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_6 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_402,), kwargs = {})
#   %mul_804 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_401, %rsqrt_6), kwargs = {})
#   return %add_401,%buf666,%rsqrt_6,%mul_804
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_51 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_51', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 167772160}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_51(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 256*x0), None)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 256*x0), None)
    tmp8 = tl.load(in_ptr3 + (r0_1 + 256*x0), None)
    tmp9 = tl.load(in_ptr4 + (r0_1 + 256*x0), None)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp7 = tmp5 + tmp6
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp7 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = 256.0
    tmp18 = (tmp16 / tmp17)
    tmp19 = 1.1920928955078125e-07
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp12 * tmp21
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp12, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp21, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp22, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\5e\c5e34tqfh2evbuvcwi2judxx2syw77qauqsxjn63pezk7x6er6g5.py
# Topologically Sorted Source Nodes: [ffn_out_2, x_7, x_norm_6], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   ffn_out_2 => view_82
#   x_7 => add_403
#   x_norm_6 => add_404, mean_6, mul_807, pow_391, rsqrt_7
# Graph fragment:
#   %add_401 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_401]
#   %mm_791 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_791]
#   %buf672 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf672]
#   %rsqrt_7 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_7]
#   %view_82 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_791, [256, 64, 256]), kwargs = {})
#   %add_403 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_401, %view_82), kwargs = {})
#   %pow_391 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_403, 2), kwargs = {})
#   %mean_6 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_391, [2], True), kwargs = {})
#   %add_404 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_6, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_7 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_404,), kwargs = {})
#   %mul_807 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_403, %rsqrt_7), kwargs = {})
#   return %buf672,%rsqrt_7,%mul_807
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_52 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 67108864}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_52(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = 256.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1.1920928955078125e-07
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp2 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp12, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\ok\cokx6daxp4c36uhvqkgaosec66sy7appzr5pwx6y4dyatdiymork.py
# Topologically Sorted Source Nodes: [linear_17, chunk_12, view_14, v_391, q_15, k_15, flex_attention_3], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   chunk_12 => split_12
#   flex_attention_3 => flex_attention_3
#   k_15 => view_99
#   linear_17 => view_84
#   q_15 => view_97
#   v_391 => permute_423
#   view_14 => view_87
# Graph fragment:
#   %mm_987 : Tensor "f32[65536, 64][64, 1]cuda:0" = PlaceHolder[target=mm_987]
#   %mm_1052 : Tensor "f32[65536, 64][64, 1]cuda:0" = PlaceHolder[target=mm_1052]
#   %mm_792 : Tensor "f32[16384, 768][768, 1]cuda:0" = PlaceHolder[target=mm_792]
#   %buf876 : Tensor "f32[256, 1, 4, 64][256, 256, 64, 1]cuda:0" = PlaceHolder[target=buf876]
#   %buf877 : Tensor "f32[256, 1, 4, 64][256, 256, 64, 1]cuda:0" = PlaceHolder[target=buf877]
#   %primals_48 : Tensor "i32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=primals_48]
#   %primals_47 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=primals_47]
#   %primals_49 : Tensor "i32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=primals_49]
#   %primals_50 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=primals_50]
#   %primals_16 : Tensor "i32[16384][1]cuda:0" = PlaceHolder[target=primals_16]
#   %primals_17 : Tensor "b8[16384][1]cuda:0" = PlaceHolder[target=primals_17]
#   %view_84 : Tensor "f32[256, 64, 768][49152, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_792, [256, 64, 768]), kwargs = {})
#   %split_12 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_84, 256, -1), kwargs = {})
#   %view_87 : Tensor "f32[256, 64, 4, 64][49152, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_40, [256, 64, 4, 64]), kwargs = {})
#   %permute_423 : Tensor "f32[256, 4, 64, 64][49152, 64, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_87, [0, 2, 1, 3]), kwargs = {})
#   %view_97 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_987, [256, 4, 64, 64]), kwargs = {})
#   %view_99 : Tensor "f32[256, 4, 64, 64][16384, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1052, [256, 4, 64, 64]), kwargs = {})
#   %flex_attention_3 : [num_users=2] = call_function[target=torch.ops.higher_order.flex_attention](args = (%view_97, %view_99, %permute_423, %sdpa_score3, (64, 64, %primals_48, %primals_47, %primals_49, %primals_50, %primals_51, %primals_52, %primals_53, %primals_54, 128, 128, %sdpa_mask3), 0.125, {PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_16, %primals_17)), kwargs = {})
#   return %buf878
triton_tem_fused__unsafe_view_split_transpose_view_53 = async_compile.triton('triton_tem_fused__unsafe_view_split_transpose_view_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=1,
num_warps=2,
triton_meta={'signature': {'arg_Q': '*fp32', 'arg_K': '*fp32', 'arg_V': '*fp32', 'arg_M': '*fp32', 'arg_L': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'in_ptr9': '*i32', 'in_ptr10': '*i1', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__unsafe_view_split_transpose_view_53', 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False, 'FLOAT32_PRECISION': "'tf32'", 'IS_DIVISIBLE': False, 'GQA_SHARED_HEADS': 1, 'HAS_FULL_BLOCKS': True, 'SM_SCALE': 0.125, 'SPLIT_KV': 1, 'QK_HEAD_DIM': 64, 'QK_HEAD_DIM_ROUNDED': 64, 'V_HEAD_DIM': 64, 'V_HEAD_DIM_ROUNDED': 64, 'SAFE_HEAD_DIM': True, 'BLOCK_M': 64, 'SAFE_M_BOUNDARY': True, 'SAFE_N_BOUNDARY': True, 'BLOCK_N': 64, 'SPARSE_KV_BLOCK_SIZE': 128, 'USE_TMA': False}},

)
@triton.jit
def triton_tem_fused__unsafe_view_split_transpose_view_53(arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, out_ptr0):
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    SPLIT_KV : tl.constexpr = 1
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    SAFE_M_BOUNDARY : tl.constexpr = True
    SAFE_N_BOUNDARY : tl.constexpr = True
    BLOCK_N : tl.constexpr = 64
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    USE_TMA : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    Q = arg_Q
    K = arg_K
    V = arg_V + 512
    M = arg_M
    L = arg_L
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX

    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # reduction buffers: M rowmax across local KV split, L local sumexp across local KV split
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # BLOCK_M, QK_HEAD_DIM: M, and D dimemsion are always assigned to the same block
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head t: Number of kv splits
    # (Modifiable) Config options:
    # SPLIT_KV: number of blocks K & V are split into
    # TILE_KV: length of each local KV split
    # BLOCK_M: block size that Q is padded along seqlen dim.
    # BLOCK_N: block size of K & V along N dimension.
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # SAFE_M_BOUNDARY: Is Q seqlen a multiple of BLOCK_M? If so, we can skip an extra boundary check for loading query.
    # SAFE_N_BOUNDARY: Is KV seqlen a multiple of BLOCK_N? If so, we can skip an extra boundary check for loading key/value.

    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base.
    #
    # SPARSE_KV_BLOCK_SIZE: sparse mask block size along KV seqlen dim.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    #
    #
    # Output: ACC output accumulated across local KV split.

    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define Q Strides
    stride_qz, stride_qh, stride_qg, stride_qm, stride_qk = 16384, 4096, 4096, 64, 1
    stride_kz, stride_kh, stride_kn, stride_kk = 16384, 4096, 64, 1
    stride_vz, stride_vh, stride_vn, stride_vk = 49152, 64, 768, 1
    stride_mz, stride_mt, stride_mh, stride_mm = 256, 256, 64, 1
    stride_lz, stride_lt, stride_lh, stride_lm = 256, 256, 64, 1


    Z = 256
    ZKV = 256
    HKV = 4
    G: tl.constexpr = GQA_SHARED_HEADS
    HQ = HKV * G
    Q_LEN = 64
    KV_LEN = 64

    MATMUL_PRECISION = Q.dtype.element_ty

    # Make sure each split is a multiple of BLOCK_N
    TILE_KV_OG = tl.cdiv(KV_LEN, SPLIT_KV)
    TILE_KV = tl.cdiv(TILE_KV_OG, BLOCK_N) * BLOCK_N
    TILE_KV_MULTIPLE: tl.constexpr = (TILE_KV // BLOCK_N)

    off_z = tl.program_id(0).to(INDEX_DTYPE) // HKV
    off_zkv = off_z % ZKV
    off_hkv = tl.program_id(0).to(INDEX_DTYPE) % HKV
    off_t = tl.program_id(1).to(INDEX_DTYPE)

    q_offset = off_z * stride_qz + off_hkv * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    K = K + k_offset
    V = V + v_offset

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_z % SPARSE_Z
    sparse_idx_h = off_hkv % SPARSE_HQ

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    SPARSE_KV_BLOCK_CNT = tl.cdiv(KV_LEN, SPARSE_KV_BLOCK_SIZE)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    # initialize offsets
    tl.device_assert(BLOCK_M % G == 0)
    BLOCK_M_PER_HQ: tl.constexpr = BLOCK_M // G
    off_g = tl.arange(0, G)                                                 # [G]
    offs_g = tl.ravel(tl.broadcast_to(off_g[:, None], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_hq = offs_g + off_hkv * G
    off_m = tl.arange(0, BLOCK_M_PER_HQ)                                    # [BLOCK_M_PER_HQ]
    offs_m = tl.ravel(tl.broadcast_to(off_m[None, :], [G, BLOCK_M_PER_HQ])) # [BLOCK_M]
    offs_d = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_vd = tl.arange(0, V_HEAD_DIM_ROUNDED)

    # Get HZ offsets for KV_NUM_BLKS and KV_IDX
    stride_block_z, stride_block_h, stride_block_row = 1, 1, 1
    sparse_block_hz_offset = sparse_idx_z * stride_block_z + sparse_idx_h * stride_block_h
    stride_kv_z, stride_kv_h, stride_kv_row, stride_kv_col = 1, 1, 1, 1
    sparse_idx_hz_offset = sparse_idx_z * stride_kv_z + sparse_idx_h * stride_kv_h

    # Calculate KV blocks that belong this CTA.
    block_n_start = off_t * TILE_KV_MULTIPLE                        # n_offset inside sparse block
    block_n_end = block_n_start + TILE_KV_MULTIPLE                  # end BLOCK_N

    q_range = stride_qg * off_g[:, None, None] + stride_qm * off_m[None, :, None] + stride_qk * offs_d[None, None, :]

    if not SAFE_M_BOUNDARY and not SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=(offs_d[None, None, :] < QK_HEAD_DIM) & (off_m[None, :, None] < Q_LEN))
    elif SAFE_M_BOUNDARY and not SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=offs_d[None, None, :] < QK_HEAD_DIM)
    elif not SAFE_M_BOUNDARY and SAFE_HEAD_DIM:
        q = tl.load(Q + q_offset + q_range, mask=off_m[None, :, None] < Q_LEN)
    else:
        q = tl.load(Q + q_offset + q_range)

    q = tl.reshape(q, [BLOCK_M, QK_HEAD_DIM_ROUNDED])


    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # find first kv block we are loading and the number of blocks we are loading
    # Offset the kv_indices tensor by the correct batch and head
    kv_indices = KV_IDX + sparse_idx_hz_offset
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_block_hz_offset)
    MAX_KV_IDX = 1
    indices_idx = (block_n_start // SPARSE_KV_MULTIPLE) % (MAX_KV_IDX)
    off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
    off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N
    # first kv block we're loading

    # last valid block according to sparse mask
    block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

    offs_n = tl.arange(0, BLOCK_N) + off_n

    desc_k = None
    desc_v = None

    acc, l_i, m_i = forward_inner(
        arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, out_ptr0,
        q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
        # accumulatd values
        acc, l_i, m_i,
        #offsets
        off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
        off_n,
        #block sparse data
        kv_indices, kv_num_blocks,
        block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
        MATMUL_PRECISION,
        stride_kk, stride_kn, stride_vn, stride_vk,
        IS_FULL_BLOCKS=False,
    )


    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        kv_indices = FULL_KV_IDX + sparse_idx_hz_offset
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_block_hz_offset)
        # Assign full block in a reverse order for off_t. Prioritize the last CTA.
        block_n_start = (SPLIT_KV - off_t - 1) * TILE_KV_MULTIPLE
        block_n_end = block_n_start + TILE_KV_MULTIPLE
        indices_idx = (block_n_start // SPARSE_KV_MULTIPLE) % (MAX_KV_IDX)
        off_n_block_in_sparse = block_n_start % SPARSE_KV_MULTIPLE
        off_n = tl.load(kv_indices + indices_idx) * SPARSE_KV_BLOCK_SIZE + off_n_block_in_sparse * BLOCK_N

        # last valid block according to sparse mask
        block_n_last_valid = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

        offs_n = tl.arange(0, BLOCK_N) + off_n

        acc, l_i, m_i = forward_inner(
            arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, out_ptr0,
            q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
            # accumulatd values
            acc, l_i, m_i,
            #offsets
            off_z, offs_hq[:, None], offs_m[:, None], offs_n[None, :],
            off_n,
            #block sparse data
            kv_indices, kv_num_blocks,
            block_n_start, block_n_end if block_n_end <= block_n_last_valid else block_n_last_valid,
            MATMUL_PRECISION,
            stride_kk, stride_kn, stride_vn, stride_vk,
            IS_FULL_BLOCKS=True,
        )

    m_offset = off_t * stride_mt + off_z * stride_mz
    l_offset = off_t * stride_lt + off_z * stride_lz

    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_mh, stride_mm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + l_offset,
        shape=(G, Q_LEN),                   # (G, M)
        strides=(stride_lh, stride_lm),
        offsets=(off_hkv*G, 0),
        block_shape=(G, BLOCK_M_PER_HQ),
        order=(1, 0)
    )

    # Store output, logsumexp and rowmax for cross CTA reduction. (all in float32, even when input data are in fp16)
    m_i = m_i.reshape(G, BLOCK_M_PER_HQ)
    l_i = l_i.reshape(G, BLOCK_M_PER_HQ)
    if SAFE_M_BOUNDARY:
        tl.store(M_block_ptr, m_i)
        tl.store(L_block_ptr, l_i)
    else:
        tl.store(M_block_ptr, m_i, boundary_check=(1,))
        tl.store(L_block_ptr, l_i, boundary_check=(1,))

    # -- store output
    idx_z = off_z
    idx_t = off_t
    idx_hq = off_hkv*G + off_g[:, None, None]
    idx_m = off_m[None, :, None]
    idx_d = offs_vd[None, None, :]

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)
    acc = acc.reshape(G, BLOCK_M_PER_HQ, V_HEAD_DIM)
    xindex = idx_d + 64*idx_m + 4096*idx_hq + 16384*idx_t + 16384*idx_z
    tl.store(out_ptr0 + (tl.broadcast_to(idx_d + 64*idx_m + 4096*idx_hq + 16384*idx_z, acc.shape)), acc, mask)


# Utility triton funcs
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_LEN: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_LEN), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_LEN), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)


# Common Imports
@triton.jit
def forward_block_mn(
    arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, out_ptr0,
    q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    kv_offset,
    MATMUL_PRECISION, RCP_LN2,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    SPLIT_KV : tl.constexpr = 1
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    SAFE_M_BOUNDARY : tl.constexpr = True
    SAFE_N_BOUNDARY : tl.constexpr = True
    BLOCK_N : tl.constexpr = 64
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    USE_TMA : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32


    # -- load k --
    # NB reversed order to since K is transposed
    kv_base_offset = kv_start + kv_offset

    # Load K as [BLOCK_N, QK_HEAD_DIM_ROUNDED] then transpose to [QK_HEAD_DIM_ROUNDED, BLOCK_N]
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_n_load = kv_base_offset + tl.arange(0, BLOCK_N)
    k = load_checked_2d(K, offs_n_load, offs_k, stride_kn, stride_kk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)

    k = tl.trans(k)
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
    # which is larger than the actual number of elements. To avoid access memory out of bound,
    # we need to mask out the elements that are out of Q_LEN & KV_LEN.
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    tmp0 = (qk)
    post_mod_scores = tmp0


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = (m)
        tmp2 = tl.load(in_ptr9 + tmp1)
        tmp3 = (n)
        tmp4 = tl.load(in_ptr9 + tmp3)
        tmp5 = tmp2 > tmp4
        tmp6 = tmp2 == tmp4
        tmp7 = tl.load(in_ptr10 + tmp1)
        tmp8 = tmp7 == 0
        tmp9 = tmp1 >= tmp3
        tmp10 = tmp8 | tmp9
        tmp11 = tmp6 & tmp10
        tmp12 = tmp5 | tmp11
        mask_mod_output = tmp12


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    # Calculate offsets for V loading - reuse kv_base_offset from K loading
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)
    v = load_checked_2d(V, offs_n_load, offs_v, stride_vn, stride_vk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

@triton.jit
def forward_inner(
    arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, out_ptr0,
    q, K, V,
    desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    SPLIT_KV : tl.constexpr = 1
    QK_HEAD_DIM : tl.constexpr = 64
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM_ROUNDED : tl.constexpr = 64
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    SAFE_M_BOUNDARY : tl.constexpr = True
    SAFE_N_BOUNDARY : tl.constexpr = True
    BLOCK_N : tl.constexpr = 64
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    USE_TMA : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32


    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    kv_offset = 0

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        # Here IS_DIVISIBLE acts are the start_n = tl.multiple_of(start_n, BLOCK_N) from triton_fused_attention.
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, out_ptr0,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, out_ptr0,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )



        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        offs_n = offs_n + offset
        kv_offset += offset


    return acc, l_i, m_i
''', device_str='cuda')


# kernel path: ./.inductor_cache\yr\cyra4ahu5emrbyeisvxlazke35dmodyd3ykafivcwr32xfitrrwn.py
# Topologically Sorted Source Nodes: [ffn_out_2, x_7, attn_7, linear_19, gate_attn_3, mul_535, x_8, x_norm_7], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.sigmoid, aten.mul, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   attn_7 => view_102
#   ffn_out_2 => view_82
#   gate_attn_3 => sigmoid_6
#   linear_19 => view_104
#   mul_535 => mul_1071
#   x_7 => add_403
#   x_8 => add_535
#   x_norm_7 => add_536, mean_7, mul_1072, pow_520, rsqrt_8
# Graph fragment:
#   %add_401 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_401]
#   %mm_791 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_791]
#   %mm_1053 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_1053]
#   %addmm_5 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %buf886 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf886]
#   %rsqrt_8 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_8]
#   %view_82 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_791, [256, 64, 256]), kwargs = {})
#   %add_403 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_401, %view_82), kwargs = {})
#   %view_102 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1053, [256, 64, 256]), kwargs = {})
#   %view_104 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [256, 64, 256]), kwargs = {})
#   %sigmoid_6 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_104,), kwargs = {})
#   %mul_1071 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_102, %sigmoid_6), kwargs = {})
#   %add_535 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_403, %mul_1071), kwargs = {})
#   %pow_520 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_535, 2), kwargs = {})
#   %mean_7 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_520, [2], True), kwargs = {})
#   %add_536 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_7, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_8 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_536,), kwargs = {})
#   %mul_1072 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_535, %rsqrt_8), kwargs = {})
#   return %buf886,%rsqrt_8,%mul_1072
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_54 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 100663296}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), None)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 256*x0), None)
    tmp4 = tl.load(in_ptr3 + (r0_1 + 256*x0), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = 256.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1.1920928955078125e-07
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp7 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp17, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\g7\cg7bbozea74anjys46wbdgjwlkqglknbb3vfmrsnclnqddhge2bo.py
# Topologically Sorted Source Nodes: [ffn_out_2, x_7, attn_7, linear_19, gate_attn_3, mul_535, x_8, ffn_out_3, x_9, x_10, input_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.sigmoid, aten.mul, aten.pow, aten.mean, aten.rsqrt, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_7 => view_102
#   ffn_out_2 => view_82
#   ffn_out_3 => view_108
#   gate_attn_3 => sigmoid_6
#   input_3 => add_539, add_540, mul_1076, mul_1077, rsqrt_10, sub_517, var_mean_1
#   linear_19 => view_104
#   mul_535 => mul_1071
#   x_10 => add_538, mean_8, mul_1075, pow_521, rsqrt_9
#   x_7 => add_403
#   x_8 => add_535
#   x_9 => add_537
# Graph fragment:
#   %add_401 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_401]
#   %mm_791 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_791]
#   %mm_1053 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_1053]
#   %addmm_5 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %mm_1055 : Tensor "f32[16384, 256][256, 1]cuda:0" = PlaceHolder[target=mm_1055]
#   %add_537 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=add_537]
#   %buf893 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf893]
#   %rsqrt_9 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %mul_1075 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0" = PlaceHolder[target=mul_1075]
#   %buf897 : Tensor "f32[256, 64, 1][64, 1, 16384]cuda:0" = PlaceHolder[target=buf897]
#   %getitem_51 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_51]
#   %rsqrt_10 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_10]
#   %primals_60 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_60]
#   %primals_61 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_61]
#   %view_82 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_791, [256, 64, 256]), kwargs = {})
#   %add_403 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_401, %view_82), kwargs = {})
#   %view_102 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1053, [256, 64, 256]), kwargs = {})
#   %view_104 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [256, 64, 256]), kwargs = {})
#   %sigmoid_6 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_104,), kwargs = {})
#   %mul_1071 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_102, %sigmoid_6), kwargs = {})
#   %add_535 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_403, %mul_1071), kwargs = {})
#   %view_108 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1055, [256, 64, 256]), kwargs = {})
#   %add_537 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_535, %view_108), kwargs = {})
#   %pow_521 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_537, 2), kwargs = {})
#   %mean_8 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_521, [2], True), kwargs = {})
#   %add_538 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_8, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_9 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_538,), kwargs = {})
#   %mul_1075 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_537, %rsqrt_9), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_1075, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_539 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-05), kwargs = {})
#   %rsqrt_10 : Tensor "f32[256, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_539,), kwargs = {})
#   %sub_517 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1075, %getitem_51), kwargs = {})
#   %mul_1076 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_517, %rsqrt_10), kwargs = {})
#   %mul_1077 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1076, %primals_60), kwargs = {})
#   %add_540 : Tensor "f32[256, 64, 256][16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1077, %primals_61), kwargs = {})
#   return %add_537,%buf893,%rsqrt_9,%mul_1075,%getitem_51,%buf897,%rsqrt_10,%add_540
triton_per_fused__unsafe_view_add_mean_mul_native_layer_norm_pow_rsqrt_sigmoid_view_55 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_native_layer_norm_pow_rsqrt_sigmoid_view_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_native_layer_norm_pow_rsqrt_sigmoid_view_55', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 5, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 150996992}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_native_layer_norm_pow_rsqrt_sigmoid_view_55(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 256*x0), None)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 256*x0), None)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 256*x0), None)
    tmp8 = tl.load(in_ptr3 + (r0_1 + 256*x0), None)
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = 256.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1.1920928955078125e-07
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp9 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp24 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp25 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = (tmp24 / tmp26)
    tmp28 = tmp20 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = (tmp32 / tmp14)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp19 - tmp27
    tmp38 = tmp37 * tmp36
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp19, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp36, None)
    tl.store(out_ptr1 + (r0_1 + 256*x0), tmp42, None)
    tl.store(out_ptr0 + (x0), tmp27, None)
''', device_str='cuda')


# kernel path: ./.inductor_cache\nv\cnv3sw6h46ayy5f2punv5zac5ccva7a44t4vlhr4jhmtwwcvw5st.py
# Topologically Sorted Source Nodes: [input_4, view_16, permute_1, reshape_1], Original ATen: [aten.addmm, aten.view, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   input_4 => add_tensor, view_110
#   permute_1 => permute_560
#   reshape_1 => clone_21
#   view_16 => view_111
# Graph fragment:
#   %mm_default : Tensor "f32[16384, 12][12, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %primals_63 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_63]
#   %add_tensor : Tensor "f32[16384, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_63), kwargs = {})
#   %view_110 : Tensor "f32[256, 64, 12][768, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [256, 64, 12]), kwargs = {})
#   %view_111 : Tensor "f32[256, 8, 8, 3, 2, 2][768, 96, 12, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_110, [256, 8, 8, 3, 2, 2]), kwargs = {})
#   %permute_560 : Tensor "f32[256, 3, 8, 2, 8, 2][768, 4, 96, 2, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_111, [0, 3, 1, 4, 2, 5]), kwargs = {})
#   %clone_21 : Tensor "f32[256, 3, 8, 2, 8, 2][768, 256, 32, 16, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_560,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_21
triton_poi_fused_addmm_clone_permute_view_56 = async_compile.triton('triton_poi_fused_addmm_clone_permute_view_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_view_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_view_56(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 8)
    x2 = ((xindex // 16) % 2)
    x3 = ((xindex // 32) % 8)
    x4 = ((xindex // 256) % 3)
    x5 = xindex // 768
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*x2 + 4*x4 + 12*x1 + 96*x3 + 768*x5), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 2*x2 + 4*x4), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x6), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63 = args
        args.clear()
        assert_size_stride(primals_1, (256, ), (1, ))
        assert_size_stride(primals_2, (4, ), (1, ))
        assert_size_stride(primals_3, (256, 3, 16, 16), (768, 256, 16, 1))
        assert_size_stride(primals_4, (256, 56), (56, 1))
        assert_size_stride(primals_5, (256, ), (1, ))
        assert_size_stride(primals_6, (256, ), (1, ))
        assert_size_stride(primals_7, (256, ), (1, ))
        assert_size_stride(primals_8, (256, 256), (256, 1))
        assert_size_stride(primals_9, (256, ), (1, ))
        assert_size_stride(primals_10, (768, 256), (256, 1))
        assert_size_stride(primals_11, (32, 64), (64, 1))
        assert_size_stride(primals_12, (5, ), (1, ))
        assert_size_stride(primals_13, (64, 3), (3, 1))
        assert_size_stride(primals_14, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_15, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_16, (16384, ), (1, ))
        assert_size_stride(primals_17, (16384, ), (1, ))
        assert_size_stride(primals_18, (16384, 2), (2, 1))
        assert_size_stride(primals_19, (), ())
        assert_size_stride(primals_20, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_21, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_22, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_23, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_24, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_25, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_26, (256, 256), (256, 1))
        assert_size_stride(primals_27, (256, 256), (256, 1))
        assert_size_stride(primals_28, (256, ), (1, ))
        assert_size_stride(primals_29, (2048, 256), (256, 1))
        assert_size_stride(primals_30, (256, 1024), (1024, 1))
        assert_size_stride(primals_31, (768, 256), (256, 1))
        assert_size_stride(primals_32, (32, 64), (64, 1))
        assert_size_stride(primals_33, (256, 256), (256, 1))
        assert_size_stride(primals_34, (256, 256), (256, 1))
        assert_size_stride(primals_35, (256, ), (1, ))
        assert_size_stride(primals_36, (2048, 256), (256, 1))
        assert_size_stride(primals_37, (256, 1024), (1024, 1))
        assert_size_stride(primals_38, (768, 256), (256, 1))
        assert_size_stride(primals_39, (32, 64), (64, 1))
        assert_size_stride(primals_40, (256, 256), (256, 1))
        assert_size_stride(primals_41, (256, 256), (256, 1))
        assert_size_stride(primals_42, (256, ), (1, ))
        assert_size_stride(primals_43, (2048, 256), (256, 1))
        assert_size_stride(primals_44, (256, 1024), (1024, 1))
        assert_size_stride(primals_45, (768, 256), (256, 1))
        assert_size_stride(primals_46, (32, 64), (64, 1))
        assert_size_stride(primals_47, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_48, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_49, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_50, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_51, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_52, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_53, (1, 1, 1), (1, 1, 1))
        assert_size_stride(primals_54, (1, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_55, (256, 256), (256, 1))
        assert_size_stride(primals_56, (256, 256), (256, 1))
        assert_size_stride(primals_57, (256, ), (1, ))
        assert_size_stride(primals_58, (2048, 256), (256, 1))
        assert_size_stride(primals_59, (256, 1024), (1024, 1))
        assert_size_stride(primals_60, (256, ), (1, ))
        assert_size_stride(primals_61, (256, ), (1, ))
        assert_size_stride(primals_62, (12, 256), (256, 1))
        assert_size_stride(primals_63, (12, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((256, 64, 56), (3584, 56, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, args, sin, cos, spins, x_pad, unfold, patches, permute, patches_1, unsqueeze_1, spins_expanded, cat_1], Original ATen: [aten.unsqueeze, aten.mul, aten.sin, aten.cos, aten.cat, aten.reflection_pad2d, aten.unfold, aten.permute, aten.clone, aten._unsafe_view, aten.expand]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_clone_cos_expand_mul_permute_reflection_pad2d_sin_unfold_unsqueeze_0.run(primals_3, primals_1, primals_2, buf0, 917504, stream=stream0)
            del primals_1
            del primals_2
            del primals_3
            buf1 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(primals_5, reinterpret_tensor(buf0, (16384, 56), (56, 1), 0), reinterpret_tensor(primals_4, (56, 256), (1, 56), 0), alpha=1, beta=1, out=buf1)
            del primals_4
            del primals_5
            buf2 = empty_strided_cuda((256, 64, 1), (64, 1, 1), torch.float32)
            buf3 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf5 = reinterpret_tensor(buf3, (256, 64, 1), (64, 1, 1), 0); del buf3  # reuse
            buf6 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, input_1], Original ATen: [aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_view_1.run(buf5, buf1, primals_6, primals_7, buf2, buf6, 16384, 256, stream=stream0)
            del primals_7
            buf7 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, input_1, input_2], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf6, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_8, (256, 256), (1, 256), 0), out=buf7)
            buf8 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf9 = reinterpret_tensor(buf8, (256, 64, 1), (64, 1, 1), 0); del buf8  # reuse
            buf10 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_2, x_norm], Original ATen: [aten.addmm, aten.view, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_mean_mul_pow_rsqrt_view_2.run(buf9, buf7, primals_9, buf10, 16384, 256, stream=stream0)
            buf11 = empty_strided_cuda((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf10, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_10, (256, 768), (1, 256), 0), out=buf11)
            buf12 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [Q], Original ATen: [aten.eye]
            stream0 = get_raw_stream(0)
            triton_poi_fused_eye_3.run(buf12, 4096, stream=stream0)
            buf13 = empty_strided_cuda((), (), torch.float32)
            buf14 = buf13; del buf13  # reuse
            buf15 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_3, v_2, pow_1, sum_1, v_norm_sq, truediv, mul_1], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4.run(buf14, primals_11, buf15, 1, 64, stream=stream0)
            buf16 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_3, v_2, t, matmul], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 0), buf12, out=buf16)
            buf17 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf16, out=buf17)
            buf18 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [Q_1], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_5.run(buf18, 4096, stream=stream0)
            buf19 = empty_strided_cuda((), (), torch.float32)
            buf20 = buf19; del buf19  # reuse
            buf21 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_4, v_3, pow_2, sum_2, v_norm_sq_1, truediv_1, mul_2], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6.run(buf20, primals_11, buf21, 1, 64, stream=stream0)
            buf22 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_4, v_3, t_1, matmul_2], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 64), buf18, out=buf22)
            buf23 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_1], Original ATen: [aten.mm]
            extern_kernels.mm(buf21, buf22, out=buf23)
            buf24 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [Q_2], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf24, buf18, 4096, stream=stream0)
            buf25 = empty_strided_cuda((), (), torch.float32)
            buf26 = buf25; del buf25  # reuse
            buf27 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_5, v_4, pow_3, sum_3, v_norm_sq_2, truediv_2, mul_3], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8.run(buf26, primals_11, buf27, 1, 64, stream=stream0)
            buf28 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_5, v_4, t_2, matmul_4], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 128), buf24, out=buf28)
            buf29 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_2], Original ATen: [aten.mm]
            extern_kernels.mm(buf27, buf28, out=buf29)
            buf30 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [Q_3], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf30, buf24, 4096, stream=stream0)
            buf31 = empty_strided_cuda((), (), torch.float32)
            buf32 = buf31; del buf31  # reuse
            buf33 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_6, v_5, pow_4, sum_4, v_norm_sq_3, truediv_3, mul_4], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9.run(buf32, primals_11, buf33, 1, 64, stream=stream0)
            buf34 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_6, v_5, t_3, matmul_6], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 192), buf30, out=buf34)
            buf35 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_3], Original ATen: [aten.mm]
            extern_kernels.mm(buf33, buf34, out=buf35)
            buf36 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [Q_4], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf36, buf30, 4096, stream=stream0)
            buf37 = empty_strided_cuda((), (), torch.float32)
            buf38 = buf37; del buf37  # reuse
            buf39 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_7, v_6, pow_5, sum_5, v_norm_sq_4, truediv_4, mul_5], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10.run(buf38, primals_11, buf39, 1, 64, stream=stream0)
            buf40 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_7, v_6, t_4, matmul_8], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 256), buf36, out=buf40)
            buf41 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_4], Original ATen: [aten.mm]
            extern_kernels.mm(buf39, buf40, out=buf41)
            buf42 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [Q_5], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf42, buf36, 4096, stream=stream0)
            buf43 = empty_strided_cuda((), (), torch.float32)
            buf44 = buf43; del buf43  # reuse
            buf45 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_8, v_7, pow_6, sum_6, v_norm_sq_5, truediv_5, mul_6], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11.run(buf44, primals_11, buf45, 1, 64, stream=stream0)
            buf46 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_8, v_7, t_5, matmul_10], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 320), buf42, out=buf46)
            buf47 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_5], Original ATen: [aten.mm]
            extern_kernels.mm(buf45, buf46, out=buf47)
            buf48 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [Q_6], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf48, buf42, 4096, stream=stream0)
            buf49 = empty_strided_cuda((), (), torch.float32)
            buf50 = buf49; del buf49  # reuse
            buf51 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_9, v_8, pow_7, sum_7, v_norm_sq_6, truediv_6, mul_7], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12.run(buf50, primals_11, buf51, 1, 64, stream=stream0)
            buf52 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_9, v_8, t_6, matmul_12], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 384), buf48, out=buf52)
            buf53 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_6], Original ATen: [aten.mm]
            extern_kernels.mm(buf51, buf52, out=buf53)
            buf54 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [Q_7], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf54, buf48, 4096, stream=stream0)
            buf55 = empty_strided_cuda((), (), torch.float32)
            buf56 = buf55; del buf55  # reuse
            buf57 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_10, v_9, pow_8, sum_8, v_norm_sq_7, truediv_7, mul_8], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13.run(buf56, primals_11, buf57, 1, 64, stream=stream0)
            buf58 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_10, v_9, t_7, matmul_14], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 448), buf54, out=buf58)
            buf59 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_7], Original ATen: [aten.mm]
            extern_kernels.mm(buf57, buf58, out=buf59)
            buf60 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [Q_8], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf60, buf54, 4096, stream=stream0)
            buf61 = empty_strided_cuda((), (), torch.float32)
            buf62 = buf61; del buf61  # reuse
            buf63 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, v_10, pow_9, sum_9, v_norm_sq_8, truediv_8, mul_9], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14.run(buf62, primals_11, buf63, 1, 64, stream=stream0)
            buf64 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_11, v_10, t_8, matmul_16], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 512), buf60, out=buf64)
            buf65 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_8], Original ATen: [aten.mm]
            extern_kernels.mm(buf63, buf64, out=buf65)
            buf66 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [Q_9], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf66, buf60, 4096, stream=stream0)
            buf67 = empty_strided_cuda((), (), torch.float32)
            buf68 = buf67; del buf67  # reuse
            buf69 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_12, v_11, pow_10, sum_10, v_norm_sq_9, truediv_9, mul_10], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15.run(buf68, primals_11, buf69, 1, 64, stream=stream0)
            buf70 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_12, v_11, t_9, matmul_18], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 576), buf66, out=buf70)
            buf71 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_9], Original ATen: [aten.mm]
            extern_kernels.mm(buf69, buf70, out=buf71)
            buf72 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [Q_10], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf72, buf66, 4096, stream=stream0)
            buf73 = empty_strided_cuda((), (), torch.float32)
            buf74 = buf73; del buf73  # reuse
            buf75 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, v_12, pow_11, sum_11, v_norm_sq_10, truediv_10, mul_11], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16.run(buf74, primals_11, buf75, 1, 64, stream=stream0)
            buf76 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, v_12, t_10, matmul_20], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 640), buf72, out=buf76)
            buf77 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_10], Original ATen: [aten.mm]
            extern_kernels.mm(buf75, buf76, out=buf77)
            buf78 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [Q_11], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf78, buf72, 4096, stream=stream0)
            buf79 = empty_strided_cuda((), (), torch.float32)
            buf80 = buf79; del buf79  # reuse
            buf81 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_14, v_13, pow_12, sum_12, v_norm_sq_11, truediv_11, mul_12], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17.run(buf80, primals_11, buf81, 1, 64, stream=stream0)
            buf82 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_14, v_13, t_11, matmul_22], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 704), buf78, out=buf82)
            buf83 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_11], Original ATen: [aten.mm]
            extern_kernels.mm(buf81, buf82, out=buf83)
            buf84 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [Q_12], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf84, buf78, 4096, stream=stream0)
            buf85 = empty_strided_cuda((), (), torch.float32)
            buf86 = buf85; del buf85  # reuse
            buf87 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_15, v_14, pow_13, sum_13, v_norm_sq_12, truediv_12, mul_13], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18.run(buf86, primals_11, buf87, 1, 64, stream=stream0)
            buf88 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_15, v_14, t_12, matmul_24], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 768), buf84, out=buf88)
            buf89 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_12], Original ATen: [aten.mm]
            extern_kernels.mm(buf87, buf88, out=buf89)
            buf90 = buf89; del buf89  # reuse
            # Topologically Sorted Source Nodes: [Q_13], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf90, buf84, 4096, stream=stream0)
            buf91 = empty_strided_cuda((), (), torch.float32)
            buf92 = buf91; del buf91  # reuse
            buf93 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, v_15, pow_14, sum_14, v_norm_sq_13, truediv_13, mul_14], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19.run(buf92, primals_11, buf93, 1, 64, stream=stream0)
            buf94 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, v_15, t_13, matmul_26], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 832), buf90, out=buf94)
            buf95 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_13], Original ATen: [aten.mm]
            extern_kernels.mm(buf93, buf94, out=buf95)
            buf96 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [Q_14], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf96, buf90, 4096, stream=stream0)
            buf97 = empty_strided_cuda((), (), torch.float32)
            buf98 = buf97; del buf97  # reuse
            buf99 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_17, v_16, pow_15, sum_15, v_norm_sq_14, truediv_14, mul_15], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20.run(buf98, primals_11, buf99, 1, 64, stream=stream0)
            buf100 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_17, v_16, t_14, matmul_28], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 896), buf96, out=buf100)
            buf101 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_14], Original ATen: [aten.mm]
            extern_kernels.mm(buf99, buf100, out=buf101)
            buf102 = buf101; del buf101  # reuse
            # Topologically Sorted Source Nodes: [Q_15], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf102, buf96, 4096, stream=stream0)
            buf103 = empty_strided_cuda((), (), torch.float32)
            buf104 = buf103; del buf103  # reuse
            buf105 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_18, v_17, pow_16, sum_16, v_norm_sq_15, truediv_15, mul_16], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21.run(buf104, primals_11, buf105, 1, 64, stream=stream0)
            buf106 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_18, v_17, t_15, matmul_30], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 960), buf102, out=buf106)
            buf107 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_15], Original ATen: [aten.mm]
            extern_kernels.mm(buf105, buf106, out=buf107)
            buf108 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [Q_16], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf108, buf102, 4096, stream=stream0)
            buf109 = empty_strided_cuda((), (), torch.float32)
            buf110 = buf109; del buf109  # reuse
            buf111 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_19, v_18, pow_17, sum_17, v_norm_sq_16, truediv_16, mul_17], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22.run(buf110, primals_11, buf111, 1, 64, stream=stream0)
            buf112 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_19, v_18, t_16, matmul_32], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1024), buf108, out=buf112)
            buf113 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_16], Original ATen: [aten.mm]
            extern_kernels.mm(buf111, buf112, out=buf113)
            buf114 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [Q_17], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf114, buf108, 4096, stream=stream0)
            buf115 = empty_strided_cuda((), (), torch.float32)
            buf116 = buf115; del buf115  # reuse
            buf117 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_20, v_19, pow_18, sum_18, v_norm_sq_17, truediv_17, mul_18], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23.run(buf116, primals_11, buf117, 1, 64, stream=stream0)
            buf118 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_20, v_19, t_17, matmul_34], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1088), buf114, out=buf118)
            buf119 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_17], Original ATen: [aten.mm]
            extern_kernels.mm(buf117, buf118, out=buf119)
            buf120 = buf119; del buf119  # reuse
            # Topologically Sorted Source Nodes: [Q_18], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf120, buf114, 4096, stream=stream0)
            buf121 = empty_strided_cuda((), (), torch.float32)
            buf122 = buf121; del buf121  # reuse
            buf123 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_21, v_20, pow_19, sum_19, v_norm_sq_18, truediv_18, mul_19], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24.run(buf122, primals_11, buf123, 1, 64, stream=stream0)
            buf124 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_21, v_20, t_18, matmul_36], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1152), buf120, out=buf124)
            buf125 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_18], Original ATen: [aten.mm]
            extern_kernels.mm(buf123, buf124, out=buf125)
            buf126 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [Q_19], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf126, buf120, 4096, stream=stream0)
            buf127 = empty_strided_cuda((), (), torch.float32)
            buf128 = buf127; del buf127  # reuse
            buf129 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_22, v_21, pow_20, sum_20, v_norm_sq_19, truediv_19, mul_20], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25.run(buf128, primals_11, buf129, 1, 64, stream=stream0)
            buf130 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_22, v_21, t_19, matmul_38], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1216), buf126, out=buf130)
            buf131 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_19], Original ATen: [aten.mm]
            extern_kernels.mm(buf129, buf130, out=buf131)
            buf132 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [Q_20], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf132, buf126, 4096, stream=stream0)
            buf133 = empty_strided_cuda((), (), torch.float32)
            buf134 = buf133; del buf133  # reuse
            buf135 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_23, v_22, pow_21, sum_21, v_norm_sq_20, truediv_20, mul_21], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26.run(buf134, primals_11, buf135, 1, 64, stream=stream0)
            buf136 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_23, v_22, t_20, matmul_40], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1280), buf132, out=buf136)
            buf137 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_20], Original ATen: [aten.mm]
            extern_kernels.mm(buf135, buf136, out=buf137)
            buf138 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [Q_21], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf138, buf132, 4096, stream=stream0)
            buf139 = empty_strided_cuda((), (), torch.float32)
            buf140 = buf139; del buf139  # reuse
            buf141 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_24, v_23, pow_22, sum_22, v_norm_sq_21, truediv_21, mul_22], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27.run(buf140, primals_11, buf141, 1, 64, stream=stream0)
            buf142 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_24, v_23, t_21, matmul_42], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1344), buf138, out=buf142)
            buf143 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_21], Original ATen: [aten.mm]
            extern_kernels.mm(buf141, buf142, out=buf143)
            buf144 = buf143; del buf143  # reuse
            # Topologically Sorted Source Nodes: [Q_22], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf144, buf138, 4096, stream=stream0)
            buf145 = empty_strided_cuda((), (), torch.float32)
            buf146 = buf145; del buf145  # reuse
            buf147 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_25, v_24, pow_23, sum_23, v_norm_sq_22, truediv_22, mul_23], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28.run(buf146, primals_11, buf147, 1, 64, stream=stream0)
            buf148 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_25, v_24, t_22, matmul_44], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1408), buf144, out=buf148)
            buf149 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_22], Original ATen: [aten.mm]
            extern_kernels.mm(buf147, buf148, out=buf149)
            buf150 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [Q_23], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf150, buf144, 4096, stream=stream0)
            buf151 = empty_strided_cuda((), (), torch.float32)
            buf152 = buf151; del buf151  # reuse
            buf153 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_26, v_25, pow_24, sum_24, v_norm_sq_23, truediv_23, mul_24], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29.run(buf152, primals_11, buf153, 1, 64, stream=stream0)
            buf154 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_26, v_25, t_23, matmul_46], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1472), buf150, out=buf154)
            buf155 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_23], Original ATen: [aten.mm]
            extern_kernels.mm(buf153, buf154, out=buf155)
            buf156 = buf155; del buf155  # reuse
            # Topologically Sorted Source Nodes: [Q_24], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf156, buf150, 4096, stream=stream0)
            buf157 = empty_strided_cuda((), (), torch.float32)
            buf158 = buf157; del buf157  # reuse
            buf159 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_27, v_26, pow_25, sum_25, v_norm_sq_24, truediv_24, mul_25], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30.run(buf158, primals_11, buf159, 1, 64, stream=stream0)
            buf160 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_27, v_26, t_24, matmul_48], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1536), buf156, out=buf160)
            buf161 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_24], Original ATen: [aten.mm]
            extern_kernels.mm(buf159, buf160, out=buf161)
            buf162 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [Q_25], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf162, buf156, 4096, stream=stream0)
            buf163 = empty_strided_cuda((), (), torch.float32)
            buf164 = buf163; del buf163  # reuse
            buf165 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_28, v_27, pow_26, sum_26, v_norm_sq_25, truediv_25, mul_26], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31.run(buf164, primals_11, buf165, 1, 64, stream=stream0)
            buf166 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_28, v_27, t_25, matmul_50], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1600), buf162, out=buf166)
            buf167 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_25], Original ATen: [aten.mm]
            extern_kernels.mm(buf165, buf166, out=buf167)
            buf168 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [Q_26], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf168, buf162, 4096, stream=stream0)
            buf169 = empty_strided_cuda((), (), torch.float32)
            buf170 = buf169; del buf169  # reuse
            buf171 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_29, v_28, pow_27, sum_27, v_norm_sq_26, truediv_26, mul_27], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32.run(buf170, primals_11, buf171, 1, 64, stream=stream0)
            buf172 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_29, v_28, t_26, matmul_52], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1664), buf168, out=buf172)
            buf173 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_26], Original ATen: [aten.mm]
            extern_kernels.mm(buf171, buf172, out=buf173)
            buf174 = buf173; del buf173  # reuse
            # Topologically Sorted Source Nodes: [Q_27], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf174, buf168, 4096, stream=stream0)
            buf175 = empty_strided_cuda((), (), torch.float32)
            buf176 = buf175; del buf175  # reuse
            buf177 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_30, v_29, pow_28, sum_28, v_norm_sq_27, truediv_27, mul_28], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33.run(buf176, primals_11, buf177, 1, 64, stream=stream0)
            buf178 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_30, v_29, t_27, matmul_54], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1728), buf174, out=buf178)
            buf179 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_27], Original ATen: [aten.mm]
            extern_kernels.mm(buf177, buf178, out=buf179)
            buf180 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [Q_28], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf180, buf174, 4096, stream=stream0)
            buf181 = empty_strided_cuda((), (), torch.float32)
            buf182 = buf181; del buf181  # reuse
            buf183 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_31, v_30, pow_29, sum_29, v_norm_sq_28, truediv_28, mul_29], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34.run(buf182, primals_11, buf183, 1, 64, stream=stream0)
            buf184 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_31, v_30, t_28, matmul_56], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1792), buf180, out=buf184)
            buf185 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_28], Original ATen: [aten.mm]
            extern_kernels.mm(buf183, buf184, out=buf185)
            buf186 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [Q_29], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf186, buf180, 4096, stream=stream0)
            buf187 = empty_strided_cuda((), (), torch.float32)
            buf188 = buf187; del buf187  # reuse
            buf189 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_32, v_31, pow_30, sum_30, v_norm_sq_29, truediv_29, mul_30], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35.run(buf188, primals_11, buf189, 1, 64, stream=stream0)
            buf190 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_32, v_31, t_29, matmul_58], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1856), buf186, out=buf190)
            buf191 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_29], Original ATen: [aten.mm]
            extern_kernels.mm(buf189, buf190, out=buf191)
            buf192 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [Q_30], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf192, buf186, 4096, stream=stream0)
            buf193 = empty_strided_cuda((), (), torch.float32)
            buf194 = buf193; del buf193  # reuse
            buf195 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_33, v_32, pow_31, sum_31, v_norm_sq_30, truediv_30, mul_31], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36.run(buf194, primals_11, buf195, 1, 64, stream=stream0)
            buf196 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_33, v_32, t_30, matmul_60], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1920), buf192, out=buf196)
            buf197 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_30], Original ATen: [aten.mm]
            extern_kernels.mm(buf195, buf196, out=buf197)
            buf198 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [Q_31], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf198, buf192, 4096, stream=stream0)
            buf199 = empty_strided_cuda((), (), torch.float32)
            buf200 = buf199; del buf199  # reuse
            buf201 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_34, v_33, pow_32, sum_32, v_norm_sq_31, truediv_31, mul_32], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37.run(buf200, primals_11, buf201, 1, 64, stream=stream0)
            buf202 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_34, v_33, t_31, matmul_62], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_11, (1, 64), (1, 1), 1984), buf198, out=buf202)
            buf203 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_31], Original ATen: [aten.mm]
            extern_kernels.mm(buf201, buf202, out=buf203)
            buf204 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [Q_32], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf204, buf198, 4096, stream=stream0)
            buf205 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, chunk, view, q_1, q_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_38.run(buf11, buf205, 4194304, stream=stream0)
            buf206 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, chunk, view, q_1, t_32, q_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf205, (65536, 64), (64, 1), 0), reinterpret_tensor(buf204, (64, 64), (1, 64), 0), out=buf206)
            buf207 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, chunk, view_1, k_1, k_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_39.run(buf11, buf207, 4194304, stream=stream0)
            buf208 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, chunk, view_1, k_1, t_32, k_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf207, (65536, 64), (64, 1), 0), reinterpret_tensor(buf204, (64, 64), (1, 64), 0), out=buf208)
            buf209 = empty_strided_cuda((64, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inv_freq_scaled, vals, f, vals_1, f_1, vals_2, f_2, full_freqs, pad_1, full_freqs_1], Original ATen: [aten.div, aten.select, aten.view, aten.mul, aten.cat, aten.zeros]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_div_mul_select_view_zeros_40.run(primals_13, primals_12, buf209, 2048, stream=stream0)
            del primals_12
            del primals_13
            buf210 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [q_2, emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, chunk_1, mul_65, neg, cat_5, mul_66, q_rot], Original ATen: [aten._unsafe_view, aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf206, buf209, buf210, 4194304, stream=stream0)
            buf211 = buf206; del buf206  # reuse
            # Topologically Sorted Source Nodes: [q_2, emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, chunk_1, mul_65, neg, cat_5, mul_66, q_rot, q_3], Original ATen: [aten._unsafe_view, aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf210, (65536, 64), (64, 1), 0), buf204, out=buf211)
            buf212 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [k_2, emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, chunk_2, mul_67, neg_1, cat_6, mul_68, k_rot], Original ATen: [aten._unsafe_view, aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf208, buf209, buf212, 4194304, stream=stream0)
            buf213 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [k_2, emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, chunk_2, mul_67, neg_1, cat_6, mul_68, k_rot, k_3], Original ATen: [aten._unsafe_view, aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf212, (65536, 64), (64, 1), 0), buf204, out=buf213)
            buf214 = empty_strided_cuda((256, 1, 4, 64), (256, 256, 64, 1), torch.float32)
            buf215 = empty_strided_cuda((256, 1, 4, 64), (256, 256, 64, 1), torch.float32)
            buf216 = empty_strided_cuda((256, 1, 4, 64, 64), (16384, 16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, chunk, view_2, v_1, q_3, k_3, flex_attention], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_split_transpose_view_42.run(buf211, buf213, buf11, buf214, buf215, primals_15, primals_14, primals_20, primals_21, primals_16, primals_17, primals_18, primals_19, buf216, 1024, 1, 1, stream=stream0)
            buf219 = reinterpret_tensor(buf216, (256, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [linear_2, chunk, view_2, v_1, q_3, k_3, flex_attention], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_43.run(buf219, buf214, buf215, 4194304, stream=stream0)
            buf220 = empty_strided_cuda((256, 4, 64), (256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, chunk, view_2, v_1, q_3, k_3, flex_attention], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_44.run(buf214, buf215, buf220, 65536, stream=stream0)
            buf221 = empty_strided_cuda((256, 64, 4, 64), (16384, 256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_3, contiguous], Original ATen: [aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_45.run(buf219, buf221, 4194304, stream=stream0)
            buf222 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_3, contiguous, attn, attn_1], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf221, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_26, (256, 256), (1, 256), 0), out=buf222)
            buf223 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_1, linear_4], Original ATen: [aten._unsafe_view, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(primals_28, buf222, reinterpret_tensor(primals_27, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf223)
            del primals_28
            buf224 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf225 = reinterpret_tensor(buf224, (256, 64, 1), (64, 1, 1), 0); del buf224  # reuse
            buf226 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_2, attn_1, linear_4, gate_attn, mul_133, x_2, x_norm_1], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_46.run(buf225, buf7, primals_9, buf222, buf223, buf226, 16384, 256, stream=stream0)
            buf227 = empty_strided_cuda((16384, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf226, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_29, (256, 2048), (1, 256), 0), out=buf227)
            buf228 = empty_strided_cuda((256, 64, 1024), (65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12, chunk_3, silu, mul_134], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_split_47.run(buf227, buf228, 16777216, stream=stream0)
            buf229 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12, chunk_3, silu, mul_134, ffn_out], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (16384, 1024), (1024, 1), 0), reinterpret_tensor(primals_30, (1024, 256), (1, 1024), 0), out=buf229)
            buf230 = reinterpret_tensor(buf7, (256, 64, 256), (16384, 256, 1), 0); del buf7  # reuse
            buf231 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf232 = reinterpret_tensor(buf231, (256, 64, 1), (64, 1, 1), 0); del buf231  # reuse
            buf233 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_2, attn_1, linear_4, gate_attn, mul_133, x_2, ffn_out, x_3, x_norm_2], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_addmm_mean_mul_pow_rsqrt_sigmoid_view_48.run(buf230, buf232, primals_9, buf222, buf223, buf229, buf233, 16384, 256, stream=stream0)
            del primals_9
            buf234 = empty_strided_cuda((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf233, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_31, (256, 768), (1, 256), 0), out=buf234)
            buf235 = empty_strided_cuda((), (), torch.float32)
            buf236 = buf235; del buf235  # reuse
            buf237 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_146, v_132, pow_129, sum_129, v_norm_sq_128, truediv_129, mul_135], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4.run(buf236, primals_32, buf237, 1, 64, stream=stream0)
            buf238 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_146, v_132, t_130, matmul_260], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 0), buf12, out=buf238)
            buf239 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_128], Original ATen: [aten.mm]
            extern_kernels.mm(buf237, buf238, out=buf239)
            buf240 = buf239; del buf239  # reuse
            # Topologically Sorted Source Nodes: [Q_133], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_5.run(buf240, 4096, stream=stream0)
            buf241 = empty_strided_cuda((), (), torch.float32)
            buf242 = buf241; del buf241  # reuse
            buf243 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_147, v_133, pow_130, sum_130, v_norm_sq_129, truediv_130, mul_136], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6.run(buf242, primals_32, buf243, 1, 64, stream=stream0)
            buf244 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_147, v_133, t_131, matmul_262], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 64), buf240, out=buf244)
            buf245 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_129], Original ATen: [aten.mm]
            extern_kernels.mm(buf243, buf244, out=buf245)
            buf246 = buf245; del buf245  # reuse
            # Topologically Sorted Source Nodes: [Q_134], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf246, buf240, 4096, stream=stream0)
            buf247 = empty_strided_cuda((), (), torch.float32)
            buf248 = buf247; del buf247  # reuse
            buf249 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_148, v_134, pow_131, sum_131, v_norm_sq_130, truediv_131, mul_137], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8.run(buf248, primals_32, buf249, 1, 64, stream=stream0)
            buf250 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_148, v_134, t_132, matmul_264], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 128), buf246, out=buf250)
            buf251 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_130], Original ATen: [aten.mm]
            extern_kernels.mm(buf249, buf250, out=buf251)
            buf252 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [Q_135], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf252, buf246, 4096, stream=stream0)
            buf253 = empty_strided_cuda((), (), torch.float32)
            buf254 = buf253; del buf253  # reuse
            buf255 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_149, v_135, pow_132, sum_132, v_norm_sq_131, truediv_132, mul_138], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9.run(buf254, primals_32, buf255, 1, 64, stream=stream0)
            buf256 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_149, v_135, t_133, matmul_266], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 192), buf252, out=buf256)
            buf257 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_131], Original ATen: [aten.mm]
            extern_kernels.mm(buf255, buf256, out=buf257)
            buf258 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [Q_136], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf258, buf252, 4096, stream=stream0)
            buf259 = empty_strided_cuda((), (), torch.float32)
            buf260 = buf259; del buf259  # reuse
            buf261 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_150, v_136, pow_133, sum_133, v_norm_sq_132, truediv_133, mul_139], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10.run(buf260, primals_32, buf261, 1, 64, stream=stream0)
            buf262 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_150, v_136, t_134, matmul_268], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 256), buf258, out=buf262)
            buf263 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_132], Original ATen: [aten.mm]
            extern_kernels.mm(buf261, buf262, out=buf263)
            buf264 = buf263; del buf263  # reuse
            # Topologically Sorted Source Nodes: [Q_137], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf264, buf258, 4096, stream=stream0)
            buf265 = empty_strided_cuda((), (), torch.float32)
            buf266 = buf265; del buf265  # reuse
            buf267 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_151, v_137, pow_134, sum_134, v_norm_sq_133, truediv_134, mul_140], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11.run(buf266, primals_32, buf267, 1, 64, stream=stream0)
            buf268 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_151, v_137, t_135, matmul_270], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 320), buf264, out=buf268)
            buf269 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_133], Original ATen: [aten.mm]
            extern_kernels.mm(buf267, buf268, out=buf269)
            buf270 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [Q_138], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf270, buf264, 4096, stream=stream0)
            buf271 = empty_strided_cuda((), (), torch.float32)
            buf272 = buf271; del buf271  # reuse
            buf273 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_152, v_138, pow_135, sum_135, v_norm_sq_134, truediv_135, mul_141], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12.run(buf272, primals_32, buf273, 1, 64, stream=stream0)
            buf274 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_152, v_138, t_136, matmul_272], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 384), buf270, out=buf274)
            buf275 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_134], Original ATen: [aten.mm]
            extern_kernels.mm(buf273, buf274, out=buf275)
            buf276 = buf275; del buf275  # reuse
            # Topologically Sorted Source Nodes: [Q_139], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf276, buf270, 4096, stream=stream0)
            buf277 = empty_strided_cuda((), (), torch.float32)
            buf278 = buf277; del buf277  # reuse
            buf279 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_153, v_139, pow_136, sum_136, v_norm_sq_135, truediv_136, mul_142], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13.run(buf278, primals_32, buf279, 1, 64, stream=stream0)
            buf280 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_153, v_139, t_137, matmul_274], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 448), buf276, out=buf280)
            buf281 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_135], Original ATen: [aten.mm]
            extern_kernels.mm(buf279, buf280, out=buf281)
            buf282 = buf281; del buf281  # reuse
            # Topologically Sorted Source Nodes: [Q_140], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf282, buf276, 4096, stream=stream0)
            buf283 = empty_strided_cuda((), (), torch.float32)
            buf284 = buf283; del buf283  # reuse
            buf285 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_154, v_140, pow_137, sum_137, v_norm_sq_136, truediv_137, mul_143], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14.run(buf284, primals_32, buf285, 1, 64, stream=stream0)
            buf286 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_154, v_140, t_138, matmul_276], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 512), buf282, out=buf286)
            buf287 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_136], Original ATen: [aten.mm]
            extern_kernels.mm(buf285, buf286, out=buf287)
            buf288 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [Q_141], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf288, buf282, 4096, stream=stream0)
            buf289 = empty_strided_cuda((), (), torch.float32)
            buf290 = buf289; del buf289  # reuse
            buf291 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_155, v_141, pow_138, sum_138, v_norm_sq_137, truediv_138, mul_144], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15.run(buf290, primals_32, buf291, 1, 64, stream=stream0)
            buf292 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_155, v_141, t_139, matmul_278], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 576), buf288, out=buf292)
            buf293 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_137], Original ATen: [aten.mm]
            extern_kernels.mm(buf291, buf292, out=buf293)
            buf294 = buf293; del buf293  # reuse
            # Topologically Sorted Source Nodes: [Q_142], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf294, buf288, 4096, stream=stream0)
            buf295 = empty_strided_cuda((), (), torch.float32)
            buf296 = buf295; del buf295  # reuse
            buf297 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_156, v_142, pow_139, sum_139, v_norm_sq_138, truediv_139, mul_145], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16.run(buf296, primals_32, buf297, 1, 64, stream=stream0)
            buf298 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_156, v_142, t_140, matmul_280], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 640), buf294, out=buf298)
            buf299 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_138], Original ATen: [aten.mm]
            extern_kernels.mm(buf297, buf298, out=buf299)
            buf300 = buf299; del buf299  # reuse
            # Topologically Sorted Source Nodes: [Q_143], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf300, buf294, 4096, stream=stream0)
            buf301 = empty_strided_cuda((), (), torch.float32)
            buf302 = buf301; del buf301  # reuse
            buf303 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_157, v_143, pow_140, sum_140, v_norm_sq_139, truediv_140, mul_146], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17.run(buf302, primals_32, buf303, 1, 64, stream=stream0)
            buf304 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_157, v_143, t_141, matmul_282], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 704), buf300, out=buf304)
            buf305 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_139], Original ATen: [aten.mm]
            extern_kernels.mm(buf303, buf304, out=buf305)
            buf306 = buf305; del buf305  # reuse
            # Topologically Sorted Source Nodes: [Q_144], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf306, buf300, 4096, stream=stream0)
            buf307 = empty_strided_cuda((), (), torch.float32)
            buf308 = buf307; del buf307  # reuse
            buf309 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_158, v_144, pow_141, sum_141, v_norm_sq_140, truediv_141, mul_147], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18.run(buf308, primals_32, buf309, 1, 64, stream=stream0)
            buf310 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_158, v_144, t_142, matmul_284], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 768), buf306, out=buf310)
            buf311 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_140], Original ATen: [aten.mm]
            extern_kernels.mm(buf309, buf310, out=buf311)
            buf312 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [Q_145], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf312, buf306, 4096, stream=stream0)
            buf313 = empty_strided_cuda((), (), torch.float32)
            buf314 = buf313; del buf313  # reuse
            buf315 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_159, v_145, pow_142, sum_142, v_norm_sq_141, truediv_142, mul_148], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19.run(buf314, primals_32, buf315, 1, 64, stream=stream0)
            buf316 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_159, v_145, t_143, matmul_286], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 832), buf312, out=buf316)
            buf317 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_141], Original ATen: [aten.mm]
            extern_kernels.mm(buf315, buf316, out=buf317)
            buf318 = buf317; del buf317  # reuse
            # Topologically Sorted Source Nodes: [Q_146], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf318, buf312, 4096, stream=stream0)
            buf319 = empty_strided_cuda((), (), torch.float32)
            buf320 = buf319; del buf319  # reuse
            buf321 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_160, v_146, pow_143, sum_143, v_norm_sq_142, truediv_143, mul_149], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20.run(buf320, primals_32, buf321, 1, 64, stream=stream0)
            buf322 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_160, v_146, t_144, matmul_288], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 896), buf318, out=buf322)
            buf323 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_142], Original ATen: [aten.mm]
            extern_kernels.mm(buf321, buf322, out=buf323)
            buf324 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [Q_147], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf324, buf318, 4096, stream=stream0)
            buf325 = empty_strided_cuda((), (), torch.float32)
            buf326 = buf325; del buf325  # reuse
            buf327 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_161, v_147, pow_144, sum_144, v_norm_sq_143, truediv_144, mul_150], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21.run(buf326, primals_32, buf327, 1, 64, stream=stream0)
            buf328 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_161, v_147, t_145, matmul_290], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 960), buf324, out=buf328)
            buf329 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_143], Original ATen: [aten.mm]
            extern_kernels.mm(buf327, buf328, out=buf329)
            buf330 = buf329; del buf329  # reuse
            # Topologically Sorted Source Nodes: [Q_148], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf330, buf324, 4096, stream=stream0)
            buf331 = empty_strided_cuda((), (), torch.float32)
            buf332 = buf331; del buf331  # reuse
            buf333 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_162, v_148, pow_145, sum_145, v_norm_sq_144, truediv_145, mul_151], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22.run(buf332, primals_32, buf333, 1, 64, stream=stream0)
            buf334 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_162, v_148, t_146, matmul_292], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1024), buf330, out=buf334)
            buf335 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_144], Original ATen: [aten.mm]
            extern_kernels.mm(buf333, buf334, out=buf335)
            buf336 = buf335; del buf335  # reuse
            # Topologically Sorted Source Nodes: [Q_149], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf336, buf330, 4096, stream=stream0)
            buf337 = empty_strided_cuda((), (), torch.float32)
            buf338 = buf337; del buf337  # reuse
            buf339 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_163, v_149, pow_146, sum_146, v_norm_sq_145, truediv_146, mul_152], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23.run(buf338, primals_32, buf339, 1, 64, stream=stream0)
            buf340 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_163, v_149, t_147, matmul_294], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1088), buf336, out=buf340)
            buf341 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_145], Original ATen: [aten.mm]
            extern_kernels.mm(buf339, buf340, out=buf341)
            buf342 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [Q_150], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf342, buf336, 4096, stream=stream0)
            buf343 = empty_strided_cuda((), (), torch.float32)
            buf344 = buf343; del buf343  # reuse
            buf345 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_164, v_150, pow_147, sum_147, v_norm_sq_146, truediv_147, mul_153], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24.run(buf344, primals_32, buf345, 1, 64, stream=stream0)
            buf346 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_164, v_150, t_148, matmul_296], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1152), buf342, out=buf346)
            buf347 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_146], Original ATen: [aten.mm]
            extern_kernels.mm(buf345, buf346, out=buf347)
            buf348 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [Q_151], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf348, buf342, 4096, stream=stream0)
            buf349 = empty_strided_cuda((), (), torch.float32)
            buf350 = buf349; del buf349  # reuse
            buf351 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_165, v_151, pow_148, sum_148, v_norm_sq_147, truediv_148, mul_154], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25.run(buf350, primals_32, buf351, 1, 64, stream=stream0)
            buf352 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_165, v_151, t_149, matmul_298], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1216), buf348, out=buf352)
            buf353 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_147], Original ATen: [aten.mm]
            extern_kernels.mm(buf351, buf352, out=buf353)
            buf354 = buf353; del buf353  # reuse
            # Topologically Sorted Source Nodes: [Q_152], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf354, buf348, 4096, stream=stream0)
            buf355 = empty_strided_cuda((), (), torch.float32)
            buf356 = buf355; del buf355  # reuse
            buf357 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_166, v_152, pow_149, sum_149, v_norm_sq_148, truediv_149, mul_155], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26.run(buf356, primals_32, buf357, 1, 64, stream=stream0)
            buf358 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_166, v_152, t_150, matmul_300], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1280), buf354, out=buf358)
            buf359 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_148], Original ATen: [aten.mm]
            extern_kernels.mm(buf357, buf358, out=buf359)
            buf360 = buf359; del buf359  # reuse
            # Topologically Sorted Source Nodes: [Q_153], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf360, buf354, 4096, stream=stream0)
            buf361 = empty_strided_cuda((), (), torch.float32)
            buf362 = buf361; del buf361  # reuse
            buf363 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_167, v_153, pow_150, sum_150, v_norm_sq_149, truediv_150, mul_156], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27.run(buf362, primals_32, buf363, 1, 64, stream=stream0)
            buf364 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_167, v_153, t_151, matmul_302], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1344), buf360, out=buf364)
            buf365 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_149], Original ATen: [aten.mm]
            extern_kernels.mm(buf363, buf364, out=buf365)
            buf366 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [Q_154], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf366, buf360, 4096, stream=stream0)
            buf367 = empty_strided_cuda((), (), torch.float32)
            buf368 = buf367; del buf367  # reuse
            buf369 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_168, v_154, pow_151, sum_151, v_norm_sq_150, truediv_151, mul_157], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28.run(buf368, primals_32, buf369, 1, 64, stream=stream0)
            buf370 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_168, v_154, t_152, matmul_304], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1408), buf366, out=buf370)
            buf371 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_150], Original ATen: [aten.mm]
            extern_kernels.mm(buf369, buf370, out=buf371)
            buf372 = buf371; del buf371  # reuse
            # Topologically Sorted Source Nodes: [Q_155], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf372, buf366, 4096, stream=stream0)
            buf373 = empty_strided_cuda((), (), torch.float32)
            buf374 = buf373; del buf373  # reuse
            buf375 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_169, v_155, pow_152, sum_152, v_norm_sq_151, truediv_152, mul_158], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29.run(buf374, primals_32, buf375, 1, 64, stream=stream0)
            buf376 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_169, v_155, t_153, matmul_306], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1472), buf372, out=buf376)
            buf377 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_151], Original ATen: [aten.mm]
            extern_kernels.mm(buf375, buf376, out=buf377)
            buf378 = buf377; del buf377  # reuse
            # Topologically Sorted Source Nodes: [Q_156], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf378, buf372, 4096, stream=stream0)
            buf379 = empty_strided_cuda((), (), torch.float32)
            buf380 = buf379; del buf379  # reuse
            buf381 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_170, v_156, pow_153, sum_153, v_norm_sq_152, truediv_153, mul_159], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30.run(buf380, primals_32, buf381, 1, 64, stream=stream0)
            buf382 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_170, v_156, t_154, matmul_308], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1536), buf378, out=buf382)
            buf383 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_152], Original ATen: [aten.mm]
            extern_kernels.mm(buf381, buf382, out=buf383)
            buf384 = buf383; del buf383  # reuse
            # Topologically Sorted Source Nodes: [Q_157], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf384, buf378, 4096, stream=stream0)
            buf385 = empty_strided_cuda((), (), torch.float32)
            buf386 = buf385; del buf385  # reuse
            buf387 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_171, v_157, pow_154, sum_154, v_norm_sq_153, truediv_154, mul_160], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31.run(buf386, primals_32, buf387, 1, 64, stream=stream0)
            buf388 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_171, v_157, t_155, matmul_310], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1600), buf384, out=buf388)
            buf389 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_153], Original ATen: [aten.mm]
            extern_kernels.mm(buf387, buf388, out=buf389)
            buf390 = buf389; del buf389  # reuse
            # Topologically Sorted Source Nodes: [Q_158], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf390, buf384, 4096, stream=stream0)
            buf391 = empty_strided_cuda((), (), torch.float32)
            buf392 = buf391; del buf391  # reuse
            buf393 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_172, v_158, pow_155, sum_155, v_norm_sq_154, truediv_155, mul_161], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32.run(buf392, primals_32, buf393, 1, 64, stream=stream0)
            buf394 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_172, v_158, t_156, matmul_312], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1664), buf390, out=buf394)
            buf395 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_154], Original ATen: [aten.mm]
            extern_kernels.mm(buf393, buf394, out=buf395)
            buf396 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [Q_159], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf396, buf390, 4096, stream=stream0)
            buf397 = empty_strided_cuda((), (), torch.float32)
            buf398 = buf397; del buf397  # reuse
            buf399 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_173, v_159, pow_156, sum_156, v_norm_sq_155, truediv_156, mul_162], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33.run(buf398, primals_32, buf399, 1, 64, stream=stream0)
            buf400 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_173, v_159, t_157, matmul_314], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1728), buf396, out=buf400)
            buf401 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_155], Original ATen: [aten.mm]
            extern_kernels.mm(buf399, buf400, out=buf401)
            buf402 = buf401; del buf401  # reuse
            # Topologically Sorted Source Nodes: [Q_160], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf402, buf396, 4096, stream=stream0)
            buf403 = empty_strided_cuda((), (), torch.float32)
            buf404 = buf403; del buf403  # reuse
            buf405 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_174, v_160, pow_157, sum_157, v_norm_sq_156, truediv_157, mul_163], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34.run(buf404, primals_32, buf405, 1, 64, stream=stream0)
            buf406 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_174, v_160, t_158, matmul_316], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1792), buf402, out=buf406)
            buf407 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_156], Original ATen: [aten.mm]
            extern_kernels.mm(buf405, buf406, out=buf407)
            buf408 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [Q_161], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf408, buf402, 4096, stream=stream0)
            buf409 = empty_strided_cuda((), (), torch.float32)
            buf410 = buf409; del buf409  # reuse
            buf411 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_175, v_161, pow_158, sum_158, v_norm_sq_157, truediv_158, mul_164], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35.run(buf410, primals_32, buf411, 1, 64, stream=stream0)
            buf412 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_175, v_161, t_159, matmul_318], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1856), buf408, out=buf412)
            buf413 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_157], Original ATen: [aten.mm]
            extern_kernels.mm(buf411, buf412, out=buf413)
            buf414 = buf413; del buf413  # reuse
            # Topologically Sorted Source Nodes: [Q_162], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf414, buf408, 4096, stream=stream0)
            buf415 = empty_strided_cuda((), (), torch.float32)
            buf416 = buf415; del buf415  # reuse
            buf417 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_176, v_162, pow_159, sum_159, v_norm_sq_158, truediv_159, mul_165], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36.run(buf416, primals_32, buf417, 1, 64, stream=stream0)
            buf418 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_176, v_162, t_160, matmul_320], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1920), buf414, out=buf418)
            buf419 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_158], Original ATen: [aten.mm]
            extern_kernels.mm(buf417, buf418, out=buf419)
            buf420 = buf419; del buf419  # reuse
            # Topologically Sorted Source Nodes: [Q_163], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf420, buf414, 4096, stream=stream0)
            buf421 = empty_strided_cuda((), (), torch.float32)
            buf422 = buf421; del buf421  # reuse
            buf423 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_177, v_163, pow_160, sum_160, v_norm_sq_159, truediv_160, mul_166], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37.run(buf422, primals_32, buf423, 1, 64, stream=stream0)
            buf424 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_177, v_163, t_161, matmul_322], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_32, (1, 64), (1, 1), 1984), buf420, out=buf424)
            buf425 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_159], Original ATen: [aten.mm]
            extern_kernels.mm(buf423, buf424, out=buf425)
            buf426 = buf425; del buf425  # reuse
            # Topologically Sorted Source Nodes: [Q_164], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf426, buf420, 4096, stream=stream0)
            buf427 = reinterpret_tensor(buf229, (256, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [linear_7, chunk_4, view_4, q_5, q_6], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_38.run(buf234, buf427, 4194304, stream=stream0)
            buf428 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, chunk_4, view_4, q_5, t_162, q_6], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf427, (65536, 64), (64, 1), 0), reinterpret_tensor(buf426, (64, 64), (1, 64), 0), out=buf428)
            buf429 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, chunk_4, view_5, k_5, k_6], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_39.run(buf234, buf429, 4194304, stream=stream0)
            buf430 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, chunk_4, view_5, k_5, t_162, k_6], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf429, (65536, 64), (64, 1), 0), reinterpret_tensor(buf426, (64, 64), (1, 64), 0), out=buf430)
            buf431 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, q_6, chunk_5, mul_199, neg_2, cat_10, mul_200, q_rot_1], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf428, buf209, buf431, 4194304, stream=stream0)
            buf432 = buf428; del buf428  # reuse
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, q_6, chunk_5, mul_199, neg_2, cat_10, mul_200, q_rot_1, q_7], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf431, (65536, 64), (64, 1), 0), buf426, out=buf432)
            buf433 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, k_6, chunk_6, mul_201, neg_3, cat_11, mul_202, k_rot_1], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf430, buf209, buf433, 4194304, stream=stream0)
            buf434 = buf430; del buf430  # reuse
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, k_6, chunk_6, mul_201, neg_3, cat_11, mul_202, k_rot_1, k_7], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf433, (65536, 64), (64, 1), 0), buf426, out=buf434)
            buf435 = buf215; del buf215  # reuse
            buf436 = buf214; del buf214  # reuse
            buf437 = empty_strided_cuda((256, 1, 4, 64, 64), (16384, 16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, chunk_4, view_6, v_131, q_7, k_7, flex_attention_1], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_split_transpose_view_42.run(buf432, buf434, buf234, buf435, buf436, primals_15, primals_14, primals_20, primals_21, primals_16, primals_17, primals_18, primals_19, buf437, 1024, 1, 1, stream=stream0)
            buf440 = reinterpret_tensor(buf437, (256, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf437  # reuse
            # Topologically Sorted Source Nodes: [linear_7, chunk_4, view_6, v_131, q_7, k_7, flex_attention_1], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_43.run(buf440, buf435, buf436, 4194304, stream=stream0)
            buf441 = empty_strided_cuda((256, 4, 64), (256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, chunk_4, view_6, v_131, q_7, k_7, flex_attention_1], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_44.run(buf435, buf436, buf441, 65536, stream=stream0)
            buf442 = empty_strided_cuda((256, 64, 4, 64), (16384, 256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_7, contiguous_1], Original ATen: [aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_45.run(buf440, buf442, 4194304, stream=stream0)
            buf443 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_7, contiguous_1, attn_2, attn_3], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf442, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_33, (256, 256), (1, 256), 0), out=buf443)
            buf444 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_3, linear_9], Original ATen: [aten._unsafe_view, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(primals_35, buf443, reinterpret_tensor(primals_34, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf444)
            del primals_35
            buf445 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf446 = reinterpret_tensor(buf445, (256, 64, 1), (64, 1, 1), 0); del buf445  # reuse
            buf447 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_3, linear_9, gate_attn_1, mul_267, x_4, x_norm_3], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_49.run(buf446, buf230, buf443, buf444, buf447, 16384, 256, stream=stream0)
            buf448 = empty_strided_cuda((16384, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_1], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf447, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_36, (256, 2048), (1, 256), 0), out=buf448)
            buf449 = empty_strided_cuda((256, 64, 1024), (65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_1, chunk_7, silu_1, mul_268], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_split_47.run(buf448, buf449, 16777216, stream=stream0)
            buf450 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_1, chunk_7, silu_1, mul_268, ffn_out_1], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf449, (16384, 1024), (1024, 1), 0), reinterpret_tensor(primals_37, (1024, 256), (1, 1024), 0), out=buf450)
            buf451 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf452 = reinterpret_tensor(buf451, (256, 64, 1), (64, 1, 1), 0); del buf451  # reuse
            buf453 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_3, linear_9, gate_attn_1, mul_267, x_4, ffn_out_1, x_5, x_norm_4], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_50.run(buf452, buf230, buf443, buf444, buf450, buf453, 16384, 256, stream=stream0)
            buf454 = empty_strided_cuda((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf453, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_38, (256, 768), (1, 256), 0), out=buf454)
            buf455 = empty_strided_cuda((), (), torch.float32)
            buf456 = buf455; del buf455  # reuse
            buf457 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_289, v_262, pow_257, sum_257, v_norm_sq_256, truediv_258, mul_269], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4.run(buf456, primals_39, buf457, 1, 64, stream=stream0)
            buf458 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_289, v_262, t_260, matmul_520], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 0), buf12, out=buf458)
            buf459 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_256], Original ATen: [aten.mm]
            extern_kernels.mm(buf457, buf458, out=buf459)
            buf460 = buf459; del buf459  # reuse
            # Topologically Sorted Source Nodes: [Q_265], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_5.run(buf460, 4096, stream=stream0)
            buf461 = empty_strided_cuda((), (), torch.float32)
            buf462 = buf461; del buf461  # reuse
            buf463 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_290, v_263, pow_258, sum_258, v_norm_sq_257, truediv_259, mul_270], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6.run(buf462, primals_39, buf463, 1, 64, stream=stream0)
            buf464 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_290, v_263, t_261, matmul_522], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 64), buf460, out=buf464)
            buf465 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_257], Original ATen: [aten.mm]
            extern_kernels.mm(buf463, buf464, out=buf465)
            buf466 = buf465; del buf465  # reuse
            # Topologically Sorted Source Nodes: [Q_266], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf466, buf460, 4096, stream=stream0)
            buf467 = empty_strided_cuda((), (), torch.float32)
            buf468 = buf467; del buf467  # reuse
            buf469 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_291, v_264, pow_259, sum_259, v_norm_sq_258, truediv_260, mul_271], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8.run(buf468, primals_39, buf469, 1, 64, stream=stream0)
            buf470 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_291, v_264, t_262, matmul_524], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 128), buf466, out=buf470)
            buf471 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_258], Original ATen: [aten.mm]
            extern_kernels.mm(buf469, buf470, out=buf471)
            buf472 = buf471; del buf471  # reuse
            # Topologically Sorted Source Nodes: [Q_267], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf472, buf466, 4096, stream=stream0)
            buf473 = empty_strided_cuda((), (), torch.float32)
            buf474 = buf473; del buf473  # reuse
            buf475 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_292, v_265, pow_260, sum_260, v_norm_sq_259, truediv_261, mul_272], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9.run(buf474, primals_39, buf475, 1, 64, stream=stream0)
            buf476 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_292, v_265, t_263, matmul_526], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 192), buf472, out=buf476)
            buf477 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_259], Original ATen: [aten.mm]
            extern_kernels.mm(buf475, buf476, out=buf477)
            buf478 = buf477; del buf477  # reuse
            # Topologically Sorted Source Nodes: [Q_268], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf478, buf472, 4096, stream=stream0)
            buf479 = empty_strided_cuda((), (), torch.float32)
            buf480 = buf479; del buf479  # reuse
            buf481 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_293, v_266, pow_261, sum_261, v_norm_sq_260, truediv_262, mul_273], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10.run(buf480, primals_39, buf481, 1, 64, stream=stream0)
            buf482 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_293, v_266, t_264, matmul_528], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 256), buf478, out=buf482)
            buf483 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_260], Original ATen: [aten.mm]
            extern_kernels.mm(buf481, buf482, out=buf483)
            buf484 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [Q_269], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf484, buf478, 4096, stream=stream0)
            buf485 = empty_strided_cuda((), (), torch.float32)
            buf486 = buf485; del buf485  # reuse
            buf487 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_294, v_267, pow_262, sum_262, v_norm_sq_261, truediv_263, mul_274], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11.run(buf486, primals_39, buf487, 1, 64, stream=stream0)
            buf488 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_294, v_267, t_265, matmul_530], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 320), buf484, out=buf488)
            buf489 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_261], Original ATen: [aten.mm]
            extern_kernels.mm(buf487, buf488, out=buf489)
            buf490 = buf489; del buf489  # reuse
            # Topologically Sorted Source Nodes: [Q_270], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf490, buf484, 4096, stream=stream0)
            buf491 = empty_strided_cuda((), (), torch.float32)
            buf492 = buf491; del buf491  # reuse
            buf493 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_295, v_268, pow_263, sum_263, v_norm_sq_262, truediv_264, mul_275], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12.run(buf492, primals_39, buf493, 1, 64, stream=stream0)
            buf494 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_295, v_268, t_266, matmul_532], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 384), buf490, out=buf494)
            buf495 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_262], Original ATen: [aten.mm]
            extern_kernels.mm(buf493, buf494, out=buf495)
            buf496 = buf495; del buf495  # reuse
            # Topologically Sorted Source Nodes: [Q_271], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf496, buf490, 4096, stream=stream0)
            buf497 = empty_strided_cuda((), (), torch.float32)
            buf498 = buf497; del buf497  # reuse
            buf499 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_296, v_269, pow_264, sum_264, v_norm_sq_263, truediv_265, mul_276], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13.run(buf498, primals_39, buf499, 1, 64, stream=stream0)
            buf500 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_296, v_269, t_267, matmul_534], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 448), buf496, out=buf500)
            buf501 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_263], Original ATen: [aten.mm]
            extern_kernels.mm(buf499, buf500, out=buf501)
            buf502 = buf501; del buf501  # reuse
            # Topologically Sorted Source Nodes: [Q_272], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf502, buf496, 4096, stream=stream0)
            buf503 = empty_strided_cuda((), (), torch.float32)
            buf504 = buf503; del buf503  # reuse
            buf505 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_297, v_270, pow_265, sum_265, v_norm_sq_264, truediv_266, mul_277], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14.run(buf504, primals_39, buf505, 1, 64, stream=stream0)
            buf506 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_297, v_270, t_268, matmul_536], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 512), buf502, out=buf506)
            buf507 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_264], Original ATen: [aten.mm]
            extern_kernels.mm(buf505, buf506, out=buf507)
            buf508 = buf507; del buf507  # reuse
            # Topologically Sorted Source Nodes: [Q_273], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf508, buf502, 4096, stream=stream0)
            buf509 = empty_strided_cuda((), (), torch.float32)
            buf510 = buf509; del buf509  # reuse
            buf511 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_298, v_271, pow_266, sum_266, v_norm_sq_265, truediv_267, mul_278], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15.run(buf510, primals_39, buf511, 1, 64, stream=stream0)
            buf512 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_298, v_271, t_269, matmul_538], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 576), buf508, out=buf512)
            buf513 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_265], Original ATen: [aten.mm]
            extern_kernels.mm(buf511, buf512, out=buf513)
            buf514 = buf513; del buf513  # reuse
            # Topologically Sorted Source Nodes: [Q_274], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf514, buf508, 4096, stream=stream0)
            buf515 = empty_strided_cuda((), (), torch.float32)
            buf516 = buf515; del buf515  # reuse
            buf517 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_299, v_272, pow_267, sum_267, v_norm_sq_266, truediv_268, mul_279], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16.run(buf516, primals_39, buf517, 1, 64, stream=stream0)
            buf518 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_299, v_272, t_270, matmul_540], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 640), buf514, out=buf518)
            buf519 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_266], Original ATen: [aten.mm]
            extern_kernels.mm(buf517, buf518, out=buf519)
            buf520 = buf519; del buf519  # reuse
            # Topologically Sorted Source Nodes: [Q_275], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf520, buf514, 4096, stream=stream0)
            buf521 = empty_strided_cuda((), (), torch.float32)
            buf522 = buf521; del buf521  # reuse
            buf523 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_300, v_273, pow_268, sum_268, v_norm_sq_267, truediv_269, mul_280], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17.run(buf522, primals_39, buf523, 1, 64, stream=stream0)
            buf524 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_300, v_273, t_271, matmul_542], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 704), buf520, out=buf524)
            buf525 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_267], Original ATen: [aten.mm]
            extern_kernels.mm(buf523, buf524, out=buf525)
            buf526 = buf525; del buf525  # reuse
            # Topologically Sorted Source Nodes: [Q_276], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf526, buf520, 4096, stream=stream0)
            buf527 = empty_strided_cuda((), (), torch.float32)
            buf528 = buf527; del buf527  # reuse
            buf529 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_301, v_274, pow_269, sum_269, v_norm_sq_268, truediv_270, mul_281], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18.run(buf528, primals_39, buf529, 1, 64, stream=stream0)
            buf530 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_301, v_274, t_272, matmul_544], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 768), buf526, out=buf530)
            buf531 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_268], Original ATen: [aten.mm]
            extern_kernels.mm(buf529, buf530, out=buf531)
            buf532 = buf531; del buf531  # reuse
            # Topologically Sorted Source Nodes: [Q_277], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf532, buf526, 4096, stream=stream0)
            buf533 = empty_strided_cuda((), (), torch.float32)
            buf534 = buf533; del buf533  # reuse
            buf535 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_302, v_275, pow_270, sum_270, v_norm_sq_269, truediv_271, mul_282], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19.run(buf534, primals_39, buf535, 1, 64, stream=stream0)
            buf536 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_302, v_275, t_273, matmul_546], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 832), buf532, out=buf536)
            buf537 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_269], Original ATen: [aten.mm]
            extern_kernels.mm(buf535, buf536, out=buf537)
            buf538 = buf537; del buf537  # reuse
            # Topologically Sorted Source Nodes: [Q_278], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf538, buf532, 4096, stream=stream0)
            buf539 = empty_strided_cuda((), (), torch.float32)
            buf540 = buf539; del buf539  # reuse
            buf541 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_303, v_276, pow_271, sum_271, v_norm_sq_270, truediv_272, mul_283], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20.run(buf540, primals_39, buf541, 1, 64, stream=stream0)
            buf542 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_303, v_276, t_274, matmul_548], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 896), buf538, out=buf542)
            buf543 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_270], Original ATen: [aten.mm]
            extern_kernels.mm(buf541, buf542, out=buf543)
            buf544 = buf543; del buf543  # reuse
            # Topologically Sorted Source Nodes: [Q_279], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf544, buf538, 4096, stream=stream0)
            buf545 = empty_strided_cuda((), (), torch.float32)
            buf546 = buf545; del buf545  # reuse
            buf547 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_304, v_277, pow_272, sum_272, v_norm_sq_271, truediv_273, mul_284], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21.run(buf546, primals_39, buf547, 1, 64, stream=stream0)
            buf548 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_304, v_277, t_275, matmul_550], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 960), buf544, out=buf548)
            buf549 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_271], Original ATen: [aten.mm]
            extern_kernels.mm(buf547, buf548, out=buf549)
            buf550 = buf549; del buf549  # reuse
            # Topologically Sorted Source Nodes: [Q_280], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf550, buf544, 4096, stream=stream0)
            buf551 = empty_strided_cuda((), (), torch.float32)
            buf552 = buf551; del buf551  # reuse
            buf553 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_305, v_278, pow_273, sum_273, v_norm_sq_272, truediv_274, mul_285], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22.run(buf552, primals_39, buf553, 1, 64, stream=stream0)
            buf554 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_305, v_278, t_276, matmul_552], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1024), buf550, out=buf554)
            buf555 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_272], Original ATen: [aten.mm]
            extern_kernels.mm(buf553, buf554, out=buf555)
            buf556 = buf555; del buf555  # reuse
            # Topologically Sorted Source Nodes: [Q_281], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf556, buf550, 4096, stream=stream0)
            buf557 = empty_strided_cuda((), (), torch.float32)
            buf558 = buf557; del buf557  # reuse
            buf559 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_306, v_279, pow_274, sum_274, v_norm_sq_273, truediv_275, mul_286], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23.run(buf558, primals_39, buf559, 1, 64, stream=stream0)
            buf560 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_306, v_279, t_277, matmul_554], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1088), buf556, out=buf560)
            buf561 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_273], Original ATen: [aten.mm]
            extern_kernels.mm(buf559, buf560, out=buf561)
            buf562 = buf561; del buf561  # reuse
            # Topologically Sorted Source Nodes: [Q_282], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf562, buf556, 4096, stream=stream0)
            buf563 = empty_strided_cuda((), (), torch.float32)
            buf564 = buf563; del buf563  # reuse
            buf565 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_307, v_280, pow_275, sum_275, v_norm_sq_274, truediv_276, mul_287], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24.run(buf564, primals_39, buf565, 1, 64, stream=stream0)
            buf566 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_307, v_280, t_278, matmul_556], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1152), buf562, out=buf566)
            buf567 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_274], Original ATen: [aten.mm]
            extern_kernels.mm(buf565, buf566, out=buf567)
            buf568 = buf567; del buf567  # reuse
            # Topologically Sorted Source Nodes: [Q_283], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf568, buf562, 4096, stream=stream0)
            buf569 = empty_strided_cuda((), (), torch.float32)
            buf570 = buf569; del buf569  # reuse
            buf571 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_308, v_281, pow_276, sum_276, v_norm_sq_275, truediv_277, mul_288], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25.run(buf570, primals_39, buf571, 1, 64, stream=stream0)
            buf572 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_308, v_281, t_279, matmul_558], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1216), buf568, out=buf572)
            buf573 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_275], Original ATen: [aten.mm]
            extern_kernels.mm(buf571, buf572, out=buf573)
            buf574 = buf573; del buf573  # reuse
            # Topologically Sorted Source Nodes: [Q_284], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf574, buf568, 4096, stream=stream0)
            buf575 = empty_strided_cuda((), (), torch.float32)
            buf576 = buf575; del buf575  # reuse
            buf577 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_309, v_282, pow_277, sum_277, v_norm_sq_276, truediv_278, mul_289], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26.run(buf576, primals_39, buf577, 1, 64, stream=stream0)
            buf578 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_309, v_282, t_280, matmul_560], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1280), buf574, out=buf578)
            buf579 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_276], Original ATen: [aten.mm]
            extern_kernels.mm(buf577, buf578, out=buf579)
            buf580 = buf579; del buf579  # reuse
            # Topologically Sorted Source Nodes: [Q_285], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf580, buf574, 4096, stream=stream0)
            buf581 = empty_strided_cuda((), (), torch.float32)
            buf582 = buf581; del buf581  # reuse
            buf583 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_310, v_283, pow_278, sum_278, v_norm_sq_277, truediv_279, mul_290], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27.run(buf582, primals_39, buf583, 1, 64, stream=stream0)
            buf584 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_310, v_283, t_281, matmul_562], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1344), buf580, out=buf584)
            buf585 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_277], Original ATen: [aten.mm]
            extern_kernels.mm(buf583, buf584, out=buf585)
            buf586 = buf585; del buf585  # reuse
            # Topologically Sorted Source Nodes: [Q_286], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf586, buf580, 4096, stream=stream0)
            buf587 = empty_strided_cuda((), (), torch.float32)
            buf588 = buf587; del buf587  # reuse
            buf589 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_311, v_284, pow_279, sum_279, v_norm_sq_278, truediv_280, mul_291], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28.run(buf588, primals_39, buf589, 1, 64, stream=stream0)
            buf590 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_311, v_284, t_282, matmul_564], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1408), buf586, out=buf590)
            buf591 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_278], Original ATen: [aten.mm]
            extern_kernels.mm(buf589, buf590, out=buf591)
            buf592 = buf591; del buf591  # reuse
            # Topologically Sorted Source Nodes: [Q_287], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf592, buf586, 4096, stream=stream0)
            buf593 = empty_strided_cuda((), (), torch.float32)
            buf594 = buf593; del buf593  # reuse
            buf595 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_312, v_285, pow_280, sum_280, v_norm_sq_279, truediv_281, mul_292], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29.run(buf594, primals_39, buf595, 1, 64, stream=stream0)
            buf596 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_312, v_285, t_283, matmul_566], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1472), buf592, out=buf596)
            buf597 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_279], Original ATen: [aten.mm]
            extern_kernels.mm(buf595, buf596, out=buf597)
            buf598 = buf597; del buf597  # reuse
            # Topologically Sorted Source Nodes: [Q_288], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf598, buf592, 4096, stream=stream0)
            buf599 = empty_strided_cuda((), (), torch.float32)
            buf600 = buf599; del buf599  # reuse
            buf601 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_313, v_286, pow_281, sum_281, v_norm_sq_280, truediv_282, mul_293], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30.run(buf600, primals_39, buf601, 1, 64, stream=stream0)
            buf602 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_313, v_286, t_284, matmul_568], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1536), buf598, out=buf602)
            buf603 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_280], Original ATen: [aten.mm]
            extern_kernels.mm(buf601, buf602, out=buf603)
            buf604 = buf603; del buf603  # reuse
            # Topologically Sorted Source Nodes: [Q_289], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf604, buf598, 4096, stream=stream0)
            buf605 = empty_strided_cuda((), (), torch.float32)
            buf606 = buf605; del buf605  # reuse
            buf607 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_314, v_287, pow_282, sum_282, v_norm_sq_281, truediv_283, mul_294], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31.run(buf606, primals_39, buf607, 1, 64, stream=stream0)
            buf608 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_314, v_287, t_285, matmul_570], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1600), buf604, out=buf608)
            buf609 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_281], Original ATen: [aten.mm]
            extern_kernels.mm(buf607, buf608, out=buf609)
            buf610 = buf609; del buf609  # reuse
            # Topologically Sorted Source Nodes: [Q_290], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf610, buf604, 4096, stream=stream0)
            buf611 = empty_strided_cuda((), (), torch.float32)
            buf612 = buf611; del buf611  # reuse
            buf613 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_315, v_288, pow_283, sum_283, v_norm_sq_282, truediv_284, mul_295], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32.run(buf612, primals_39, buf613, 1, 64, stream=stream0)
            buf614 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_315, v_288, t_286, matmul_572], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1664), buf610, out=buf614)
            buf615 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_282], Original ATen: [aten.mm]
            extern_kernels.mm(buf613, buf614, out=buf615)
            buf616 = buf615; del buf615  # reuse
            # Topologically Sorted Source Nodes: [Q_291], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf616, buf610, 4096, stream=stream0)
            buf617 = empty_strided_cuda((), (), torch.float32)
            buf618 = buf617; del buf617  # reuse
            buf619 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_316, v_289, pow_284, sum_284, v_norm_sq_283, truediv_285, mul_296], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33.run(buf618, primals_39, buf619, 1, 64, stream=stream0)
            buf620 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_316, v_289, t_287, matmul_574], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1728), buf616, out=buf620)
            buf621 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_283], Original ATen: [aten.mm]
            extern_kernels.mm(buf619, buf620, out=buf621)
            buf622 = buf621; del buf621  # reuse
            # Topologically Sorted Source Nodes: [Q_292], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf622, buf616, 4096, stream=stream0)
            buf623 = empty_strided_cuda((), (), torch.float32)
            buf624 = buf623; del buf623  # reuse
            buf625 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_317, v_290, pow_285, sum_285, v_norm_sq_284, truediv_286, mul_297], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34.run(buf624, primals_39, buf625, 1, 64, stream=stream0)
            buf626 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_317, v_290, t_288, matmul_576], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1792), buf622, out=buf626)
            buf627 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_284], Original ATen: [aten.mm]
            extern_kernels.mm(buf625, buf626, out=buf627)
            buf628 = buf627; del buf627  # reuse
            # Topologically Sorted Source Nodes: [Q_293], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf628, buf622, 4096, stream=stream0)
            buf629 = empty_strided_cuda((), (), torch.float32)
            buf630 = buf629; del buf629  # reuse
            buf631 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_318, v_291, pow_286, sum_286, v_norm_sq_285, truediv_287, mul_298], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35.run(buf630, primals_39, buf631, 1, 64, stream=stream0)
            buf632 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_318, v_291, t_289, matmul_578], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1856), buf628, out=buf632)
            buf633 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_285], Original ATen: [aten.mm]
            extern_kernels.mm(buf631, buf632, out=buf633)
            buf634 = buf633; del buf633  # reuse
            # Topologically Sorted Source Nodes: [Q_294], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf634, buf628, 4096, stream=stream0)
            buf635 = empty_strided_cuda((), (), torch.float32)
            buf636 = buf635; del buf635  # reuse
            buf637 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_319, v_292, pow_287, sum_287, v_norm_sq_286, truediv_288, mul_299], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36.run(buf636, primals_39, buf637, 1, 64, stream=stream0)
            buf638 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_319, v_292, t_290, matmul_580], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1920), buf634, out=buf638)
            buf639 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_286], Original ATen: [aten.mm]
            extern_kernels.mm(buf637, buf638, out=buf639)
            buf640 = buf639; del buf639  # reuse
            # Topologically Sorted Source Nodes: [Q_295], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf640, buf634, 4096, stream=stream0)
            buf641 = empty_strided_cuda((), (), torch.float32)
            buf642 = buf641; del buf641  # reuse
            buf643 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_320, v_293, pow_288, sum_288, v_norm_sq_287, truediv_289, mul_300], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37.run(buf642, primals_39, buf643, 1, 64, stream=stream0)
            buf644 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_320, v_293, t_291, matmul_582], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_39, (1, 64), (1, 1), 1984), buf640, out=buf644)
            buf645 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_287], Original ATen: [aten.mm]
            extern_kernels.mm(buf643, buf644, out=buf645)
            buf646 = buf645; del buf645  # reuse
            # Topologically Sorted Source Nodes: [Q_296], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf646, buf640, 4096, stream=stream0)
            buf647 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12, chunk_8, view_8, q_9, q_10], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_38.run(buf454, buf647, 4194304, stream=stream0)
            buf648 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12, chunk_8, view_8, q_9, t_292, q_10], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf647, (65536, 64), (64, 1), 0), reinterpret_tensor(buf646, (64, 64), (1, 64), 0), out=buf648)
            buf649 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12, chunk_8, view_9, k_9, k_10], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_39.run(buf454, buf649, 4194304, stream=stream0)
            buf650 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12, chunk_8, view_9, k_9, t_292, k_10], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf649, (65536, 64), (64, 1), 0), reinterpret_tensor(buf646, (64, 64), (1, 64), 0), out=buf650)
            buf651 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, q_10, chunk_9, mul_333, neg_4, cat_15, mul_334, q_rot_2], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf648, buf209, buf651, 4194304, stream=stream0)
            buf652 = buf648; del buf648  # reuse
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, q_10, chunk_9, mul_333, neg_4, cat_15, mul_334, q_rot_2, q_11], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf651, (65536, 64), (64, 1), 0), buf646, out=buf652)
            buf653 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, k_10, chunk_10, mul_335, neg_5, cat_16, mul_336, k_rot_2], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf650, buf209, buf653, 4194304, stream=stream0)
            buf654 = buf650; del buf650  # reuse
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, k_10, chunk_10, mul_335, neg_5, cat_16, mul_336, k_rot_2, k_11], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf653, (65536, 64), (64, 1), 0), buf646, out=buf654)
            buf655 = buf436; del buf436  # reuse
            buf656 = buf435; del buf435  # reuse
            buf657 = empty_strided_cuda((256, 1, 4, 64, 64), (16384, 16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12, chunk_8, view_10, v_261, q_11, k_11, flex_attention_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_split_transpose_view_42.run(buf652, buf654, buf454, buf655, buf656, primals_15, primals_14, primals_20, primals_21, primals_16, primals_17, primals_18, primals_19, buf657, 1024, 1, 1, stream=stream0)
            buf660 = reinterpret_tensor(buf657, (256, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf657  # reuse
            # Topologically Sorted Source Nodes: [linear_12, chunk_8, view_10, v_261, q_11, k_11, flex_attention_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_43.run(buf660, buf655, buf656, 4194304, stream=stream0)
            buf661 = empty_strided_cuda((256, 4, 64), (256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12, chunk_8, view_10, v_261, q_11, k_11, flex_attention_2], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_44.run(buf655, buf656, buf661, 65536, stream=stream0)
            buf662 = empty_strided_cuda((256, 64, 4, 64), (16384, 256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_11, contiguous_2], Original ATen: [aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_45.run(buf660, buf662, 4194304, stream=stream0)
            buf663 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_11, contiguous_2, attn_4, attn_5], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf662, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_40, (256, 256), (1, 256), 0), out=buf663)
            buf664 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_5, linear_14], Original ATen: [aten._unsafe_view, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(primals_42, buf663, reinterpret_tensor(primals_41, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf664)
            del primals_42
            buf665 = buf230; del buf230  # reuse
            buf666 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf667 = reinterpret_tensor(buf666, (256, 64, 1), (64, 1, 1), 0); del buf666  # reuse
            buf668 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_3, linear_9, gate_attn_1, mul_267, x_4, ffn_out_1, x_5, attn_5, linear_14, gate_attn_2, mul_401, x_6, x_norm_5], Original ATen: [aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.add, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_51.run(buf665, buf667, buf443, buf444, buf450, buf663, buf664, buf668, 16384, 256, stream=stream0)
            buf669 = empty_strided_cuda((16384, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_2], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf668, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_43, (256, 2048), (1, 256), 0), out=buf669)
            buf670 = empty_strided_cuda((256, 64, 1024), (65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_2, chunk_11, silu_2, mul_402], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_split_47.run(buf669, buf670, 16777216, stream=stream0)
            buf671 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [x12_2, chunk_11, silu_2, mul_402, ffn_out_2], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf670, (16384, 1024), (1024, 1), 0), reinterpret_tensor(primals_44, (1024, 256), (1, 1024), 0), out=buf671)
            buf672 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf673 = reinterpret_tensor(buf672, (256, 64, 1), (64, 1, 1), 0); del buf672  # reuse
            buf674 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_out_2, x_7, x_norm_6], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_52.run(buf673, buf665, buf671, buf674, 16384, 256, stream=stream0)
            buf675 = empty_strided_cuda((16384, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf674, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_45, (256, 768), (1, 256), 0), out=buf675)
            buf676 = empty_strided_cuda((), (), torch.float32)
            buf677 = buf676; del buf676  # reuse
            buf678 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_432, v_392, pow_385, sum_385, v_norm_sq_384, truediv_387, mul_403], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_4.run(buf677, primals_46, buf678, 1, 64, stream=stream0)
            buf679 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_432, v_392, t_390, matmul_780], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 0), buf12, out=buf679)
            buf680 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_384], Original ATen: [aten.mm]
            extern_kernels.mm(buf678, buf679, out=buf680)
            buf681 = buf680; del buf680  # reuse
            # Topologically Sorted Source Nodes: [Q_397], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_5.run(buf681, 4096, stream=stream0)
            buf682 = empty_strided_cuda((), (), torch.float32)
            buf683 = buf682; del buf682  # reuse
            buf684 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_433, v_393, pow_386, sum_386, v_norm_sq_385, truediv_388, mul_404], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_6.run(buf683, primals_46, buf684, 1, 64, stream=stream0)
            buf685 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_433, v_393, t_391, matmul_782], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 64), buf681, out=buf685)
            buf686 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_385], Original ATen: [aten.mm]
            extern_kernels.mm(buf684, buf685, out=buf686)
            buf687 = buf686; del buf686  # reuse
            # Topologically Sorted Source Nodes: [Q_398], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf687, buf681, 4096, stream=stream0)
            buf688 = empty_strided_cuda((), (), torch.float32)
            buf689 = buf688; del buf688  # reuse
            buf690 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_434, v_394, pow_387, sum_387, v_norm_sq_386, truediv_389, mul_405], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_8.run(buf689, primals_46, buf690, 1, 64, stream=stream0)
            buf691 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_434, v_394, t_392, matmul_784], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 128), buf687, out=buf691)
            buf692 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_386], Original ATen: [aten.mm]
            extern_kernels.mm(buf690, buf691, out=buf692)
            buf693 = buf692; del buf692  # reuse
            # Topologically Sorted Source Nodes: [Q_399], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf693, buf687, 4096, stream=stream0)
            buf694 = empty_strided_cuda((), (), torch.float32)
            buf695 = buf694; del buf694  # reuse
            buf696 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_435, v_395, pow_388, sum_388, v_norm_sq_387, truediv_390, mul_406], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_9.run(buf695, primals_46, buf696, 1, 64, stream=stream0)
            buf697 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_435, v_395, t_393, matmul_786], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 192), buf693, out=buf697)
            buf698 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_387], Original ATen: [aten.mm]
            extern_kernels.mm(buf696, buf697, out=buf698)
            buf699 = buf698; del buf698  # reuse
            # Topologically Sorted Source Nodes: [Q_400], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf699, buf693, 4096, stream=stream0)
            buf700 = empty_strided_cuda((), (), torch.float32)
            buf701 = buf700; del buf700  # reuse
            buf702 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_436, v_396, pow_389, sum_389, v_norm_sq_388, truediv_391, mul_407], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_10.run(buf701, primals_46, buf702, 1, 64, stream=stream0)
            buf703 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_436, v_396, t_394, matmul_788], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 256), buf699, out=buf703)
            buf704 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_388], Original ATen: [aten.mm]
            extern_kernels.mm(buf702, buf703, out=buf704)
            buf705 = buf704; del buf704  # reuse
            # Topologically Sorted Source Nodes: [Q_401], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf705, buf699, 4096, stream=stream0)
            buf706 = empty_strided_cuda((), (), torch.float32)
            buf707 = buf706; del buf706  # reuse
            buf708 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_437, v_397, pow_390, sum_390, v_norm_sq_389, truediv_392, mul_408], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_11.run(buf707, primals_46, buf708, 1, 64, stream=stream0)
            buf709 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_437, v_397, t_395, matmul_790], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 320), buf705, out=buf709)
            buf710 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_389], Original ATen: [aten.mm]
            extern_kernels.mm(buf708, buf709, out=buf710)
            buf711 = buf710; del buf710  # reuse
            # Topologically Sorted Source Nodes: [Q_402], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf711, buf705, 4096, stream=stream0)
            buf712 = empty_strided_cuda((), (), torch.float32)
            buf713 = buf712; del buf712  # reuse
            buf714 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_438, v_398, pow_391, sum_391, v_norm_sq_390, truediv_393, mul_409], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_12.run(buf713, primals_46, buf714, 1, 64, stream=stream0)
            buf715 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_438, v_398, t_396, matmul_792], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 384), buf711, out=buf715)
            buf716 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_390], Original ATen: [aten.mm]
            extern_kernels.mm(buf714, buf715, out=buf716)
            buf717 = buf716; del buf716  # reuse
            # Topologically Sorted Source Nodes: [Q_403], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf717, buf711, 4096, stream=stream0)
            buf718 = empty_strided_cuda((), (), torch.float32)
            buf719 = buf718; del buf718  # reuse
            buf720 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_439, v_399, pow_392, sum_392, v_norm_sq_391, truediv_394, mul_410], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_13.run(buf719, primals_46, buf720, 1, 64, stream=stream0)
            buf721 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_439, v_399, t_397, matmul_794], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 448), buf717, out=buf721)
            buf722 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_391], Original ATen: [aten.mm]
            extern_kernels.mm(buf720, buf721, out=buf722)
            buf723 = buf722; del buf722  # reuse
            # Topologically Sorted Source Nodes: [Q_404], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf723, buf717, 4096, stream=stream0)
            buf724 = empty_strided_cuda((), (), torch.float32)
            buf725 = buf724; del buf724  # reuse
            buf726 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_440, v_400, pow_393, sum_393, v_norm_sq_392, truediv_395, mul_411], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_14.run(buf725, primals_46, buf726, 1, 64, stream=stream0)
            buf727 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_440, v_400, t_398, matmul_796], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 512), buf723, out=buf727)
            buf728 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_392], Original ATen: [aten.mm]
            extern_kernels.mm(buf726, buf727, out=buf728)
            buf729 = buf728; del buf728  # reuse
            # Topologically Sorted Source Nodes: [Q_405], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf729, buf723, 4096, stream=stream0)
            buf730 = empty_strided_cuda((), (), torch.float32)
            buf731 = buf730; del buf730  # reuse
            buf732 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_441, v_401, pow_394, sum_394, v_norm_sq_393, truediv_396, mul_412], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_15.run(buf731, primals_46, buf732, 1, 64, stream=stream0)
            buf733 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_441, v_401, t_399, matmul_798], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 576), buf729, out=buf733)
            buf734 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_393], Original ATen: [aten.mm]
            extern_kernels.mm(buf732, buf733, out=buf734)
            buf735 = buf734; del buf734  # reuse
            # Topologically Sorted Source Nodes: [Q_406], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf735, buf729, 4096, stream=stream0)
            buf736 = empty_strided_cuda((), (), torch.float32)
            buf737 = buf736; del buf736  # reuse
            buf738 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_442, v_402, pow_395, sum_395, v_norm_sq_394, truediv_397, mul_413], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_16.run(buf737, primals_46, buf738, 1, 64, stream=stream0)
            buf739 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_442, v_402, t_400, matmul_800], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 640), buf735, out=buf739)
            buf740 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_394], Original ATen: [aten.mm]
            extern_kernels.mm(buf738, buf739, out=buf740)
            buf741 = buf740; del buf740  # reuse
            # Topologically Sorted Source Nodes: [Q_407], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf741, buf735, 4096, stream=stream0)
            buf742 = empty_strided_cuda((), (), torch.float32)
            buf743 = buf742; del buf742  # reuse
            buf744 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_443, v_403, pow_396, sum_396, v_norm_sq_395, truediv_398, mul_414], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_17.run(buf743, primals_46, buf744, 1, 64, stream=stream0)
            buf745 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_443, v_403, t_401, matmul_802], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 704), buf741, out=buf745)
            buf746 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_395], Original ATen: [aten.mm]
            extern_kernels.mm(buf744, buf745, out=buf746)
            buf747 = buf746; del buf746  # reuse
            # Topologically Sorted Source Nodes: [Q_408], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf747, buf741, 4096, stream=stream0)
            buf748 = empty_strided_cuda((), (), torch.float32)
            buf749 = buf748; del buf748  # reuse
            buf750 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_444, v_404, pow_397, sum_397, v_norm_sq_396, truediv_399, mul_415], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_18.run(buf749, primals_46, buf750, 1, 64, stream=stream0)
            buf751 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_444, v_404, t_402, matmul_804], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 768), buf747, out=buf751)
            buf752 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_396], Original ATen: [aten.mm]
            extern_kernels.mm(buf750, buf751, out=buf752)
            buf753 = buf752; del buf752  # reuse
            # Topologically Sorted Source Nodes: [Q_409], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf753, buf747, 4096, stream=stream0)
            buf754 = empty_strided_cuda((), (), torch.float32)
            buf755 = buf754; del buf754  # reuse
            buf756 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_445, v_405, pow_398, sum_398, v_norm_sq_397, truediv_400, mul_416], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_19.run(buf755, primals_46, buf756, 1, 64, stream=stream0)
            buf757 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_445, v_405, t_403, matmul_806], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 832), buf753, out=buf757)
            buf758 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_397], Original ATen: [aten.mm]
            extern_kernels.mm(buf756, buf757, out=buf758)
            buf759 = buf758; del buf758  # reuse
            # Topologically Sorted Source Nodes: [Q_410], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf759, buf753, 4096, stream=stream0)
            buf760 = empty_strided_cuda((), (), torch.float32)
            buf761 = buf760; del buf760  # reuse
            buf762 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_446, v_406, pow_399, sum_399, v_norm_sq_398, truediv_401, mul_417], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_20.run(buf761, primals_46, buf762, 1, 64, stream=stream0)
            buf763 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_446, v_406, t_404, matmul_808], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 896), buf759, out=buf763)
            buf764 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_398], Original ATen: [aten.mm]
            extern_kernels.mm(buf762, buf763, out=buf764)
            buf765 = buf764; del buf764  # reuse
            # Topologically Sorted Source Nodes: [Q_411], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf765, buf759, 4096, stream=stream0)
            buf766 = empty_strided_cuda((), (), torch.float32)
            buf767 = buf766; del buf766  # reuse
            buf768 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_447, v_407, pow_400, sum_400, v_norm_sq_399, truediv_402, mul_418], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_21.run(buf767, primals_46, buf768, 1, 64, stream=stream0)
            buf769 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_447, v_407, t_405, matmul_810], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 960), buf765, out=buf769)
            buf770 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_399], Original ATen: [aten.mm]
            extern_kernels.mm(buf768, buf769, out=buf770)
            buf771 = buf770; del buf770  # reuse
            # Topologically Sorted Source Nodes: [Q_412], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf771, buf765, 4096, stream=stream0)
            buf772 = empty_strided_cuda((), (), torch.float32)
            buf773 = buf772; del buf772  # reuse
            buf774 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_448, v_408, pow_401, sum_401, v_norm_sq_400, truediv_403, mul_419], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_22.run(buf773, primals_46, buf774, 1, 64, stream=stream0)
            buf775 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_448, v_408, t_406, matmul_812], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1024), buf771, out=buf775)
            buf776 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_400], Original ATen: [aten.mm]
            extern_kernels.mm(buf774, buf775, out=buf776)
            buf777 = buf776; del buf776  # reuse
            # Topologically Sorted Source Nodes: [Q_413], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf777, buf771, 4096, stream=stream0)
            buf778 = empty_strided_cuda((), (), torch.float32)
            buf779 = buf778; del buf778  # reuse
            buf780 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_449, v_409, pow_402, sum_402, v_norm_sq_401, truediv_404, mul_420], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_23.run(buf779, primals_46, buf780, 1, 64, stream=stream0)
            buf781 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_449, v_409, t_407, matmul_814], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1088), buf777, out=buf781)
            buf782 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_401], Original ATen: [aten.mm]
            extern_kernels.mm(buf780, buf781, out=buf782)
            buf783 = buf782; del buf782  # reuse
            # Topologically Sorted Source Nodes: [Q_414], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf783, buf777, 4096, stream=stream0)
            buf784 = empty_strided_cuda((), (), torch.float32)
            buf785 = buf784; del buf784  # reuse
            buf786 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_450, v_410, pow_403, sum_403, v_norm_sq_402, truediv_405, mul_421], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_24.run(buf785, primals_46, buf786, 1, 64, stream=stream0)
            buf787 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_450, v_410, t_408, matmul_816], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1152), buf783, out=buf787)
            buf788 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_402], Original ATen: [aten.mm]
            extern_kernels.mm(buf786, buf787, out=buf788)
            buf789 = buf788; del buf788  # reuse
            # Topologically Sorted Source Nodes: [Q_415], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf789, buf783, 4096, stream=stream0)
            buf790 = empty_strided_cuda((), (), torch.float32)
            buf791 = buf790; del buf790  # reuse
            buf792 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_451, v_411, pow_404, sum_404, v_norm_sq_403, truediv_406, mul_422], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_25.run(buf791, primals_46, buf792, 1, 64, stream=stream0)
            buf793 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_451, v_411, t_409, matmul_818], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1216), buf789, out=buf793)
            buf794 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_403], Original ATen: [aten.mm]
            extern_kernels.mm(buf792, buf793, out=buf794)
            buf795 = buf794; del buf794  # reuse
            # Topologically Sorted Source Nodes: [Q_416], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf795, buf789, 4096, stream=stream0)
            buf796 = empty_strided_cuda((), (), torch.float32)
            buf797 = buf796; del buf796  # reuse
            buf798 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_452, v_412, pow_405, sum_405, v_norm_sq_404, truediv_407, mul_423], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_26.run(buf797, primals_46, buf798, 1, 64, stream=stream0)
            buf799 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_452, v_412, t_410, matmul_820], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1280), buf795, out=buf799)
            buf800 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_404], Original ATen: [aten.mm]
            extern_kernels.mm(buf798, buf799, out=buf800)
            buf801 = buf800; del buf800  # reuse
            # Topologically Sorted Source Nodes: [Q_417], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf801, buf795, 4096, stream=stream0)
            buf802 = empty_strided_cuda((), (), torch.float32)
            buf803 = buf802; del buf802  # reuse
            buf804 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_453, v_413, pow_406, sum_406, v_norm_sq_405, truediv_408, mul_424], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_27.run(buf803, primals_46, buf804, 1, 64, stream=stream0)
            buf805 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_453, v_413, t_411, matmul_822], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1344), buf801, out=buf805)
            buf806 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_405], Original ATen: [aten.mm]
            extern_kernels.mm(buf804, buf805, out=buf806)
            buf807 = buf806; del buf806  # reuse
            # Topologically Sorted Source Nodes: [Q_418], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf807, buf801, 4096, stream=stream0)
            buf808 = empty_strided_cuda((), (), torch.float32)
            buf809 = buf808; del buf808  # reuse
            buf810 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_454, v_414, pow_407, sum_407, v_norm_sq_406, truediv_409, mul_425], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_28.run(buf809, primals_46, buf810, 1, 64, stream=stream0)
            buf811 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_454, v_414, t_412, matmul_824], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1408), buf807, out=buf811)
            buf812 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_406], Original ATen: [aten.mm]
            extern_kernels.mm(buf810, buf811, out=buf812)
            buf813 = buf812; del buf812  # reuse
            # Topologically Sorted Source Nodes: [Q_419], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf813, buf807, 4096, stream=stream0)
            buf814 = empty_strided_cuda((), (), torch.float32)
            buf815 = buf814; del buf814  # reuse
            buf816 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_455, v_415, pow_408, sum_408, v_norm_sq_407, truediv_410, mul_426], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_29.run(buf815, primals_46, buf816, 1, 64, stream=stream0)
            buf817 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_455, v_415, t_413, matmul_826], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1472), buf813, out=buf817)
            buf818 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_407], Original ATen: [aten.mm]
            extern_kernels.mm(buf816, buf817, out=buf818)
            buf819 = buf818; del buf818  # reuse
            # Topologically Sorted Source Nodes: [Q_420], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf819, buf813, 4096, stream=stream0)
            buf820 = empty_strided_cuda((), (), torch.float32)
            buf821 = buf820; del buf820  # reuse
            buf822 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_456, v_416, pow_409, sum_409, v_norm_sq_408, truediv_411, mul_427], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_30.run(buf821, primals_46, buf822, 1, 64, stream=stream0)
            buf823 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_456, v_416, t_414, matmul_828], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1536), buf819, out=buf823)
            buf824 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_408], Original ATen: [aten.mm]
            extern_kernels.mm(buf822, buf823, out=buf824)
            buf825 = buf824; del buf824  # reuse
            # Topologically Sorted Source Nodes: [Q_421], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf825, buf819, 4096, stream=stream0)
            buf826 = empty_strided_cuda((), (), torch.float32)
            buf827 = buf826; del buf826  # reuse
            buf828 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_457, v_417, pow_410, sum_410, v_norm_sq_409, truediv_412, mul_428], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_31.run(buf827, primals_46, buf828, 1, 64, stream=stream0)
            buf829 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_457, v_417, t_415, matmul_830], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1600), buf825, out=buf829)
            buf830 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_409], Original ATen: [aten.mm]
            extern_kernels.mm(buf828, buf829, out=buf830)
            buf831 = buf830; del buf830  # reuse
            # Topologically Sorted Source Nodes: [Q_422], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf831, buf825, 4096, stream=stream0)
            buf832 = empty_strided_cuda((), (), torch.float32)
            buf833 = buf832; del buf832  # reuse
            buf834 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_458, v_418, pow_411, sum_411, v_norm_sq_410, truediv_413, mul_429], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_32.run(buf833, primals_46, buf834, 1, 64, stream=stream0)
            buf835 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_458, v_418, t_416, matmul_832], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1664), buf831, out=buf835)
            buf836 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_410], Original ATen: [aten.mm]
            extern_kernels.mm(buf834, buf835, out=buf836)
            buf837 = buf836; del buf836  # reuse
            # Topologically Sorted Source Nodes: [Q_423], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf837, buf831, 4096, stream=stream0)
            buf838 = empty_strided_cuda((), (), torch.float32)
            buf839 = buf838; del buf838  # reuse
            buf840 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_459, v_419, pow_412, sum_412, v_norm_sq_411, truediv_414, mul_430], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_33.run(buf839, primals_46, buf840, 1, 64, stream=stream0)
            buf841 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_459, v_419, t_417, matmul_834], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1728), buf837, out=buf841)
            buf842 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_411], Original ATen: [aten.mm]
            extern_kernels.mm(buf840, buf841, out=buf842)
            buf843 = buf842; del buf842  # reuse
            # Topologically Sorted Source Nodes: [Q_424], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf843, buf837, 4096, stream=stream0)
            buf844 = empty_strided_cuda((), (), torch.float32)
            buf845 = buf844; del buf844  # reuse
            buf846 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_460, v_420, pow_413, sum_413, v_norm_sq_412, truediv_415, mul_431], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_34.run(buf845, primals_46, buf846, 1, 64, stream=stream0)
            buf847 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_460, v_420, t_418, matmul_836], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1792), buf843, out=buf847)
            buf848 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_412], Original ATen: [aten.mm]
            extern_kernels.mm(buf846, buf847, out=buf848)
            buf849 = buf848; del buf848  # reuse
            # Topologically Sorted Source Nodes: [Q_425], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf849, buf843, 4096, stream=stream0)
            buf850 = empty_strided_cuda((), (), torch.float32)
            buf851 = buf850; del buf850  # reuse
            buf852 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_461, v_421, pow_414, sum_414, v_norm_sq_413, truediv_416, mul_432], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_35.run(buf851, primals_46, buf852, 1, 64, stream=stream0)
            buf853 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_461, v_421, t_419, matmul_838], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1856), buf849, out=buf853)
            buf854 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_413], Original ATen: [aten.mm]
            extern_kernels.mm(buf852, buf853, out=buf854)
            buf855 = buf854; del buf854  # reuse
            # Topologically Sorted Source Nodes: [Q_426], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf855, buf849, 4096, stream=stream0)
            buf856 = empty_strided_cuda((), (), torch.float32)
            buf857 = buf856; del buf856  # reuse
            buf858 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_462, v_422, pow_415, sum_415, v_norm_sq_414, truediv_417, mul_433], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_36.run(buf857, primals_46, buf858, 1, 64, stream=stream0)
            buf859 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_462, v_422, t_420, matmul_840], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1920), buf855, out=buf859)
            buf860 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_414], Original ATen: [aten.mm]
            extern_kernels.mm(buf858, buf859, out=buf860)
            buf861 = buf860; del buf860  # reuse
            # Topologically Sorted Source Nodes: [Q_427], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf861, buf855, 4096, stream=stream0)
            buf862 = empty_strided_cuda((), (), torch.float32)
            buf863 = buf862; del buf862  # reuse
            buf864 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_463, v_423, pow_416, sum_416, v_norm_sq_415, truediv_418, mul_434], Original ATen: [aten.select, aten.unsqueeze, aten.pow, aten.sum, aten.add, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_pow_reciprocal_select_sum_unsqueeze_37.run(buf863, primals_46, buf864, 1, 64, stream=stream0)
            buf865 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_463, v_423, t_421, matmul_842], Original ATen: [aten.select, aten.unsqueeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(primals_46, (1, 64), (1, 1), 1984), buf861, out=buf865)
            buf866 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [term_415], Original ATen: [aten.mm]
            extern_kernels.mm(buf864, buf865, out=buf866)
            buf867 = buf866; del buf866  # reuse
            # Topologically Sorted Source Nodes: [Q_428], Original ATen: [aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sub_7.run(buf867, buf861, 4096, stream=stream0)
            buf868 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_17, chunk_12, view_12, q_13, q_14], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_38.run(buf675, buf868, 4194304, stream=stream0)
            buf869 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_17, chunk_12, view_12, q_13, t_422, q_14], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf868, (65536, 64), (64, 1), 0), reinterpret_tensor(buf867, (64, 64), (1, 64), 0), out=buf869)
            buf870 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_17, chunk_12, view_13, k_13, k_14], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_split_transpose_view_39.run(buf675, buf870, 4194304, stream=stream0)
            buf871 = empty_strided_cuda((65536, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_17, chunk_12, view_13, k_13, t_422, k_14], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten.t, aten.clone, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf870, (65536, 64), (64, 1), 0), reinterpret_tensor(buf867, (64, 64), (1, 64), 0), out=buf871)
            buf872 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, q_14, chunk_13, mul_467, neg_6, cat_20, mul_468, q_rot_3], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf869, buf209, buf872, 4194304, stream=stream0)
            buf873 = buf869; del buf869  # reuse
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, q_14, chunk_13, mul_467, neg_6, cat_20, mul_468, q_rot_3, q_15], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf872, (65536, 64), (64, 1), 0), buf867, out=buf873)
            buf874 = empty_strided_cuda((256, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, k_14, chunk_14, mul_469, neg_7, cat_21, mul_470, k_rot_3], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_cat_cos_mul_neg_sin_split_unsqueeze_41.run(buf871, buf209, buf874, 4194304, stream=stream0)
            buf875 = buf871; del buf871  # reuse
            # Topologically Sorted Source Nodes: [emb, cos_1, unsqueeze_66, cos_2, sin_1, unsqueeze_68, sin_2, k_14, chunk_14, mul_469, neg_7, cat_21, mul_470, k_rot_3, k_15], Original ATen: [aten.cat, aten.cos, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.split, aten.mul, aten.neg, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf874, (65536, 64), (64, 1), 0), buf867, out=buf875)
            buf876 = buf656; del buf656  # reuse
            buf877 = buf655; del buf655  # reuse
            buf878 = empty_strided_cuda((256, 1, 4, 64, 64), (16384, 16384, 4096, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_17, chunk_12, view_14, v_391, q_15, k_15, flex_attention_3], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_tem_fused__unsafe_view_split_transpose_view_53.run(buf873, buf875, buf675, buf876, buf877, primals_48, primals_47, primals_49, primals_50, primals_16, primals_17, buf878, 1024, 1, 1, stream=stream0)
            buf881 = reinterpret_tensor(buf878, (256, 4, 64, 64), (16384, 4096, 64, 1), 0); del buf878  # reuse
            # Topologically Sorted Source Nodes: [linear_17, chunk_12, view_14, v_391, q_15, k_15, flex_attention_3], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_43.run(buf881, buf876, buf877, 4194304, stream=stream0)
            buf882 = empty_strided_cuda((256, 4, 64), (256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_17, chunk_12, view_14, v_391, q_15, k_15, flex_attention_3], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_split_transpose_view_44.run(buf876, buf877, buf882, 65536, stream=stream0)
            del buf876
            del buf877
            buf883 = empty_strided_cuda((256, 64, 4, 64), (16384, 256, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_15, contiguous_3], Original ATen: [aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_45.run(buf881, buf883, 4194304, stream=stream0)
            buf884 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose_15, contiguous_3, attn_6, attn_7], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf883, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_55, (256, 256), (1, 256), 0), out=buf884)
            buf885 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_7, linear_19], Original ATen: [aten._unsafe_view, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(primals_57, buf884, reinterpret_tensor(primals_56, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf885)
            del primals_57
            buf886 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf887 = reinterpret_tensor(buf886, (256, 64, 1), (64, 1, 1), 0); del buf886  # reuse
            buf888 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_out_2, x_7, attn_7, linear_19, gate_attn_3, mul_535, x_8, x_norm_7], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.sigmoid, aten.mul, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_sigmoid_view_54.run(buf887, buf665, buf671, buf884, buf885, buf888, 16384, 256, stream=stream0)
            buf889 = empty_strided_cuda((16384, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_3], Original ATen: [aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf888, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_58, (256, 2048), (1, 256), 0), out=buf889)
            buf890 = empty_strided_cuda((256, 64, 1024), (65536, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_3, chunk_15, silu_3, mul_536], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_split_47.run(buf889, buf890, 16777216, stream=stream0)
            buf891 = empty_strided_cuda((16384, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x12_3, chunk_15, silu_3, mul_536, ffn_out_3], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf890, (16384, 1024), (1024, 1), 0), reinterpret_tensor(primals_59, (1024, 256), (1, 1024), 0), out=buf891)
            buf892 = buf665; del buf665  # reuse
            buf893 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf894 = reinterpret_tensor(buf893, (256, 64, 1), (64, 1, 1), 0); del buf893  # reuse
            buf895 = buf892; del buf892  # reuse
            buf896 = empty_strided_cuda((256, 64, 1), (64, 1, 1), torch.float32)
            buf897 = empty_strided_cuda((256, 64, 1), (64, 1, 16384), torch.float32)
            buf899 = reinterpret_tensor(buf897, (256, 64, 1), (64, 1, 1), 0); del buf897  # reuse
            buf900 = empty_strided_cuda((256, 64, 256), (16384, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ffn_out_2, x_7, attn_7, linear_19, gate_attn_3, mul_535, x_8, ffn_out_3, x_9, x_10, input_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.sigmoid, aten.mul, aten.pow, aten.mean, aten.rsqrt, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_native_layer_norm_pow_rsqrt_sigmoid_view_55.run(buf895, buf894, buf899, buf671, buf884, buf885, buf891, primals_60, primals_61, buf896, buf900, 16384, 256, stream=stream0)
            del buf671
            del buf891
            del primals_61
            buf901 = empty_strided_cuda((16384, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf900, (16384, 256), (256, 1), 0), reinterpret_tensor(primals_62, (256, 12), (1, 256), 0), out=buf901)
            buf902 = empty_strided_cuda((256, 3, 8, 2, 8, 2), (768, 256, 32, 16, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_4, view_16, permute_1, reshape_1], Original ATen: [aten.addmm, aten.view, aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_clone_permute_view_56.run(buf901, primals_63, buf902, 196608, stream=stream0)
            del buf901
            del primals_63
        return (reinterpret_tensor(buf902, (256, 3, 16, 16), (768, 256, 16, 1), 0), primals_6, primals_8, primals_10, primals_11, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_58, primals_59, primals_60, primals_62, reinterpret_tensor(buf0, (16384, 56), (56, 1), 0), buf1, buf2, buf5, reinterpret_tensor(buf6, (16384, 256), (256, 1), 0), buf9, buf10, reinterpret_tensor(buf11, (256, 4, 64, 64), (49152, 64, 768, 1), 512), buf12, buf14, buf18, buf20, buf24, buf26, buf30, buf32, buf36, buf38, buf42, buf44, buf48, buf50, buf54, buf56, buf60, buf62, buf66, buf68, buf72, buf74, buf78, buf80, buf84, buf86, buf90, buf92, buf96, buf98, buf102, buf104, buf108, buf110, buf114, buf116, buf120, buf122, buf126, buf128, buf132, buf134, buf138, buf140, buf144, buf146, buf150, buf152, buf156, buf158, buf162, buf164, buf168, buf170, buf174, buf176, buf180, buf182, buf186, buf188, buf192, buf194, buf198, buf200, reinterpret_tensor(buf204, (64, 64), (1, 64), 0), reinterpret_tensor(buf205, (65536, 64), (64, 1), 0), reinterpret_tensor(buf207, (65536, 64), (64, 1), 0), buf209, reinterpret_tensor(buf211, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), reinterpret_tensor(buf213, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), buf219, buf220, reinterpret_tensor(buf221, (16384, 256), (256, 1), 0), buf222, buf223, buf225, buf226, buf227, reinterpret_tensor(buf228, (16384, 1024), (1024, 1), 0), buf232, buf233, reinterpret_tensor(buf234, (256, 4, 64, 64), (49152, 64, 768, 1), 512), buf236, buf240, buf242, buf246, buf248, buf252, buf254, buf258, buf260, buf264, buf266, buf270, buf272, buf276, buf278, buf282, buf284, buf288, buf290, buf294, buf296, buf300, buf302, buf306, buf308, buf312, buf314, buf318, buf320, buf324, buf326, buf330, buf332, buf336, buf338, buf342, buf344, buf348, buf350, buf354, buf356, buf360, buf362, buf366, buf368, buf372, buf374, buf378, buf380, buf384, buf386, buf390, buf392, buf396, buf398, buf402, buf404, buf408, buf410, buf414, buf416, buf420, buf422, reinterpret_tensor(buf426, (64, 64), (1, 64), 0), reinterpret_tensor(buf427, (65536, 64), (64, 1), 0), reinterpret_tensor(buf429, (65536, 64), (64, 1), 0), reinterpret_tensor(buf432, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), reinterpret_tensor(buf434, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), buf440, buf441, reinterpret_tensor(buf442, (16384, 256), (256, 1), 0), buf443, buf444, buf446, buf447, buf448, reinterpret_tensor(buf449, (16384, 1024), (1024, 1), 0), buf452, buf453, reinterpret_tensor(buf454, (256, 4, 64, 64), (49152, 64, 768, 1), 512), buf456, buf460, buf462, buf466, buf468, buf472, buf474, buf478, buf480, buf484, buf486, buf490, buf492, buf496, buf498, buf502, buf504, buf508, buf510, buf514, buf516, buf520, buf522, buf526, buf528, buf532, buf534, buf538, buf540, buf544, buf546, buf550, buf552, buf556, buf558, buf562, buf564, buf568, buf570, buf574, buf576, buf580, buf582, buf586, buf588, buf592, buf594, buf598, buf600, buf604, buf606, buf610, buf612, buf616, buf618, buf622, buf624, buf628, buf630, buf634, buf636, buf640, buf642, reinterpret_tensor(buf646, (64, 64), (1, 64), 0), reinterpret_tensor(buf647, (65536, 64), (64, 1), 0), reinterpret_tensor(buf649, (65536, 64), (64, 1), 0), reinterpret_tensor(buf652, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), reinterpret_tensor(buf654, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), buf660, buf661, reinterpret_tensor(buf662, (16384, 256), (256, 1), 0), buf663, buf664, buf667, buf668, buf669, reinterpret_tensor(buf670, (16384, 1024), (1024, 1), 0), buf673, buf674, reinterpret_tensor(buf675, (256, 4, 64, 64), (49152, 64, 768, 1), 512), buf677, buf681, buf683, buf687, buf689, buf693, buf695, buf699, buf701, buf705, buf707, buf711, buf713, buf717, buf719, buf723, buf725, buf729, buf731, buf735, buf737, buf741, buf743, buf747, buf749, buf753, buf755, buf759, buf761, buf765, buf767, buf771, buf773, buf777, buf779, buf783, buf785, buf789, buf791, buf795, buf797, buf801, buf803, buf807, buf809, buf813, buf815, buf819, buf821, buf825, buf827, buf831, buf833, buf837, buf839, buf843, buf845, buf849, buf851, buf855, buf857, buf861, buf863, reinterpret_tensor(buf867, (64, 64), (1, 64), 0), reinterpret_tensor(buf868, (65536, 64), (64, 1), 0), reinterpret_tensor(buf870, (65536, 64), (64, 1), 0), reinterpret_tensor(buf873, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), reinterpret_tensor(buf875, (256, 4, 64, 64), (16384, 4096, 64, 1), 0), buf881, buf882, reinterpret_tensor(buf883, (16384, 256), (256, 1), 0), buf884, buf885, buf887, buf888, buf889, reinterpret_tensor(buf890, (16384, 1024), (1024, 1), 0), buf894, buf895, buf896, buf899, reinterpret_tensor(buf900, (16384, 256), (256, 1), 0), reinterpret_tensor(buf874, (64, 65536), (1, 64), 0), reinterpret_tensor(buf864, (1, 64), (1, 1), 0), reinterpret_tensor(buf865, (64, 1), (1, 64), 0), reinterpret_tensor(buf858, (1, 64), (1, 1), 0), reinterpret_tensor(buf859, (64, 1), (1, 64), 0), reinterpret_tensor(buf852, (1, 64), (1, 1), 0), reinterpret_tensor(buf853, (64, 1), (1, 64), 0), reinterpret_tensor(buf846, (1, 64), (1, 1), 0), reinterpret_tensor(buf847, (64, 1), (1, 64), 0), reinterpret_tensor(buf840, (1, 64), (1, 1), 0), reinterpret_tensor(buf841, (64, 1), (1, 64), 0), reinterpret_tensor(buf834, (1, 64), (1, 1), 0), reinterpret_tensor(buf835, (64, 1), (1, 64), 0), reinterpret_tensor(buf828, (1, 64), (1, 1), 0), reinterpret_tensor(buf829, (64, 1), (1, 64), 0), reinterpret_tensor(buf822, (1, 64), (1, 1), 0), reinterpret_tensor(buf823, (64, 1), (1, 64), 0), reinterpret_tensor(buf816, (1, 64), (1, 1), 0), reinterpret_tensor(buf817, (64, 1), (1, 64), 0), reinterpret_tensor(buf810, (1, 64), (1, 1), 0), reinterpret_tensor(buf811, (64, 1), (1, 64), 0), reinterpret_tensor(buf804, (1, 64), (1, 1), 0), reinterpret_tensor(buf805, (64, 1), (1, 64), 0), reinterpret_tensor(buf798, (1, 64), (1, 1), 0), reinterpret_tensor(buf799, (64, 1), (1, 64), 0), reinterpret_tensor(buf792, (1, 64), (1, 1), 0), reinterpret_tensor(buf793, (64, 1), (1, 64), 0), reinterpret_tensor(buf786, (1, 64), (1, 1), 0), reinterpret_tensor(buf787, (64, 1), (1, 64), 0), reinterpret_tensor(buf780, (1, 64), (1, 1), 0), reinterpret_tensor(buf781, (64, 1), (1, 64), 0), reinterpret_tensor(buf774, (1, 64), (1, 1), 0), reinterpret_tensor(buf775, (64, 1), (1, 64), 0), reinterpret_tensor(buf768, (1, 64), (1, 1), 0), reinterpret_tensor(buf769, (64, 1), (1, 64), 0), reinterpret_tensor(buf762, (1, 64), (1, 1), 0), reinterpret_tensor(buf763, (64, 1), (1, 64), 0), reinterpret_tensor(buf756, (1, 64), (1, 1), 0), reinterpret_tensor(buf757, (64, 1), (1, 64), 0), reinterpret_tensor(buf750, (1, 64), (1, 1), 0), reinterpret_tensor(buf751, (64, 1), (1, 64), 0), reinterpret_tensor(buf744, (1, 64), (1, 1), 0), reinterpret_tensor(buf745, (64, 1), (1, 64), 0), reinterpret_tensor(buf738, (1, 64), (1, 1), 0), reinterpret_tensor(buf739, (64, 1), (1, 64), 0), reinterpret_tensor(buf732, (1, 64), (1, 1), 0), reinterpret_tensor(buf733, (64, 1), (1, 64), 0), reinterpret_tensor(buf726, (1, 64), (1, 1), 0), reinterpret_tensor(buf727, (64, 1), (1, 64), 0), reinterpret_tensor(buf720, (1, 64), (1, 1), 0), reinterpret_tensor(buf721, (64, 1), (1, 64), 0), reinterpret_tensor(buf714, (1, 64), (1, 1), 0), reinterpret_tensor(buf715, (64, 1), (1, 64), 0), reinterpret_tensor(buf708, (1, 64), (1, 1), 0), reinterpret_tensor(buf709, (64, 1), (1, 64), 0), reinterpret_tensor(buf702, (1, 64), (1, 1), 0), reinterpret_tensor(buf703, (64, 1), (1, 64), 0), reinterpret_tensor(buf696, (1, 64), (1, 1), 0), reinterpret_tensor(buf697, (64, 1), (1, 64), 0), reinterpret_tensor(buf690, (1, 64), (1, 1), 0), reinterpret_tensor(buf691, (64, 1), (1, 64), 0), reinterpret_tensor(buf684, (1, 64), (1, 1), 0), reinterpret_tensor(buf685, (64, 1), (1, 64), 0), reinterpret_tensor(buf678, (1, 64), (1, 1), 0), reinterpret_tensor(buf679, (64, 1), (1, 64), 0), reinterpret_tensor(buf872, (64, 65536), (1, 64), 0), reinterpret_tensor(buf653, (64, 65536), (1, 64), 0), reinterpret_tensor(buf643, (1, 64), (1, 1), 0), reinterpret_tensor(buf644, (64, 1), (1, 64), 0), reinterpret_tensor(buf637, (1, 64), (1, 1), 0), reinterpret_tensor(buf638, (64, 1), (1, 64), 0), reinterpret_tensor(buf631, (1, 64), (1, 1), 0), reinterpret_tensor(buf632, (64, 1), (1, 64), 0), reinterpret_tensor(buf625, (1, 64), (1, 1), 0), reinterpret_tensor(buf626, (64, 1), (1, 64), 0), reinterpret_tensor(buf619, (1, 64), (1, 1), 0), reinterpret_tensor(buf620, (64, 1), (1, 64), 0), reinterpret_tensor(buf613, (1, 64), (1, 1), 0), reinterpret_tensor(buf614, (64, 1), (1, 64), 0), reinterpret_tensor(buf607, (1, 64), (1, 1), 0), reinterpret_tensor(buf608, (64, 1), (1, 64), 0), reinterpret_tensor(buf601, (1, 64), (1, 1), 0), reinterpret_tensor(buf602, (64, 1), (1, 64), 0), reinterpret_tensor(buf595, (1, 64), (1, 1), 0), reinterpret_tensor(buf596, (64, 1), (1, 64), 0), reinterpret_tensor(buf589, (1, 64), (1, 1), 0), reinterpret_tensor(buf590, (64, 1), (1, 64), 0), reinterpret_tensor(buf583, (1, 64), (1, 1), 0), reinterpret_tensor(buf584, (64, 1), (1, 64), 0), reinterpret_tensor(buf577, (1, 64), (1, 1), 0), reinterpret_tensor(buf578, (64, 1), (1, 64), 0), reinterpret_tensor(buf571, (1, 64), (1, 1), 0), reinterpret_tensor(buf572, (64, 1), (1, 64), 0), reinterpret_tensor(buf565, (1, 64), (1, 1), 0), reinterpret_tensor(buf566, (64, 1), (1, 64), 0), reinterpret_tensor(buf559, (1, 64), (1, 1), 0), reinterpret_tensor(buf560, (64, 1), (1, 64), 0), reinterpret_tensor(buf553, (1, 64), (1, 1), 0), reinterpret_tensor(buf554, (64, 1), (1, 64), 0), reinterpret_tensor(buf547, (1, 64), (1, 1), 0), reinterpret_tensor(buf548, (64, 1), (1, 64), 0), reinterpret_tensor(buf541, (1, 64), (1, 1), 0), reinterpret_tensor(buf542, (64, 1), (1, 64), 0), reinterpret_tensor(buf535, (1, 64), (1, 1), 0), reinterpret_tensor(buf536, (64, 1), (1, 64), 0), reinterpret_tensor(buf529, (1, 64), (1, 1), 0), reinterpret_tensor(buf530, (64, 1), (1, 64), 0), reinterpret_tensor(buf523, (1, 64), (1, 1), 0), reinterpret_tensor(buf524, (64, 1), (1, 64), 0), reinterpret_tensor(buf517, (1, 64), (1, 1), 0), reinterpret_tensor(buf518, (64, 1), (1, 64), 0), reinterpret_tensor(buf511, (1, 64), (1, 1), 0), reinterpret_tensor(buf512, (64, 1), (1, 64), 0), reinterpret_tensor(buf505, (1, 64), (1, 1), 0), reinterpret_tensor(buf506, (64, 1), (1, 64), 0), reinterpret_tensor(buf499, (1, 64), (1, 1), 0), reinterpret_tensor(buf500, (64, 1), (1, 64), 0), reinterpret_tensor(buf493, (1, 64), (1, 1), 0), reinterpret_tensor(buf494, (64, 1), (1, 64), 0), reinterpret_tensor(buf487, (1, 64), (1, 1), 0), reinterpret_tensor(buf488, (64, 1), (1, 64), 0), reinterpret_tensor(buf481, (1, 64), (1, 1), 0), reinterpret_tensor(buf482, (64, 1), (1, 64), 0), reinterpret_tensor(buf475, (1, 64), (1, 1), 0), reinterpret_tensor(buf476, (64, 1), (1, 64), 0), reinterpret_tensor(buf469, (1, 64), (1, 1), 0), reinterpret_tensor(buf470, (64, 1), (1, 64), 0), reinterpret_tensor(buf463, (1, 64), (1, 1), 0), reinterpret_tensor(buf464, (64, 1), (1, 64), 0), reinterpret_tensor(buf457, (1, 64), (1, 1), 0), reinterpret_tensor(buf458, (64, 1), (1, 64), 0), reinterpret_tensor(buf651, (64, 65536), (1, 64), 0), reinterpret_tensor(buf433, (64, 65536), (1, 64), 0), reinterpret_tensor(buf423, (1, 64), (1, 1), 0), reinterpret_tensor(buf424, (64, 1), (1, 64), 0), reinterpret_tensor(buf417, (1, 64), (1, 1), 0), reinterpret_tensor(buf418, (64, 1), (1, 64), 0), reinterpret_tensor(buf411, (1, 64), (1, 1), 0), reinterpret_tensor(buf412, (64, 1), (1, 64), 0), reinterpret_tensor(buf405, (1, 64), (1, 1), 0), reinterpret_tensor(buf406, (64, 1), (1, 64), 0), reinterpret_tensor(buf399, (1, 64), (1, 1), 0), reinterpret_tensor(buf400, (64, 1), (1, 64), 0), reinterpret_tensor(buf393, (1, 64), (1, 1), 0), reinterpret_tensor(buf394, (64, 1), (1, 64), 0), reinterpret_tensor(buf387, (1, 64), (1, 1), 0), reinterpret_tensor(buf388, (64, 1), (1, 64), 0), reinterpret_tensor(buf381, (1, 64), (1, 1), 0), reinterpret_tensor(buf382, (64, 1), (1, 64), 0), reinterpret_tensor(buf375, (1, 64), (1, 1), 0), reinterpret_tensor(buf376, (64, 1), (1, 64), 0), reinterpret_tensor(buf369, (1, 64), (1, 1), 0), reinterpret_tensor(buf370, (64, 1), (1, 64), 0), reinterpret_tensor(buf363, (1, 64), (1, 1), 0), reinterpret_tensor(buf364, (64, 1), (1, 64), 0), reinterpret_tensor(buf357, (1, 64), (1, 1), 0), reinterpret_tensor(buf358, (64, 1), (1, 64), 0), reinterpret_tensor(buf351, (1, 64), (1, 1), 0), reinterpret_tensor(buf352, (64, 1), (1, 64), 0), reinterpret_tensor(buf345, (1, 64), (1, 1), 0), reinterpret_tensor(buf346, (64, 1), (1, 64), 0), reinterpret_tensor(buf339, (1, 64), (1, 1), 0), reinterpret_tensor(buf340, (64, 1), (1, 64), 0), reinterpret_tensor(buf333, (1, 64), (1, 1), 0), reinterpret_tensor(buf334, (64, 1), (1, 64), 0), reinterpret_tensor(buf327, (1, 64), (1, 1), 0), reinterpret_tensor(buf328, (64, 1), (1, 64), 0), reinterpret_tensor(buf321, (1, 64), (1, 1), 0), reinterpret_tensor(buf322, (64, 1), (1, 64), 0), reinterpret_tensor(buf315, (1, 64), (1, 1), 0), reinterpret_tensor(buf316, (64, 1), (1, 64), 0), reinterpret_tensor(buf309, (1, 64), (1, 1), 0), reinterpret_tensor(buf310, (64, 1), (1, 64), 0), reinterpret_tensor(buf303, (1, 64), (1, 1), 0), reinterpret_tensor(buf304, (64, 1), (1, 64), 0), reinterpret_tensor(buf297, (1, 64), (1, 1), 0), reinterpret_tensor(buf298, (64, 1), (1, 64), 0), reinterpret_tensor(buf291, (1, 64), (1, 1), 0), reinterpret_tensor(buf292, (64, 1), (1, 64), 0), reinterpret_tensor(buf285, (1, 64), (1, 1), 0), reinterpret_tensor(buf286, (64, 1), (1, 64), 0), reinterpret_tensor(buf279, (1, 64), (1, 1), 0), reinterpret_tensor(buf280, (64, 1), (1, 64), 0), reinterpret_tensor(buf273, (1, 64), (1, 1), 0), reinterpret_tensor(buf274, (64, 1), (1, 64), 0), reinterpret_tensor(buf267, (1, 64), (1, 1), 0), reinterpret_tensor(buf268, (64, 1), (1, 64), 0), reinterpret_tensor(buf261, (1, 64), (1, 1), 0), reinterpret_tensor(buf262, (64, 1), (1, 64), 0), reinterpret_tensor(buf255, (1, 64), (1, 1), 0), reinterpret_tensor(buf256, (64, 1), (1, 64), 0), reinterpret_tensor(buf249, (1, 64), (1, 1), 0), reinterpret_tensor(buf250, (64, 1), (1, 64), 0), reinterpret_tensor(buf243, (1, 64), (1, 1), 0), reinterpret_tensor(buf244, (64, 1), (1, 64), 0), reinterpret_tensor(buf237, (1, 64), (1, 1), 0), reinterpret_tensor(buf238, (64, 1), (1, 64), 0), reinterpret_tensor(buf431, (64, 65536), (1, 64), 0), reinterpret_tensor(buf212, (64, 65536), (1, 64), 0), reinterpret_tensor(buf201, (1, 64), (1, 1), 0), reinterpret_tensor(buf202, (64, 1), (1, 64), 0), reinterpret_tensor(buf195, (1, 64), (1, 1), 0), reinterpret_tensor(buf196, (64, 1), (1, 64), 0), reinterpret_tensor(buf189, (1, 64), (1, 1), 0), reinterpret_tensor(buf190, (64, 1), (1, 64), 0), reinterpret_tensor(buf183, (1, 64), (1, 1), 0), reinterpret_tensor(buf184, (64, 1), (1, 64), 0), reinterpret_tensor(buf177, (1, 64), (1, 1), 0), reinterpret_tensor(buf178, (64, 1), (1, 64), 0), reinterpret_tensor(buf171, (1, 64), (1, 1), 0), reinterpret_tensor(buf172, (64, 1), (1, 64), 0), reinterpret_tensor(buf165, (1, 64), (1, 1), 0), reinterpret_tensor(buf166, (64, 1), (1, 64), 0), reinterpret_tensor(buf159, (1, 64), (1, 1), 0), reinterpret_tensor(buf160, (64, 1), (1, 64), 0), reinterpret_tensor(buf153, (1, 64), (1, 1), 0), reinterpret_tensor(buf154, (64, 1), (1, 64), 0), reinterpret_tensor(buf147, (1, 64), (1, 1), 0), reinterpret_tensor(buf148, (64, 1), (1, 64), 0), reinterpret_tensor(buf141, (1, 64), (1, 1), 0), reinterpret_tensor(buf142, (64, 1), (1, 64), 0), reinterpret_tensor(buf135, (1, 64), (1, 1), 0), reinterpret_tensor(buf136, (64, 1), (1, 64), 0), reinterpret_tensor(buf129, (1, 64), (1, 1), 0), reinterpret_tensor(buf130, (64, 1), (1, 64), 0), reinterpret_tensor(buf123, (1, 64), (1, 1), 0), reinterpret_tensor(buf124, (64, 1), (1, 64), 0), reinterpret_tensor(buf117, (1, 64), (1, 1), 0), reinterpret_tensor(buf118, (64, 1), (1, 64), 0), reinterpret_tensor(buf111, (1, 64), (1, 1), 0), reinterpret_tensor(buf112, (64, 1), (1, 64), 0), reinterpret_tensor(buf105, (1, 64), (1, 1), 0), reinterpret_tensor(buf106, (64, 1), (1, 64), 0), reinterpret_tensor(buf99, (1, 64), (1, 1), 0), reinterpret_tensor(buf100, (64, 1), (1, 64), 0), reinterpret_tensor(buf93, (1, 64), (1, 1), 0), reinterpret_tensor(buf94, (64, 1), (1, 64), 0), reinterpret_tensor(buf87, (1, 64), (1, 1), 0), reinterpret_tensor(buf88, (64, 1), (1, 64), 0), reinterpret_tensor(buf81, (1, 64), (1, 1), 0), reinterpret_tensor(buf82, (64, 1), (1, 64), 0), reinterpret_tensor(buf75, (1, 64), (1, 1), 0), reinterpret_tensor(buf76, (64, 1), (1, 64), 0), reinterpret_tensor(buf69, (1, 64), (1, 1), 0), reinterpret_tensor(buf70, (64, 1), (1, 64), 0), reinterpret_tensor(buf63, (1, 64), (1, 1), 0), reinterpret_tensor(buf64, (64, 1), (1, 64), 0), reinterpret_tensor(buf57, (1, 64), (1, 1), 0), reinterpret_tensor(buf58, (64, 1), (1, 64), 0), reinterpret_tensor(buf51, (1, 64), (1, 1), 0), reinterpret_tensor(buf52, (64, 1), (1, 64), 0), reinterpret_tensor(buf45, (1, 64), (1, 1), 0), reinterpret_tensor(buf46, (64, 1), (1, 64), 0), reinterpret_tensor(buf39, (1, 64), (1, 1), 0), reinterpret_tensor(buf40, (64, 1), (1, 64), 0), reinterpret_tensor(buf33, (1, 64), (1, 1), 0), reinterpret_tensor(buf34, (64, 1), (1, 64), 0), reinterpret_tensor(buf27, (1, 64), (1, 1), 0), reinterpret_tensor(buf28, (64, 1), (1, 64), 0), reinterpret_tensor(buf21, (1, 64), (1, 1), 0), reinterpret_tensor(buf22, (64, 1), (1, 64), 0), reinterpret_tensor(buf15, (1, 64), (1, 1), 0), reinterpret_tensor(buf16, (64, 1), (1, 64), 0), reinterpret_tensor(buf210, (64, 65536), (1, 64), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((5, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_15 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_16 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.int32)
    primals_17 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.bool)
    primals_18 = rand_strided((16384, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_21 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_22 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_23 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_24 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_25 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_26 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_48 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_49 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_50 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_51 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_52 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_53 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_54 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.int32)
    primals_55 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((12, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
