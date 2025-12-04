
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
