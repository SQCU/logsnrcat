
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_cat_fill_mul_silu_split_sub_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 939524096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_cat_fill_mul_silu_split_sub_view_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1024*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (1024 + 2048*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr1 + (2048*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 2048, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr0 + (1024*x1 + ((-1024) + x0)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr1 + (2048*x1 + ((-1024) + x0)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp21 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp18, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp17, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
