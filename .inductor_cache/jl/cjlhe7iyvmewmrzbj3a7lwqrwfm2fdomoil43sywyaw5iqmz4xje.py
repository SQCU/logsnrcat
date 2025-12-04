
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_cos_mul_neg_sin_slice_unsqueeze_view_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83902464}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_cos_mul_neg_sin_slice_unsqueeze_view_50(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp10 = tl.load(in_ptr1 + (32*x1 + (((x0) % 32))), tmp8, eviction_policy='evict_last', other=0.0)
    tmp11 = tl_math.sin(tmp10)
    tmp12 = tmp9 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp8, tmp12, tmp13)
    tmp15 = tmp4 >= tmp7
    tmp16 = tl.full([1], 64, tl.int64)
    tmp17 = tmp4 < tmp16
    tmp18 = tl.load(in_ptr0 + (64*x4 + ((-32) + x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr1 + (32*x1 + ((((-32) + x0) % 32))), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tl_math.sin(tmp19)
    tmp21 = tmp18 * tmp20
    tmp22 = -tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp8, tmp14, tmp24)
    tmp26 = tmp3 + tmp25
    tl.store(out_ptr0 + (x3), tmp26, None)
