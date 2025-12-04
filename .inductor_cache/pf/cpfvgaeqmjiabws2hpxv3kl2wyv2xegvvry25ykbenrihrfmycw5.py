
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
