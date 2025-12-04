
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
