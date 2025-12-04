
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'in_ptr40': '*fp32', 'in_ptr41': '*fp32', 'in_ptr42': '*fp32', 'in_ptr43': '*fp32', 'in_ptr44': '*fp32', 'in_ptr45': '*fp32', 'in_ptr46': '*fp32', 'in_ptr47': '*fp32', 'in_ptr48': '*fp32', 'in_ptr49': '*fp32', 'in_ptr50': '*fp32', 'in_ptr51': '*fp32', 'in_ptr52': '*fp32', 'in_ptr53': '*fp32', 'in_ptr54': '*fp32', 'in_ptr55': '*fp32', 'in_ptr56': '*fp32', 'in_ptr57': '*fp32', 'in_ptr58': '*fp32', 'in_ptr59': '*fp32', 'in_ptr60': '*fp32', 'in_ptr61': '*fp32', 'in_ptr62': '*fp32', 'in_ptr63': '*fp32', 'in_ptr64': '*fp32', 'in_ptr65': '*fp32', 'in_ptr66': '*fp32', 'in_ptr67': '*fp32', 'in_ptr68': '*fp32', 'in_ptr69': '*fp32', 'in_ptr70': '*fp32', 'in_ptr71': '*fp32', 'in_ptr72': '*fp32', 'in_ptr73': '*fp32', 'in_ptr74': '*fp32', 'in_ptr75': '*fp32', 'in_ptr76': '*fp32', 'in_ptr77': '*fp32', 'in_ptr78': '*fp32', 'in_ptr79': '*fp32', 'in_ptr80': '*fp32', 'in_ptr81': '*fp32', 'in_ptr82': '*fp32', 'in_ptr83': '*fp32', 'in_ptr84': '*fp32', 'in_ptr85': '*fp32', 'in_ptr86': '*fp32', 'in_ptr87': '*fp32', 'in_ptr88': '*fp32', 'in_ptr89': '*fp32', 'in_ptr90': '*fp32', 'in_ptr91': '*fp32', 'in_ptr92': '*fp32', 'in_ptr93': '*fp32', 'in_ptr94': '*fp32', 'in_ptr95': '*fp32', 'in_ptr96': '*fp32', 'in_ptr97': '*fp32', 'in_ptr98': '*fp32', 'in_ptr99': '*fp32', 'in_ptr100': '*fp32', 'in_ptr101': '*fp32', 'in_ptr102': '*fp32', 'in_ptr103': '*fp32', 'in_ptr104': '*fp32', 'in_ptr105': '*fp32', 'in_ptr106': '*fp32', 'in_ptr107': '*fp32', 'in_ptr108': '*fp32', 'in_ptr109': '*fp32', 'in_ptr110': '*fp32', 'in_ptr111': '*fp32', 'in_ptr112': '*fp32', 'in_ptr113': '*fp32', 'in_ptr114': '*fp32', 'in_ptr115': '*fp32', 'in_ptr116': '*fp32', 'in_ptr117': '*fp32', 'in_ptr118': '*fp32', 'in_ptr119': '*fp32', 'in_ptr120': '*fp32', 'in_ptr121': '*fp32', 'in_ptr122': '*fp32', 'in_ptr123': '*fp32', 'in_ptr124': '*fp32', 'in_ptr125': '*fp32', 'in_ptr126': '*fp32', 'in_ptr127': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (25,): [['tt.divisibility', 16]], (26,): [['tt.divisibility', 16]], (27,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (29,): [['tt.divisibility', 16]], (30,): [['tt.divisibility', 16]], (31,): [['tt.divisibility', 16]], (32,): [['tt.divisibility', 16]], (33,): [['tt.divisibility', 16]], (34,): [['tt.divisibility', 16]], (35,): [['tt.divisibility', 16]], (36,): [['tt.divisibility', 16]], (37,): [['tt.divisibility', 16]], (38,): [['tt.divisibility', 16]], (39,): [['tt.divisibility', 16]], (40,): [['tt.divisibility', 16]], (41,): [['tt.divisibility', 16]], (42,): [['tt.divisibility', 16]], (43,): [['tt.divisibility', 16]], (44,): [['tt.divisibility', 16]], (45,): [['tt.divisibility', 16]], (46,): [['tt.divisibility', 16]], (47,): [['tt.divisibility', 16]], (48,): [['tt.divisibility', 16]], (49,): [['tt.divisibility', 16]], (50,): [['tt.divisibility', 16]], (51,): [['tt.divisibility', 16]], (52,): [['tt.divisibility', 16]], (53,): [['tt.divisibility', 16]], (54,): [['tt.divisibility', 16]], (55,): [['tt.divisibility', 16]], (56,): [['tt.divisibility', 16]], (57,): [['tt.divisibility', 16]], (58,): [['tt.divisibility', 16]], (59,): [['tt.divisibility', 16]], (60,): [['tt.divisibility', 16]], (61,): [['tt.divisibility', 16]], (62,): [['tt.divisibility', 16]], (63,): [['tt.divisibility', 16]], (64,): [['tt.divisibility', 16]], (65,): [['tt.divisibility', 16]], (66,): [['tt.divisibility', 16]], (67,): [['tt.divisibility', 16]], (68,): [['tt.divisibility', 16]], (69,): [['tt.divisibility', 16]], (70,): [['tt.divisibility', 16]], (71,): [['tt.divisibility', 16]], (72,): [['tt.divisibility', 16]], (73,): [['tt.divisibility', 16]], (74,): [['tt.divisibility', 16]], (75,): [['tt.divisibility', 16]], (76,): [['tt.divisibility', 16]], (77,): [['tt.divisibility', 16]], (78,): [['tt.divisibility', 16]], (79,): [['tt.divisibility', 16]], (80,): [['tt.divisibility', 16]], (81,): [['tt.divisibility', 16]], (82,): [['tt.divisibility', 16]], (83,): [['tt.divisibility', 16]], (84,): [['tt.divisibility', 16]], (85,): [['tt.divisibility', 16]], (86,): [['tt.divisibility', 16]], (87,): [['tt.divisibility', 16]], (88,): [['tt.divisibility', 16]], (89,): [['tt.divisibility', 16]], (90,): [['tt.divisibility', 16]], (91,): [['tt.divisibility', 16]], (92,): [['tt.divisibility', 16]], (93,): [['tt.divisibility', 16]], (94,): [['tt.divisibility', 16]], (95,): [['tt.divisibility', 16]], (96,): [['tt.divisibility', 16]], (97,): [['tt.divisibility', 16]], (98,): [['tt.divisibility', 16]], (99,): [['tt.divisibility', 16]], (100,): [['tt.divisibility', 16]], (101,): [['tt.divisibility', 16]], (102,): [['tt.divisibility', 16]], (103,): [['tt.divisibility', 16]], (104,): [['tt.divisibility', 16]], (105,): [['tt.divisibility', 16]], (106,): [['tt.divisibility', 16]], (107,): [['tt.divisibility', 16]], (108,): [['tt.divisibility', 16]], (109,): [['tt.divisibility', 16]], (110,): [['tt.divisibility', 16]], (111,): [['tt.divisibility', 16]], (112,): [['tt.divisibility', 16]], (113,): [['tt.divisibility', 16]], (114,): [['tt.divisibility', 16]], (115,): [['tt.divisibility', 16]], (116,): [['tt.divisibility', 16]], (117,): [['tt.divisibility', 16]], (118,): [['tt.divisibility', 16]], (119,): [['tt.divisibility', 16]], (120,): [['tt.divisibility', 16]], (121,): [['tt.divisibility', 16]], (122,): [['tt.divisibility', 16]], (123,): [['tt.divisibility', 16]], (124,): [['tt.divisibility', 16]], (125,): [['tt.divisibility', 16]], (126,): [['tt.divisibility', 16]], (127,): [['tt.divisibility', 16]], (128,): [['tt.divisibility', 16]], (129,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_expand_mul_neg_pow_select_select_backward_squeeze_unsqueeze_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 128, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_expand_mul_neg_pow_select_select_backward_squeeze_unsqueeze_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, in_ptr100, in_ptr101, in_ptr102, in_ptr103, in_ptr104, in_ptr105, in_ptr106, in_ptr107, in_ptr108, in_ptr109, in_ptr110, in_ptr111, in_ptr112, in_ptr113, in_ptr114, in_ptr115, in_ptr116, in_ptr117, in_ptr118, in_ptr119, in_ptr120, in_ptr121, in_ptr122, in_ptr123, in_ptr124, in_ptr125, in_ptr126, in_ptr127, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr13 + (x0), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr14 + (x0), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr15 + (x0), xmask, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr16 + (x0), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr17 + (x0), xmask, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr18 + (x0), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr19 + (x0), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr20 + (x0), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr21 + (x0), xmask, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr22 + (x0), xmask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr23 + (x0), xmask, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr24 + (x0), xmask, eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr25 + (x0), xmask, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr26 + (x0), xmask, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr27 + (x0), xmask, eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr28 + (x0), xmask, eviction_policy='evict_last')
    tmp148 = tl.load(in_ptr29 + (x0), xmask, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr30 + (x0), xmask, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr31 + (x0), xmask, eviction_policy='evict_last')
    tmp161 = tl.load(in_ptr32 + (x0), xmask, eviction_policy='evict_last')
    tmp164 = tl.load(in_ptr33 + (x0), xmask, eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr34 + (x0), xmask, eviction_policy='evict_last')
    tmp170 = tl.load(in_ptr35 + (x0), xmask, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr36 + (x0), xmask, eviction_policy='evict_last')
    tmp176 = tl.load(in_ptr37 + (x0), xmask, eviction_policy='evict_last')
    tmp179 = tl.load(in_ptr38 + (x0), xmask, eviction_policy='evict_last')
    tmp182 = tl.load(in_ptr39 + (x0), xmask, eviction_policy='evict_last')
    tmp185 = tl.load(in_ptr40 + (x0), xmask, eviction_policy='evict_last')
    tmp188 = tl.load(in_ptr41 + (x0), xmask, eviction_policy='evict_last')
    tmp191 = tl.load(in_ptr42 + (x0), xmask, eviction_policy='evict_last')
    tmp194 = tl.load(in_ptr43 + (x0), xmask, eviction_policy='evict_last')
    tmp197 = tl.load(in_ptr44 + (x0), xmask, eviction_policy='evict_last')
    tmp200 = tl.load(in_ptr45 + (x0), xmask, eviction_policy='evict_last')
    tmp203 = tl.load(in_ptr46 + (x0), xmask, eviction_policy='evict_last')
    tmp206 = tl.load(in_ptr47 + (x0), xmask, eviction_policy='evict_last')
    tmp209 = tl.load(in_ptr48 + (x0), xmask, eviction_policy='evict_last')
    tmp212 = tl.load(in_ptr49 + (x0), xmask, eviction_policy='evict_last')
    tmp215 = tl.load(in_ptr50 + (x0), xmask, eviction_policy='evict_last')
    tmp218 = tl.load(in_ptr51 + (x0), xmask, eviction_policy='evict_last')
    tmp221 = tl.load(in_ptr52 + (x0), xmask, eviction_policy='evict_last')
    tmp224 = tl.load(in_ptr53 + (x0), xmask, eviction_policy='evict_last')
    tmp227 = tl.load(in_ptr54 + (x0), xmask, eviction_policy='evict_last')
    tmp230 = tl.load(in_ptr55 + (x0), xmask, eviction_policy='evict_last')
    tmp233 = tl.load(in_ptr56 + (x0), xmask, eviction_policy='evict_last')
    tmp236 = tl.load(in_ptr57 + (x0), xmask, eviction_policy='evict_last')
    tmp239 = tl.load(in_ptr58 + (x0), xmask, eviction_policy='evict_last')
    tmp242 = tl.load(in_ptr59 + (x0), xmask, eviction_policy='evict_last')
    tmp245 = tl.load(in_ptr60 + (x0), xmask, eviction_policy='evict_last')
    tmp248 = tl.load(in_ptr61 + (x0), xmask, eviction_policy='evict_last')
    tmp251 = tl.load(in_ptr62 + (x0), xmask, eviction_policy='evict_last')
    tmp254 = tl.load(in_ptr63 + (x0), xmask, eviction_policy='evict_last')
    tmp257 = tl.load(in_ptr64 + (x0), xmask, eviction_policy='evict_last')
    tmp260 = tl.load(in_ptr65 + (x0), xmask, eviction_policy='evict_last')
    tmp263 = tl.load(in_ptr66 + (x0), xmask, eviction_policy='evict_last')
    tmp266 = tl.load(in_ptr67 + (x0), xmask, eviction_policy='evict_last')
    tmp269 = tl.load(in_ptr68 + (x0), xmask, eviction_policy='evict_last')
    tmp272 = tl.load(in_ptr69 + (x0), xmask, eviction_policy='evict_last')
    tmp275 = tl.load(in_ptr70 + (x0), xmask, eviction_policy='evict_last')
    tmp278 = tl.load(in_ptr71 + (x0), xmask, eviction_policy='evict_last')
    tmp281 = tl.load(in_ptr72 + (x0), xmask, eviction_policy='evict_last')
    tmp284 = tl.load(in_ptr73 + (x0), xmask, eviction_policy='evict_last')
    tmp287 = tl.load(in_ptr74 + (x0), xmask, eviction_policy='evict_last')
    tmp290 = tl.load(in_ptr75 + (x0), xmask, eviction_policy='evict_last')
    tmp293 = tl.load(in_ptr76 + (x0), xmask, eviction_policy='evict_last')
    tmp296 = tl.load(in_ptr77 + (x0), xmask, eviction_policy='evict_last')
    tmp299 = tl.load(in_ptr78 + (x0), xmask, eviction_policy='evict_last')
    tmp302 = tl.load(in_ptr79 + (x0), xmask, eviction_policy='evict_last')
    tmp305 = tl.load(in_ptr80 + (x0), xmask, eviction_policy='evict_last')
    tmp308 = tl.load(in_ptr81 + (x0), xmask, eviction_policy='evict_last')
    tmp311 = tl.load(in_ptr82 + (x0), xmask, eviction_policy='evict_last')
    tmp314 = tl.load(in_ptr83 + (x0), xmask, eviction_policy='evict_last')
    tmp317 = tl.load(in_ptr84 + (x0), xmask, eviction_policy='evict_last')
    tmp320 = tl.load(in_ptr85 + (x0), xmask, eviction_policy='evict_last')
    tmp323 = tl.load(in_ptr86 + (x0), xmask, eviction_policy='evict_last')
    tmp326 = tl.load(in_ptr87 + (x0), xmask, eviction_policy='evict_last')
    tmp329 = tl.load(in_ptr88 + (x0), xmask, eviction_policy='evict_last')
    tmp332 = tl.load(in_ptr89 + (x0), xmask, eviction_policy='evict_last')
    tmp335 = tl.load(in_ptr90 + (x0), xmask, eviction_policy='evict_last')
    tmp338 = tl.load(in_ptr91 + (x0), xmask, eviction_policy='evict_last')
    tmp341 = tl.load(in_ptr92 + (x0), xmask, eviction_policy='evict_last')
    tmp344 = tl.load(in_ptr93 + (x0), xmask, eviction_policy='evict_last')
    tmp347 = tl.load(in_ptr94 + (x0), xmask, eviction_policy='evict_last')
    tmp350 = tl.load(in_ptr95 + (x0), xmask, eviction_policy='evict_last')
    tmp353 = tl.load(in_ptr96 + (x0), xmask, eviction_policy='evict_last')
    tmp356 = tl.load(in_ptr97 + (x0), xmask, eviction_policy='evict_last')
    tmp359 = tl.load(in_ptr98 + (x0), xmask, eviction_policy='evict_last')
    tmp362 = tl.load(in_ptr99 + (x0), xmask, eviction_policy='evict_last')
    tmp365 = tl.load(in_ptr100 + (x0), xmask, eviction_policy='evict_last')
    tmp368 = tl.load(in_ptr101 + (x0), xmask, eviction_policy='evict_last')
    tmp371 = tl.load(in_ptr102 + (x0), xmask, eviction_policy='evict_last')
    tmp374 = tl.load(in_ptr103 + (x0), xmask, eviction_policy='evict_last')
    tmp377 = tl.load(in_ptr104 + (x0), xmask, eviction_policy='evict_last')
    tmp380 = tl.load(in_ptr105 + (x0), xmask, eviction_policy='evict_last')
    tmp383 = tl.load(in_ptr106 + (x0), xmask, eviction_policy='evict_last')
    tmp386 = tl.load(in_ptr107 + (x0), xmask, eviction_policy='evict_last')
    tmp389 = tl.load(in_ptr108 + (x0), xmask, eviction_policy='evict_last')
    tmp392 = tl.load(in_ptr109 + (x0), xmask, eviction_policy='evict_last')
    tmp395 = tl.load(in_ptr110 + (x0), xmask, eviction_policy='evict_last')
    tmp398 = tl.load(in_ptr111 + (x0), xmask, eviction_policy='evict_last')
    tmp401 = tl.load(in_ptr112 + (x0), xmask, eviction_policy='evict_last')
    tmp404 = tl.load(in_ptr113 + (x0), xmask, eviction_policy='evict_last')
    tmp407 = tl.load(in_ptr114 + (x0), xmask, eviction_policy='evict_last')
    tmp410 = tl.load(in_ptr115 + (x0), xmask, eviction_policy='evict_last')
    tmp413 = tl.load(in_ptr116 + (x0), xmask, eviction_policy='evict_last')
    tmp416 = tl.load(in_ptr117 + (x0), xmask, eviction_policy='evict_last')
    tmp419 = tl.load(in_ptr118 + (x0), xmask, eviction_policy='evict_last')
    tmp422 = tl.load(in_ptr119 + (x0), xmask, eviction_policy='evict_last')
    tmp425 = tl.load(in_ptr120 + (x0), xmask, eviction_policy='evict_last')
    tmp428 = tl.load(in_ptr121 + (x0), xmask, eviction_policy='evict_last')
    tmp431 = tl.load(in_ptr122 + (x0), xmask, eviction_policy='evict_last')
    tmp434 = tl.load(in_ptr123 + (x0), xmask, eviction_policy='evict_last')
    tmp437 = tl.load(in_ptr124 + (x0), xmask, eviction_policy='evict_last')
    tmp440 = tl.load(in_ptr125 + (x0), xmask, eviction_policy='evict_last')
    tmp443 = tl.load(in_ptr126 + (x0), xmask, eviction_policy='evict_last')
    tmp446 = tl.load(in_ptr127 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 31, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 30, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 29, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tmp16 = tl.full([1], 28, tl.int32)
    tmp17 = tmp0 == tmp16
    tmp19 = tl.where(tmp17, tmp18, tmp4)
    tmp20 = tmp15 + tmp19
    tmp21 = tl.full([1], 27, tl.int32)
    tmp22 = tmp0 == tmp21
    tmp24 = tl.where(tmp22, tmp23, tmp4)
    tmp25 = tmp20 + tmp24
    tmp26 = tl.full([1], 26, tl.int32)
    tmp27 = tmp0 == tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp4)
    tmp30 = tmp25 + tmp29
    tmp31 = tl.full([1], 25, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp34 = tl.where(tmp32, tmp33, tmp4)
    tmp35 = tmp30 + tmp34
    tmp36 = tl.full([1], 24, tl.int32)
    tmp37 = tmp0 == tmp36
    tmp39 = tl.where(tmp37, tmp38, tmp4)
    tmp40 = tmp35 + tmp39
    tmp41 = tl.full([1], 23, tl.int32)
    tmp42 = tmp0 == tmp41
    tmp44 = tl.where(tmp42, tmp43, tmp4)
    tmp45 = tmp40 + tmp44
    tmp46 = tl.full([1], 22, tl.int32)
    tmp47 = tmp0 == tmp46
    tmp49 = tl.where(tmp47, tmp48, tmp4)
    tmp50 = tmp45 + tmp49
    tmp51 = tl.full([1], 21, tl.int32)
    tmp52 = tmp0 == tmp51
    tmp54 = tl.where(tmp52, tmp53, tmp4)
    tmp55 = tmp50 + tmp54
    tmp56 = tl.full([1], 20, tl.int32)
    tmp57 = tmp0 == tmp56
    tmp59 = tl.where(tmp57, tmp58, tmp4)
    tmp60 = tmp55 + tmp59
    tmp61 = tl.full([1], 19, tl.int32)
    tmp62 = tmp0 == tmp61
    tmp64 = tl.where(tmp62, tmp63, tmp4)
    tmp65 = tmp60 + tmp64
    tmp66 = tl.full([1], 18, tl.int32)
    tmp67 = tmp0 == tmp66
    tmp69 = tl.where(tmp67, tmp68, tmp4)
    tmp70 = tmp65 + tmp69
    tmp71 = tl.full([1], 17, tl.int32)
    tmp72 = tmp0 == tmp71
    tmp74 = tl.where(tmp72, tmp73, tmp4)
    tmp75 = tmp70 + tmp74
    tmp76 = tl.full([1], 16, tl.int32)
    tmp77 = tmp0 == tmp76
    tmp79 = tl.where(tmp77, tmp78, tmp4)
    tmp80 = tmp75 + tmp79
    tmp81 = tl.full([1], 15, tl.int32)
    tmp82 = tmp0 == tmp81
    tmp84 = tl.where(tmp82, tmp83, tmp4)
    tmp85 = tmp80 + tmp84
    tmp86 = tl.full([1], 14, tl.int32)
    tmp87 = tmp0 == tmp86
    tmp89 = tl.where(tmp87, tmp88, tmp4)
    tmp90 = tmp85 + tmp89
    tmp91 = tl.full([1], 13, tl.int32)
    tmp92 = tmp0 == tmp91
    tmp94 = tl.where(tmp92, tmp93, tmp4)
    tmp95 = tmp90 + tmp94
    tmp96 = tl.full([1], 12, tl.int32)
    tmp97 = tmp0 == tmp96
    tmp99 = tl.where(tmp97, tmp98, tmp4)
    tmp100 = tmp95 + tmp99
    tmp101 = tl.full([1], 11, tl.int32)
    tmp102 = tmp0 == tmp101
    tmp104 = tl.where(tmp102, tmp103, tmp4)
    tmp105 = tmp100 + tmp104
    tmp106 = tl.full([1], 10, tl.int32)
    tmp107 = tmp0 == tmp106
    tmp109 = tl.where(tmp107, tmp108, tmp4)
    tmp110 = tmp105 + tmp109
    tmp111 = tl.full([1], 9, tl.int32)
    tmp112 = tmp0 == tmp111
    tmp114 = tl.where(tmp112, tmp113, tmp4)
    tmp115 = tmp110 + tmp114
    tmp116 = tl.full([1], 8, tl.int32)
    tmp117 = tmp0 == tmp116
    tmp119 = tl.where(tmp117, tmp118, tmp4)
    tmp120 = tmp115 + tmp119
    tmp121 = tl.full([1], 7, tl.int32)
    tmp122 = tmp0 == tmp121
    tmp124 = tl.where(tmp122, tmp123, tmp4)
    tmp125 = tmp120 + tmp124
    tmp126 = tl.full([1], 6, tl.int32)
    tmp127 = tmp0 == tmp126
    tmp129 = tl.where(tmp127, tmp128, tmp4)
    tmp130 = tmp125 + tmp129
    tmp131 = tl.full([1], 5, tl.int32)
    tmp132 = tmp0 == tmp131
    tmp134 = tl.where(tmp132, tmp133, tmp4)
    tmp135 = tmp130 + tmp134
    tmp136 = tl.full([1], 4, tl.int32)
    tmp137 = tmp0 == tmp136
    tmp139 = tl.where(tmp137, tmp138, tmp4)
    tmp140 = tmp135 + tmp139
    tmp141 = tl.full([1], 3, tl.int32)
    tmp142 = tmp0 == tmp141
    tmp144 = tl.where(tmp142, tmp143, tmp4)
    tmp145 = tmp140 + tmp144
    tmp146 = tl.full([1], 2, tl.int32)
    tmp147 = tmp0 == tmp146
    tmp149 = tl.where(tmp147, tmp148, tmp4)
    tmp150 = tmp145 + tmp149
    tmp151 = tl.full([1], 1, tl.int32)
    tmp152 = tmp0 == tmp151
    tmp154 = tl.where(tmp152, tmp153, tmp4)
    tmp155 = tmp150 + tmp154
    tmp156 = tl.full([1], 0, tl.int32)
    tmp157 = tmp0 == tmp156
    tmp159 = tl.where(tmp157, tmp158, tmp4)
    tmp160 = tmp155 + tmp159
    tmp162 = tl.where(tmp2, tmp161, tmp4)
    tmp163 = tmp160 + tmp162
    tmp165 = tl.where(tmp7, tmp164, tmp4)
    tmp166 = tmp163 + tmp165
    tmp168 = tl.where(tmp12, tmp167, tmp4)
    tmp169 = tmp166 + tmp168
    tmp171 = tl.where(tmp17, tmp170, tmp4)
    tmp172 = tmp169 + tmp171
    tmp174 = tl.where(tmp22, tmp173, tmp4)
    tmp175 = tmp172 + tmp174
    tmp177 = tl.where(tmp27, tmp176, tmp4)
    tmp178 = tmp175 + tmp177
    tmp180 = tl.where(tmp32, tmp179, tmp4)
    tmp181 = tmp178 + tmp180
    tmp183 = tl.where(tmp37, tmp182, tmp4)
    tmp184 = tmp181 + tmp183
    tmp186 = tl.where(tmp42, tmp185, tmp4)
    tmp187 = tmp184 + tmp186
    tmp189 = tl.where(tmp47, tmp188, tmp4)
    tmp190 = tmp187 + tmp189
    tmp192 = tl.where(tmp52, tmp191, tmp4)
    tmp193 = tmp190 + tmp192
    tmp195 = tl.where(tmp57, tmp194, tmp4)
    tmp196 = tmp193 + tmp195
    tmp198 = tl.where(tmp62, tmp197, tmp4)
    tmp199 = tmp196 + tmp198
    tmp201 = tl.where(tmp67, tmp200, tmp4)
    tmp202 = tmp199 + tmp201
    tmp204 = tl.where(tmp72, tmp203, tmp4)
    tmp205 = tmp202 + tmp204
    tmp207 = tl.where(tmp77, tmp206, tmp4)
    tmp208 = tmp205 + tmp207
    tmp210 = tl.where(tmp82, tmp209, tmp4)
    tmp211 = tmp208 + tmp210
    tmp213 = tl.where(tmp87, tmp212, tmp4)
    tmp214 = tmp211 + tmp213
    tmp216 = tl.where(tmp92, tmp215, tmp4)
    tmp217 = tmp214 + tmp216
    tmp219 = tl.where(tmp97, tmp218, tmp4)
    tmp220 = tmp217 + tmp219
    tmp222 = tl.where(tmp102, tmp221, tmp4)
    tmp223 = tmp220 + tmp222
    tmp225 = tl.where(tmp107, tmp224, tmp4)
    tmp226 = tmp223 + tmp225
    tmp228 = tl.where(tmp112, tmp227, tmp4)
    tmp229 = tmp226 + tmp228
    tmp231 = tl.where(tmp117, tmp230, tmp4)
    tmp232 = tmp229 + tmp231
    tmp234 = tl.where(tmp122, tmp233, tmp4)
    tmp235 = tmp232 + tmp234
    tmp237 = tl.where(tmp127, tmp236, tmp4)
    tmp238 = tmp235 + tmp237
    tmp240 = tl.where(tmp132, tmp239, tmp4)
    tmp241 = tmp238 + tmp240
    tmp243 = tl.where(tmp137, tmp242, tmp4)
    tmp244 = tmp241 + tmp243
    tmp246 = tl.where(tmp142, tmp245, tmp4)
    tmp247 = tmp244 + tmp246
    tmp249 = tl.where(tmp147, tmp248, tmp4)
    tmp250 = tmp247 + tmp249
    tmp252 = tl.where(tmp152, tmp251, tmp4)
    tmp253 = tmp250 + tmp252
    tmp255 = tl.where(tmp157, tmp254, tmp4)
    tmp256 = tmp253 + tmp255
    tmp258 = tl.where(tmp2, tmp257, tmp4)
    tmp259 = tmp256 + tmp258
    tmp261 = tl.where(tmp7, tmp260, tmp4)
    tmp262 = tmp259 + tmp261
    tmp264 = tl.where(tmp12, tmp263, tmp4)
    tmp265 = tmp262 + tmp264
    tmp267 = tl.where(tmp17, tmp266, tmp4)
    tmp268 = tmp265 + tmp267
    tmp270 = tl.where(tmp22, tmp269, tmp4)
    tmp271 = tmp268 + tmp270
    tmp273 = tl.where(tmp27, tmp272, tmp4)
    tmp274 = tmp271 + tmp273
    tmp276 = tl.where(tmp32, tmp275, tmp4)
    tmp277 = tmp274 + tmp276
    tmp279 = tl.where(tmp37, tmp278, tmp4)
    tmp280 = tmp277 + tmp279
    tmp282 = tl.where(tmp42, tmp281, tmp4)
    tmp283 = tmp280 + tmp282
    tmp285 = tl.where(tmp47, tmp284, tmp4)
    tmp286 = tmp283 + tmp285
    tmp288 = tl.where(tmp52, tmp287, tmp4)
    tmp289 = tmp286 + tmp288
    tmp291 = tl.where(tmp57, tmp290, tmp4)
    tmp292 = tmp289 + tmp291
    tmp294 = tl.where(tmp62, tmp293, tmp4)
    tmp295 = tmp292 + tmp294
    tmp297 = tl.where(tmp67, tmp296, tmp4)
    tmp298 = tmp295 + tmp297
    tmp300 = tl.where(tmp72, tmp299, tmp4)
    tmp301 = tmp298 + tmp300
    tmp303 = tl.where(tmp77, tmp302, tmp4)
    tmp304 = tmp301 + tmp303
    tmp306 = tl.where(tmp82, tmp305, tmp4)
    tmp307 = tmp304 + tmp306
    tmp309 = tl.where(tmp87, tmp308, tmp4)
    tmp310 = tmp307 + tmp309
    tmp312 = tl.where(tmp92, tmp311, tmp4)
    tmp313 = tmp310 + tmp312
    tmp315 = tl.where(tmp97, tmp314, tmp4)
    tmp316 = tmp313 + tmp315
    tmp318 = tl.where(tmp102, tmp317, tmp4)
    tmp319 = tmp316 + tmp318
    tmp321 = tl.where(tmp107, tmp320, tmp4)
    tmp322 = tmp319 + tmp321
    tmp324 = tl.where(tmp112, tmp323, tmp4)
    tmp325 = tmp322 + tmp324
    tmp327 = tl.where(tmp117, tmp326, tmp4)
    tmp328 = tmp325 + tmp327
    tmp330 = tl.where(tmp122, tmp329, tmp4)
    tmp331 = tmp328 + tmp330
    tmp333 = tl.where(tmp127, tmp332, tmp4)
    tmp334 = tmp331 + tmp333
    tmp336 = tl.where(tmp132, tmp335, tmp4)
    tmp337 = tmp334 + tmp336
    tmp339 = tl.where(tmp137, tmp338, tmp4)
    tmp340 = tmp337 + tmp339
    tmp342 = tl.where(tmp142, tmp341, tmp4)
    tmp343 = tmp340 + tmp342
    tmp345 = tl.where(tmp147, tmp344, tmp4)
    tmp346 = tmp343 + tmp345
    tmp348 = tl.where(tmp152, tmp347, tmp4)
    tmp349 = tmp346 + tmp348
    tmp351 = tl.where(tmp157, tmp350, tmp4)
    tmp352 = tmp349 + tmp351
    tmp354 = tl.where(tmp2, tmp353, tmp4)
    tmp355 = tmp352 + tmp354
    tmp357 = tl.where(tmp7, tmp356, tmp4)
    tmp358 = tmp355 + tmp357
    tmp360 = tl.where(tmp12, tmp359, tmp4)
    tmp361 = tmp358 + tmp360
    tmp363 = tl.where(tmp17, tmp362, tmp4)
    tmp364 = tmp361 + tmp363
    tmp366 = tl.where(tmp22, tmp365, tmp4)
    tmp367 = tmp364 + tmp366
    tmp369 = tl.where(tmp27, tmp368, tmp4)
    tmp370 = tmp367 + tmp369
    tmp372 = tl.where(tmp32, tmp371, tmp4)
    tmp373 = tmp370 + tmp372
    tmp375 = tl.where(tmp37, tmp374, tmp4)
    tmp376 = tmp373 + tmp375
    tmp378 = tl.where(tmp42, tmp377, tmp4)
    tmp379 = tmp376 + tmp378
    tmp381 = tl.where(tmp47, tmp380, tmp4)
    tmp382 = tmp379 + tmp381
    tmp384 = tl.where(tmp52, tmp383, tmp4)
    tmp385 = tmp382 + tmp384
    tmp387 = tl.where(tmp57, tmp386, tmp4)
    tmp388 = tmp385 + tmp387
    tmp390 = tl.where(tmp62, tmp389, tmp4)
    tmp391 = tmp388 + tmp390
    tmp393 = tl.where(tmp67, tmp392, tmp4)
    tmp394 = tmp391 + tmp393
    tmp396 = tl.where(tmp72, tmp395, tmp4)
    tmp397 = tmp394 + tmp396
    tmp399 = tl.where(tmp77, tmp398, tmp4)
    tmp400 = tmp397 + tmp399
    tmp402 = tl.where(tmp82, tmp401, tmp4)
    tmp403 = tmp400 + tmp402
    tmp405 = tl.where(tmp87, tmp404, tmp4)
    tmp406 = tmp403 + tmp405
    tmp408 = tl.where(tmp92, tmp407, tmp4)
    tmp409 = tmp406 + tmp408
    tmp411 = tl.where(tmp97, tmp410, tmp4)
    tmp412 = tmp409 + tmp411
    tmp414 = tl.where(tmp102, tmp413, tmp4)
    tmp415 = tmp412 + tmp414
    tmp417 = tl.where(tmp107, tmp416, tmp4)
    tmp418 = tmp415 + tmp417
    tmp420 = tl.where(tmp112, tmp419, tmp4)
    tmp421 = tmp418 + tmp420
    tmp423 = tl.where(tmp117, tmp422, tmp4)
    tmp424 = tmp421 + tmp423
    tmp426 = tl.where(tmp122, tmp425, tmp4)
    tmp427 = tmp424 + tmp426
    tmp429 = tl.where(tmp127, tmp428, tmp4)
    tmp430 = tmp427 + tmp429
    tmp432 = tl.where(tmp132, tmp431, tmp4)
    tmp433 = tmp430 + tmp432
    tmp435 = tl.where(tmp137, tmp434, tmp4)
    tmp436 = tmp433 + tmp435
    tmp438 = tl.where(tmp142, tmp437, tmp4)
    tmp439 = tmp436 + tmp438
    tmp441 = tl.where(tmp147, tmp440, tmp4)
    tmp442 = tmp439 + tmp441
    tmp444 = tl.where(tmp152, tmp443, tmp4)
    tmp445 = tmp442 + tmp444
    tmp447 = tl.where(tmp157, tmp446, tmp4)
    tmp448 = tmp445 + tmp447
    tl.store(in_out_ptr0 + (x2), tmp448, xmask)
