    
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