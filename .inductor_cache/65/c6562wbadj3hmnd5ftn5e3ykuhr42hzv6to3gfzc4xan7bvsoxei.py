# AOT ID: ['2_inference']
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


# kernel path: ./.inductor_cache\ra\cra3mqk2f7q6kbpxeaaq2s3pqziulbz536z37de7fslesjuylv6p.py
# Topologically Sorted Source Nodes: [rope_pos, setitem, arange, setitem_1, active_coords], Original ATen: [aten.zeros, aten.select, aten.arange, aten.copy, aten.slice]
# Source node to ATen node mapping:
#   active_coords => slice_1
#   arange => iota
#   rope_pos => full_default
#   setitem => copy, select
#   setitem_1 => copy_1, slice_3
# Graph fragment:
#   %arg1_1 : Tensor "f32[16384, 2][2, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %full_default : Tensor "f32[s2, 3][3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg0_1, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select : Tensor "f32[s2][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%full_default, 1, 0), kwargs = {})
#   %iota : Tensor "i64[s2][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %copy : Tensor "f32[s2][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %iota), kwargs = {})
#   %select_scatter_default : Tensor "f32[s2, 3][3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %copy, 1, 0), kwargs = {})
#   %slice_3 : Tensor "f32[s2, 2][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_scatter_default, 1, 1, 3), kwargs = {})
#   %slice_1 : Tensor "f32[s2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 0, 0, %arg0_1), kwargs = {})
#   %copy_1 : Tensor "f32[s2, 2][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_3, %slice_1), kwargs = {})
#   %slice_scatter_default : Tensor "f32[s2, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_scatter_default, %copy_1, 1, 1, 3), kwargs = {})
#   return %slice_scatter_default
triton_poi_fused_arange_copy_select_slice_zeros_0 = async_compile.triton('triton_poi_fused_arange_copy_select_slice_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_copy_select_slice_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '19838AED018D8011B66C11B0225D309931656BCD5997815B2E573DBF03530A55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_copy_select_slice_zeros_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x1 = xindex // 3
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-1) + x0 + 2*x1), tmp2 & xmask, other=0.0)
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = x1
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.where(tmp2, tmp3, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


cpp_fused_scalar_tensor_1 = async_compile.cpp_pybinding(['double*', 'const int64_t'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C" __declspec(dllexport) void  kernel(double* out_ptr0,
                       const int64_t ks0)
{
    {
        {
            {
                auto tmp0 = ((static_cast<double>(ks0)) / 64.000000000000000);
                auto tmp1 = c10::convert<double>(tmp0);
                out_ptr0[static_cast<int64_t>(0LL)] = tmp1;
            }
        }
    }
}
''')


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
        arg0_1, arg1_1 = args
        args.clear()
        s2 = arg0_1
        assert_size_stride(arg1_1, (16384, 2), (2, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s2, 3), (3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [rope_pos, setitem, arange, setitem_1, active_coords], Original ATen: [aten.zeros, aten.select, aten.arange, aten.copy, aten.slice]
            triton_poi_fused_arange_copy_select_slice_zeros_0_xnumel = 3*s2
            stream0 = get_raw_stream(0)
            triton_poi_fused_arange_copy_select_slice_zeros_0.run(arg1_1, buf0, triton_poi_fused_arange_copy_select_slice_zeros_0_xnumel, stream=stream0)
            del arg1_1
        buf1 = empty_strided_cpu((), (), torch.float64)
        cpp_fused_scalar_tensor_1(buf1, s2)
        return (buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 64
    arg1_1 = rand_strided((16384, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
