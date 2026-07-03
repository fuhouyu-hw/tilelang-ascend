import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n


@tilelang.jit(out_idx=[-1])
def dynamic_fill(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.barrier_all()
                # Dynamic fill length: choose 32 or 64 based on a runtime value.
                idx = T.if_then_else(A[0, 0] > 0, 32, 64)
                T.tile.fill(c_ub[0, 0: idx], 1.0)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


func = dynamic_fill(M, N, 128, 256)

torch.manual_seed(0)

a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()

torch.npu.synchronize()
print("init successful!")

c = func(a, b).cpu()

# Verify the dynamic fill length: A[0,0]>0 -> first 32 cols of row 0 filled
# with 1.0, otherwise first 64 cols. T.tile.fill(c_ub[0, 0:idx], 1.0) only
# fills row 0 of the UB tile; the rest is left uninitialized.
fill_len = 32 if a[0, 0].item() > 0 else 64
filled = c[:1, :fill_len]
torch.testing.assert_close(filled, torch.ones_like(filled), rtol=0, atol=0)
print("Kernel Output Match!")
