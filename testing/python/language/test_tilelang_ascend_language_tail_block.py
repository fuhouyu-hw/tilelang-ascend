import pytest
import tilelang
import tilelang.language as T
import torch

"""
Tail-block (尾块) guard suite.

Feature under test
------------------
"尾块处理": the framework automatically handles the partial last tile when a
tensor dimension is NOT a multiple of the tile/block size. The frontend simply
allocates full-size ``block_M x block_N`` tiles, drives the grid/loops with
``T.ceildiv``, and indexes with ``bx * block_M`` -- it never special-cases the
edge. CUBE / VECTOR / CV-fusion operators are all covered, and the frontend is
"无需感知" (does not need to be aware of) the tail.

Mechanism (src/op/ascend.cc :: compute_valid_extent)
----------------------------------------------------
Every GM<->on-chip ``T.copy`` is clamped at lowering time::

    valid = Select(shape - off >= extent, extent,          # full block
                   Select(shape - off > 0,  shape - off,   # tail block
                          0))                               # fully OOB

where ``shape`` is the GM tensor's real dim and ``off`` is the tile offset
(e.g. ``bx * block_M``). The clamp is emitted for these copy directions:

    CUBE   : gm2l1 (load A/B)   + l0c2gm (store C)   -> M / N / K tails
    VECTOR : gm2ub (load)       + ub2gm  (store)     -> M / N tails
    CV     : C-scope uses the cube path, V-scope the vector path

pad_value (the subtle VECTOR case)
----------------------------------
On ``gm2ub`` loads the UB area outside ``validRow x validCol`` is filled with
``pad_value`` (``T.copy(..., pad_value=...)``; default 0 -- ascend.cc:58 /
copy.py:277). Correctness impact:

    * element-wise (add/abs/...) : pad is computed but NOT stored back
      (ub2gm re-clamps the store) -> pad_value is irrelevant, default 0 is fine.
    * reduce sum                 : pad must be 0   (default already correct).
    * reduce max                 : pad must be -inf (default 0 is WRONG on
      all-negative data).
    * reduce min                 : pad must be +inf (default 0 is WRONG on
      all-positive data).
    * CUBE gemm K-tail           : the L1 tail is implicitly 0, and 0 * B = 0,
      so the matmul stays correct with the default.

``reduce`` additionally accepts ``real_shape=[M, N]`` (reduce_ascend.py) as an
alternative to pad_value: it tells the reduce the true valid extent so it never
touches the pad region. This suite guards the ``pad_value`` path; the
``test_reduce_max_tail`` case fails if pad_value plumbing regresses.

NOTE: these cases execute on real NPU hardware (``.npu()``); they cannot run in a
CPU-only environment. Risk levels are annotated per group so unsupported
(target, dtype) combos can be dropped after an NPU run, per the established
workflow (cf. #683 / #700 tail-block iterations).
"""

# CUBE: mirrors examples/gemm/example_gemm_tail_block_developer.py
CUBE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# VECTOR: mirrors the elementwise suite's config (CV combine is harmless for a
# pure-vector kernel and matches the existing passing tests).
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {"float": torch.float32, "float16": torch.float16}[dtype]


# =============================================================================
# Group 1 - CUBE (gemm) tail block      [risk: low]
# M / N / K all non-divisible. Guards gm2l1 (load A/B) + l0c2gm (store C) clamp.
# Structure copied verbatim from example_gemm_tail_block_developer.py.
# =============================================================================
def cube_matmul_tail(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)  # gm2l1: M & K tail
                    T.copy(B[k * K_L1, by * block_N], B_L1)  # gm2l1: K & N tail
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])  # l0c2gm: M & N tail

    return main


def run_test_cube_matmul_tail(M, N, K, block_M, block_N, K_L1, target):
    torch.manual_seed(0)
    func = cube_matmul_tail(M, N, K, block_M, block_N, K_L1)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_PASS_CONFIGS, target=target)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    torch.npu.synchronize()
    c = func(a, b)

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# (M, N, K, block_M, block_N, K_L1) - every dim deliberately non-divisible.
cube_tail_configs = [
    (32 * 3 + 30, 32 * 2 + 16, 32 * 4 + 31, 32, 32, 32),  # (126, 80, 159)
    (64 * 8 + 45, 64 * 8, 64 * 8 + 27, 64, 64, 64),       # (557, 512, 539) - N exact
    (128 * 4, 128 * 4 + 99, 128 * 4, 128, 128, 128),      # (512, 611, 512) - only N tail
    (1024 + 118, 1024 + 206, 1024 + 55, 128, 256, 64),    # (1142, 1230, 1079)
]


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,K,block_M,block_N,K_L1", cube_tail_configs)
def test_cube_matmul_tail(M, N, K, block_M, block_N, K_L1, target):
    run_test_cube_matmul_tail(M, N, K, block_M, block_N, K_L1, target=target)


# =============================================================================
# Group 2a - VECTOR element-wise tail   [risk: low]
# M / N non-divisible. Guards gm2ub (load) + ub2gm (store) clamp. Full-block
# layout (no vid split) to isolate the tail mechanism. pad_value irrelevant here
# (the padded UB region is never stored back).
# =============================================================================
def vec_add_tail(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)  # gm2ub: M & N tail
            T.copy(B[bx * block_M, by * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M, by * block_N])  # ub2gm: M & N tail

    return main


def run_test_vec_add_tail(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = vec_add_tail(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)

    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu()
    b = torch.randn(M, N, dtype=td).npu()

    torch.npu.synchronize()
    c = func(a, b)

    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# =============================================================================
# Group 2b - VECTOR single-input tail   [risk: low]
# =============================================================================
def vec_abs_tail(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.abs(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_abs_tail(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = vec_abs_tail(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)

    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu()

    torch.npu.synchronize()
    b = func(a)

    ref_b = torch.abs(a)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


# (M, N, block_M, block_N) - both dims non-divisible.
vec_tail_configs = [
    (32 * 2 + 13, 32 * 3 + 7, 32, 32),       # (77, 103)
    (128 * 3 + 30, 128 * 2 + 50, 128, 128),  # (414, 306)
    (256 + 5, 512 + 11, 128, 256),           # (261, 523)
]


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,block_M,block_N", vec_tail_configs)
def test_vec_add_tail(M, N, block_M, block_N, dtype, target):
    run_test_vec_add_tail(M, N, block_M, block_N, dtype, target=target)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,block_M,block_N", vec_tail_configs)
def test_vec_abs_tail(M, N, block_M, block_N, dtype, target):
    run_test_vec_abs_tail(M, N, block_M, block_N, dtype, target=target)


# =============================================================================
# Group 2c - VECTOR reduce_max tail + pad_value   [risk: medium]
# THE pad_value guard. block_N >= N, so the N tail lands in the padded region of
# a single UB tile. Data is all-negative, so a wrong pad (default 0) would win
# the max and break the assertion -- this is what catches a pad_value regression.
# Mirrors examples/softmax/example_online_softmax.py (pad_value=-T.infinity +
# reduce_max).
# =============================================================================
def reduce_max_tail(M, N, block_M, block_N, dtype="float16"):
    m_num = T.ceildiv(M, block_M)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, 1), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, _):
            bx = cid

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, 1), dtype)

            # Over-extends to block_N (> N): the [N, block_N) columns are filled
            # with -inf so they cannot win the row max.
            T.copy(A[bx * block_M, 0], a_ub, pad_value=-T.infinity(dtype))
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[bx * block_M, 0])  # ub2gm: M tail clamp

    return main


def run_test_reduce_max_tail(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = reduce_max_tail(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)

    td = _torch_dtype(dtype)
    # All-negative input: a correct -inf pad keeps the row max negative, a wrong
    # 0 pad would report 0.
    a = (-torch.rand(M, N, dtype=td) - 0.5).npu()

    torch.npu.synchronize()
    b = func(a)

    ref_b = a.max(dim=1, keepdim=True).values
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


# (M, N, block_M, block_N) with block_N >= N so the N tail becomes pad columns.
reduce_tail_configs = [
    (32 * 3 + 30, 200, 32, 256),  # (126, 200): M tail + 56 pad cols
    (64 * 4 + 7, 100, 64, 128),   # (263, 100): M tail + 28 pad cols
]


@pytest.mark.parametrize("dtype", ["float16", "float"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,block_M,block_N", reduce_tail_configs)
def test_reduce_max_tail(M, N, block_M, block_N, dtype, target):
    run_test_reduce_max_tail(M, N, block_M, block_N, dtype, target=target)


# =============================================================================
# Group 3 - CV fusion (matmul + add) tail   [risk: medium]
# Mirrors examples/simple_fusion/matmul_add.py, but the grid uses T.ceildiv with
# non-divisible M/N. C-scope (cube) tails ride gm2l1/l0c2gm; V-scope (vector,
# dual-AIV vid split) tails ride gm2ub/ub2gm. The same clamp formula covers the
# `bx*block_M + vid*block_M//VEC_NUM` per-vid offset. Manual cross-core sync, so
# no auto pass_configs (faithful to the example's plain @jit).
# =============================================================================
def cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
        D: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            d_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)  # gm2l1: M & K tail
                    T.copy(B[k * block_K, by * block_N], B_L1)  # gm2l1: K & N tail

                    T.barrier_all()
                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])  # l0c2gm: M & N tail
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)

                T.copy(C[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)  # gm2ub tail
                T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N], d_ub)

                T.barrier_all()
                T.tile.add(c_ub, c_ub, d_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])  # ub2gm tail

    return main


def run_test_cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, target):
    torch.manual_seed(0)
    func = cv_matmul_add_tail(M, N, K, block_M, block_N, block_K)
    # out_idx=[-2] -> C (A@B written by cube, then += D by vector). Faithful to
    # examples/simple_fusion/matmul_add.py: plain compile, manual sync, no auto
    # pass_configs.
    func = tilelang.compile(func, out_idx=[-2], target=target)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    d = torch.randn(M, N).half().npu()

    torch.npu.synchronize()
    c = func(a, b, d)

    ref_c = a @ b + d
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# (M, N, K, block_M, block_N, block_K) - M/N/K non-divisible.
cv_tail_configs = [
    (128 + 30, 256 + 16, 64 + 8, 128, 256, 64),   # (158, 272, 72)
    (256 + 33, 256 + 40, 128 + 5, 128, 256, 64),  # (289, 296, 133)
]


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,K,block_M,block_N,block_K", cv_tail_configs)
def test_cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, target):
    run_test_cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, target=target)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
