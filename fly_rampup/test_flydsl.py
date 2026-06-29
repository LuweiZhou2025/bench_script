# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

NUM_WAVE = 4
NUM_CU = 1
N_ITER = 1
M_ITER = 1


def enable_dump_ir(enable_debug_info=True):
    import os
    import flydsl
    from flydsl.utils.env import DebugEnvManager
    from flydsl._mlir import ir

    DebugEnvManager.enable_debug_info = enable_debug_info
    DebugEnvManager.dump_asm = True
    DebugEnvManager.dump_ir = True
    DebugEnvManager.dump_dir = "my_ir_dumps"
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


@flyc.kernel
def copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    block_m = 4 * NUM_WAVE * M_ITER
    block_n = 64 * N_ITER

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)

    bA = fx.zipped_divide(A, (block_m, block_n))
    bB = fx.zipped_divide(B, (block_m, block_n))
    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))

    thr_layout = fx.make_layout((4 * NUM_WAVE, 16), (16, 1))
    val_layout = fx.make_layout((1, 4), (1, 1))
    # thr_layout = fx.make_layout((16, 4 * NUM_WAVE), (1, 16))
    # val_layout = fx.make_layout((4, 1), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)

    fx.utils.print_typst(tiled_copy, file="tiled_copy.typ")
    print(f"{tile_mn=}")
    print(f"{tv_layout=}")
    print(f"\n####\n{tiled_copy=}")
    thr_copy = tiled_copy.get_slice(tid)
    print(f"\n####\n{thr_copy=}")

    partition_src = thr_copy.partition_S(bA)
    print(f"\n####\n{partition_src=} \n")
    partition_dst = thr_copy.partition_D(bB)

    frag = fx.make_fragment_like(partition_src)

    fx.copy(copy_atom, partition_src, frag)
    fx.copy(copy_atom, frag, partition_dst)


@flyc.jit
def tiledCopy(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    copy_kernel(A, B).launch(grid=(NUM_CU, 1, 1), block=(NUM_WAVE * 64, 1, 1), stream=stream)


enable_dump_ir(True)
M, N = NUM_WAVE * 4 * NUM_CU * M_ITER, 64 * N_ITER
A = torch.arange(M * N, dtype=torch.float32).reshape(M, N).cuda()
B = torch.zeros(M, N, dtype=torch.float32).cuda()


tiledCopy(A, B, stream=torch.cuda.Stream())

torch.cuda.synchronize()

is_correct = torch.allclose(A, B)
print("Result correct:", is_correct)
if not is_correct:
    print("A:", A)
    print("B:", B)
