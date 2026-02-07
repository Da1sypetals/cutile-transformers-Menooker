# cutile-lsp: start
import math

import cuda.tile as ct


# Type aliases for constants
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def apply_splitk_reduce(out_splitk: ct.Tile, lse_splitk: ct.Tile, dtype: ct.DType) -> ct.Tile:
    NUM_KV_SPLITS_POW2, TILE_D = out_splitk.shape
    lse_max = ct.max(lse_splitk)

    # Compute sumexp_normalized_splitk
    sumexp_normalized_splitk = ct.exp2(lse_splitk - lse_max)
    sumexp_normalized_splitk = ct.astype(sumexp_normalized_splitk, ct.float32)

    # Compute sumexp_normalized
    sumexp_normalized = ct.sum(sumexp_normalized_splitk)

    # Compute numerator_normalized
    if NUM_KV_SPLITS_POW2 >= 16:
        mma_result = ct.mma(
            sumexp_normalized_splitk[None, :],
            ct.astype(out_splitk, ct.float32),
            ct.zeros((1, TILE_D), dtype=ct.float32),
        )
        numerator_normalized = ct.extract(mma_result, (0, 0), shape=(1, TILE_D))
    else:
        numerator_normalized = ct.sum(
            out_splitk * ct.reshape(sumexp_normalized_splitk, (NUM_KV_SPLITS_POW2, 1)),
            axis=0,
        )

    # Compute final accumulator
    acc = numerator_normalized / sumexp_normalized

    # Cast to output dtype before storing
    acc = ct.astype(acc, dtype)
    return acc


@ct.kernel(occupancy=4)
def splitk_reduce_kernel(
    attn_splitk_out,
    lse_splitk_out,
    attn_out,
    B: ConstInt,
    S_kv: ConstInt,
    num_heads: ConstInt,
    head_dim: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    NUM_KV_SPLITS_POW2: ConstInt,
    TILE_D: ConstInt,
    USE_DOT: ConstBool,
):
    """
    <typecheck>
    Tensor((1, 12, 8, 128), dtype="float16")
    Tensor((1, 12, 8), dtype="float32")
    Tensor((1, 12, 128), dtype="float16")
    1
    256
    12
    128
    8
    8
    128
    True
    </typecheck>
    """
    # Get program IDs
    batch_id = ct.bid(0)  # batch index
    head_id = ct.bid(1)  # head index
    tile_id = ct.bid(2)  # tile index

    # Get data type
    dtype = attn_out.dtype

    # Load intermediate attention results with latency hint
    out_splitk = ct.load(
        attn_splitk_out,
        (batch_id, head_id, 0, tile_id),
        shape=(1, 1, NUM_KV_SPLITS_POW2, TILE_D),
        order=(0, 1, 2, 3),
        allow_tma=True,
        latency=2,
    )
    out_splitk = ct.reshape(out_splitk, (NUM_KV_SPLITS_POW2, TILE_D))

    # Load and process lse results
    offs_lse = ct.arange(NUM_KV_SPLITS_POW2, dtype=ct.int32)
    lse_splitk = ct.gather(
        lse_splitk_out,
        (batch_id, head_id, offs_lse),
        padding_value=-math.inf,
    )
    acc = apply_splitk_reduce(out_splitk, lse_splitk, dtype)
    # Store final result with latency hint
    ct.store(
        attn_out,
        index=(batch_id, head_id, tile_id),
        tile=ct.reshape(acc, (1, 1, TILE_D)),
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
    )


# cutile-lsp: end


import json
import traceback
from pathlib import Path

import cuda.tile as ct
from cuda.tile._exception import Loc, TileError

from cutile_lsp.lsp_pipeline.drive_compiler_pipeline import Tensor, check_semantics_and_type


if __name__ == "__main__":
    _hints = []
    _diagnostics = []

    # Launch func definitions, code is indented outside

    def launc_splitk_reduce_kernel(_hints, _diagnostics):
        args = (
            Tensor((1, 12, 8, 128), dtype="float16"),
            Tensor((1, 12, 8), dtype="float32"),
            Tensor((1, 12, 128), dtype="float16"),
            1,
            256,
            12,
            128,
            8,
            8,
            128,
            True,
        )
        type_info = check_semantics_and_type(splitk_reduce_kernel, args)
        _hints.extend(type_info)

    # Launch kernels, code is indented outside
    launc_splitk_reduce_kernel(_hints, _diagnostics)

    # Save hints, code is indented outside
    lsp_code_exec_results = dict(hints=_hints, diagnostics=_diagnostics)
    with open("/home/da1sypetals/.cutile_lsp/1f06a1169c3002e5/lsp_code_exec_results.json", "w") as f:
        json.dump(lsp_code_exec_results, f)
