# modified from https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/silu_and_mul.py
import cuda.tile as ct
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

ConstInt = ct.Constant[int]
def is_power_of_two_bitwise(n):
    return n > 0 and (n & (n - 1)) == 0

@ct.kernel
def silu_and_mul(
    input,
    output,
    hidden_size: ConstInt,
    approx: ct.Constant[bool],
):
    bid = ct.bid(0)  # this gives us our row

    # For 2D input (batch_size, 2*hidden_size), we need 2D indices
    # Row index is just bid (scalar), column indices are offsets-based
    row_idx = bid

    a_tile = ct.load(input, index=(row_idx, 0), shape=(1, hidden_size), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(input, index=(row_idx, 1), shape=(1, hidden_size), padding_mode=ct.PaddingMode.ZERO)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    # Implement sigmoid for SiLU
    denom = ct.add(1, ct.exp(-a_tile), flush_to_zero=True)
    rounding_mode = RMd.APPROX if approx else None
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=rounding_mode)

    # Perform SiLU(a) * b
    silu_a = ct.mul(a_tile, sigmoid_a, flush_to_zero=True)
    result = ct.mul(silu_a, b_tile, flush_to_zero=True)
    result = ct.astype(result, input.dtype)

    # Store result using scatter with 2D indices
    # output is also 2D: (batch_size, hidden_size)
    ct.store(output, index=(row_idx, 0), tile=result)

def launch_silu_and_mul(input: torch.Tensor, output: torch.Tensor, approx: bool = True):
    """
    input: (batch_size, 2*hidden_size)
    output: (batch_size, hidden_size)
    """
    stream = torch.cuda.current_stream()
    batch_size, total_hidden_size = input.shape
    hidden_size = total_hidden_size // 2
    grid = (batch_size, 1, 1)
    assert output.shape == (batch_size, hidden_size)
    assert is_power_of_two_bitwise(hidden_size), f"hidden_size must be power of 2, got {hidden_size}"
    ct.launch(
        stream,
        grid,
        silu_and_mul,
        (input, output, hidden_size, approx),
    )

if __name__ == "__main__":
    import torch
    # Simple test
    batch_size = 8
    hidden_size = 128
    input = torch.rand((batch_size, 2 * hidden_size), device='cuda', dtype=torch.float16)
    output = torch.zeros((batch_size, hidden_size), device='cuda', dtype=torch.float16)

    launch_silu_and_mul(input, output)

    # Verify results
    input_fp32 = input.float()
    a = input_fp32[:, :hidden_size]
    b = input_fp32[:, hidden_size:]
    expected = a * torch.sigmoid(a) * b
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)

    print("✓ silu_and_mul test passed!")