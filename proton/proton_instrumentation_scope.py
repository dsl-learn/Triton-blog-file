import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl


pl.enable_semantic("triton")


def launch_metadata(grid, metadata, args):
    n_elements = int(args["n_elements"])
    block_size = int(args["BLOCK_SIZE"])
    return {
        "name": f"instrumented_vec_add[{n_elements}]",
        "bytes": n_elements * 3 * 4,
        "flops": n_elements,
        "blocks": int(grid[0]),
        "block_size": block_size,
    }


@triton.jit(launch_metadata=launch_metadata)
def instrumented_vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    with pl.scope("kernel"):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        with pl.scope("load"):
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        with pl.scope("store"):
            tl.store(out_ptr + offsets, x + y, mask=mask)


def run_vec_add(n_elements, block_size):
    torch.manual_seed(1)
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, block_size),)
    instrumented_vec_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)
    return [round(float(v), 4) for v in out[:8].detach().cpu()]


def main():
    parser = argparse.ArgumentParser(description="Triton Proton in-kernel instrumentation example.")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--buffer-type", choices=["shared", "global"], default="shared")
    parser.add_argument("--profile-name", type=str, default="artifacts/proton_instrumentation")
    parser.add_argument("--dump-data", action="store_true", help="Dump proton.data.get() to <profile-name>.data.json.")
    args = parser.parse_args()

    Path(args.profile_name).parent.mkdir(parents=True, exist_ok=True)
    mode = proton.mode.Default(buffer_type=args.buffer_type, buffer_size=4096)
    session = proton.start(args.profile_name, backend="instrumentation", hook="triton", mode=mode)
    sample = run_vec_add(args.n, args.block_size)

    if session is not None and args.dump_data:
        data = proton.data.get(session)
        Path(args.profile_name + ".data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    if session is not None:
        proton.finalize(session)

    print("PASS", sample)


if __name__ == "__main__":
    main()
