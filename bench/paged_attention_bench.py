"""Paged append attention microbenchmark: today's gather + masked SDPA against
the Triton kernel in train/history_paged_attention.py, one layer at a time.

CUDA only; on a CPU-only host it exits with a message. ``--smoke`` runs tiny
shapes to check the script (on CPU through Triton's interpreter, which it
enables before Triton is imported); smoke numbers are not timings.

Shapes follow the rollout census: 8 heads of depth 16, 64-token pages, batch
64/128 streams, query width W in {1, 4, 16, 128} (stream counts ~U[1, W], one
stream at W), ends ~U[1, max_end] (at least the stream's count), pages shuffled
through one pool. FP32 with TF32 off, under ``torch.inference_mode``.

Per (B, W) it reports, as medians of ``--repeats`` timed loops of ``--iters``
calls (Python launch cost included, as in rollout):
- ``reference``: per-layer ``index_select`` + masked SDPA over the padded span
  (``_append``'s power-of-two span, or ``--page-span``); ``setup`` is the row
  index and mask construction ``_append`` does once per append for all layers;
- ``kernel[...]``: the Triton kernel with the automatic launch and with each
  forced ``--splits``/``--num-warps`` choice, first-call (compile) time apart;
- ``GB/s``: valid K/V bytes (sum of ends x 2 x heads x depth x 4) over time;
- max |error| of each path against a float64 reference, all rows included.

Example (L40S): python bench/paged_attention_bench.py --json .work/paged_attention.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402


def parse_list(text: str, cast=int) -> list:
    return [cast(x) for x in text.split(",") if x.strip()]


def bucket(length: int) -> int:     # train.history_inference.bucket
    return max(32, 1 << (max(length, 1) - 1).bit_length())


def build_case(batch: int, width: int, args, generator: torch.Generator, device: torch.device):
    P, H, D = args.page, args.heads, args.depth
    counts = torch.randint(1, width + 1, (batch,), generator=generator)
    counts[0] = width
    ends = torch.maximum(torch.randint(1, args.max_end + 1, (batch,), generator=generator), counts)
    starts = ends - counts
    need = (ends + P - 1) // P
    longest = int(ends.max())
    span = -(-longest // P) * P if args.page_span else bucket(longest)
    blocks = -(-span // P)
    total = 1 + int(need.sum())
    ids = torch.randperm(total - 1, generator=generator) + 1
    table = torch.zeros(batch, blocks, dtype=torch.int64)
    used = 0
    for b in range(batch):
        n = int(need[b])
        table[b, :n] = ids[used:used + n]
        used += n
    kv = torch.randn(total * P, 2, H, D, generator=generator)
    kv[:P] = 0
    # Production layout: q is a strided view of the fused [B, W, 3, H, D] projection.
    qkv = torch.randn(batch, width, 3, H, D, generator=generator)
    return dict(q=qkv.to(device)[:, :, 0].transpose(1, 2), kv=kv.to(device),
                table=table.to(device), starts=starts.to(device), ends=ends.to(device),
                span=span, max_end=longest, valid_bytes=int(ends.sum()) * 2 * H * D * 4,
                gathered_bytes=batch * span * 2 * H * D * 4)


def timer(device: torch.device, iters: int, repeats: int):
    def measure(fn) -> float:
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(iters):
                    fn()
                end.record()
                torch.cuda.synchronize()
                samples.append(begin.elapsed_time(end) / iters)
            else:
                begin = time.perf_counter()
                for _ in range(iters):
                    fn()
                samples.append((time.perf_counter() - begin) * 1e3 / iters)
        return statistics.median(samples)
    return measure


def first_call(fn, device: torch.device) -> tuple[float, torch.Tensor]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    begin = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e3, out


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.double() - expected.double()).abs().max())


def run(args) -> dict:
    cuda = torch.cuda.is_available()
    if not cuda and not args.smoke:
        raise SystemExit("paged_attention_bench: no CUDA device; the timings need a GPU. "
                         "Use --smoke to check the script on CPU (Triton interpreter).")
    if args.smoke:
        if not cuda:
            os.environ["TRITON_INTERPRET"] = "1"     # before Triton's first import
        args.batches, args.widths, args.max_end = [2], [1, 4, 20], 200
        args.heads, args.iters, args.repeats = 2, 1, 1
    import train.history_paged_attention as paged   # imports Triton

    device = torch.device("cuda" if cuda else "cpu")
    if not paged.kernel_eligible(*(torch.zeros(1, args.heads, 1, args.depth, device=device),
                                   torch.zeros(args.page, 2, args.heads, args.depth, device=device),
                                   torch.zeros(1, 1, dtype=torch.int64, device=device),
                                   torch.zeros(1, dtype=torch.int64, device=device),
                                   torch.ones(1, dtype=torch.int64, device=device)), args.page):
        raise SystemExit("paged_attention_bench: the Triton kernel is not available here "
                         f"(triton={'missing' if paged.triton is None else paged.triton.__version__}, "
                         f"interpreted={paged.INTERPRETED})")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    measure = timer(device, args.iters, args.repeats)
    generator = torch.Generator().manual_seed(args.seed)
    P = args.page
    meta = dict(device=torch.cuda.get_device_name() if cuda else platform.processor() or "cpu",
                torch=torch.__version__, triton=paged.triton.__version__,
                interpreted=paged.INTERPRETED, ieee_dot=paged._ieee_dot(), smoke=args.smoke,
                heads=args.heads, depth=args.depth, page=P, max_end=args.max_end,
                page_span=args.page_span, iters=args.iters, repeats=args.repeats, seed=args.seed)
    print(" ".join(f"{k}={v}" for k, v in meta.items()))
    print(f"{'B':>4} {'W':>4} {'span':>5} {'variant':<22} {'ms':>9} {'x ref':>6} {'GB/s':>7} "
          f"{'first ms':>9} {'err fp64':>9} {'vs ref':>9}")
    rows = []
    with torch.inference_mode():
        for batch in args.batches:
            for width in args.widths:
                case = build_case(batch, width, args, generator, device)
                q, kv, table, starts, ends = (case[k] for k in ("q", "kv", "table", "starts", "ends"))
                span = case["span"]
                exact = paged.paged_append_attention_reference(
                    q.double(), kv.double(), table, starts, ends, P, span=span)
                positions = torch.arange(span, device=device)

                def setup():
                    rows_ = paged._rows(table, ends, positions, P).view(-1)
                    query = starts[:, None] + positions[:width]
                    allowed_ = ((positions[None, None] <= query[:, :, None])
                                & (positions[None, None] < ends[:, None, None]))
                    return rows_, allowed_

                gather_rows, allowed = setup()

                def reference():
                    packed = torch.index_select(kv, 0, gather_rows).view(
                        batch, span, 2, args.heads, args.depth).permute(2, 0, 3, 1, 4)
                    return F.scaled_dot_product_attention(q, packed[0], packed[1],
                                                          attn_mask=allowed[:, None], dropout_p=0.0)

                first, ref_out = first_call(reference, device)
                ref_ms = measure(reference)
                setup_ms = measure(setup)
                base = dict(batch=batch, width=width, span=span, max_end=case["max_end"],
                            valid_mb=case["valid_bytes"] / 1e6,
                            gathered_mb=case["gathered_bytes"] / 1e6)
                line = lambda name, ms, gbps, first_ms, err, diff: print(
                    f"{batch:>4} {width:>4} {span:>5} {name:<22} {ms:>9.4f} {ref_ms / ms:>6.2f} "
                    f"{gbps:>7.1f} {first_ms:>9.1f} {err:>9.2e} {diff:>9.2e}")
                gbps = lambda ms: case["valid_bytes"] / (ms * 1e-3) / 1e9
                err_ref = max_error(ref_out, exact)
                line("reference", ref_ms, gbps(ref_ms), first, err_ref, 0.0)
                print(f"{'':>15} {'setup (per append)':<22} {setup_ms:>9.4f}")
                rows.append(dict(base, variant="reference", ms=ref_ms, setup_ms=setup_ms,
                                 first_ms=first, err_fp64=err_ref))
                for splits in args.splits:
                    for warps in args.num_warps:
                        forced = None if splits == "auto" else int(splits)
                        config = paged.launch_config(
                            batch, args.heads, width, max(1, -(-case["max_end"] // P)),
                            sm_count=paged._sm_count(torch.cuda.current_device()) if cuda and not paged.INTERPRETED else None,
                            splits=forced, num_warps=warps)
                        name = f"kernel[{splits}->{config.splits},q{config.block_q},w{config.num_warps}]"

                        def kernel():
                            return paged.paged_append_attention_triton(
                                q, kv, table, starts, ends, P, max_end=case["max_end"],
                                splits=forced, num_warps=warps)

                        first, out = first_call(kernel, device)
                        ms = measure(kernel)
                        err, diff = max_error(out, exact), max_error(out, ref_out)
                        line(name, ms, gbps(ms), first, err, diff)
                        rows.append(dict(base, variant=name, splits=config.splits,
                                         block_q=config.block_q, num_warps=config.num_warps,
                                         ms=ms, speedup=ref_ms / ms, first_ms=first,
                                         err_fp64=err, diff_reference=diff))
                del case, q, kv, exact, ref_out
                if cuda:
                    torch.cuda.empty_cache()
    return dict(meta=meta, rows=rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batches", type=parse_list, default=[64, 128])
    parser.add_argument("--widths", type=parse_list, default=[1, 4, 16, 128])
    parser.add_argument("--max-end", type=int, default=3000)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--page-span", action="store_true",
                        help="reference pads to whole pages instead of the power-of-two bucket")
    parser.add_argument("--splits", type=lambda s: parse_list(s, str), default=["auto", "1", "2", "4", "8"])
    parser.add_argument("--num-warps", type=parse_list, default=[4])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny shapes to check the script (CPU: Triton interpreter); not timings")
    parser.add_argument("--json", type=Path, help="also write the rows here")
    args = parser.parse_args(argv)
    result = run(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
