"""Qualify exact selection against the original CUDA kernels and a CPU total-order reference."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import subprocess

import torch


class Kernel:
    def __init__(self, path, name, block):
        self.driver = C.CDLL('libcuda.so.1')
        self.module, self.function = C.c_void_p(), C.c_void_p()
        self.data = C.create_string_buffer(path.read_bytes())
        self.check(self.driver.cuModuleLoadDataEx(C.byref(self.module), self.data, 0, None, None))
        self.check(self.driver.cuModuleGetFunction(C.byref(self.function), self.module, name.encode()))
        self.block = block
        self.driver.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p] * 3

    def check(self, code):
        if code:
            text = C.c_char_p()
            self.driver.cuGetErrorString(code, C.byref(text))
            raise RuntimeError(f'CUDA {code}: {text.value}')

    def launch(self, data, output, k, vis=None, parts=None):
        rows, n = data.shape
        values = [C.c_uint64(data.data_ptr()), C.c_int64(data.stride(0)), C.c_int(n), C.c_int(k)]
        if vis is not None:
            values.append(C.c_uint64(vis.data_ptr()))
        if parts is not None:
            values.append(C.c_int(parts))
        values.append(C.c_uint64(output.data_ptr()))
        ptrs = (C.c_void_p * len(values))(*(C.cast(C.pointer(v), C.c_void_p) for v in values))
        stream = torch.cuda.current_stream().cuda_stream
        self.check(self.driver.cuLaunchKernel(self.function, rows, parts or 1, 1, self.block, 1, 1, 0, stream, ptrs, None))


def timed(call, repeats):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            call()
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1000 / repeats)
    return statistics.median(samples)


def score_keys(x):
    bits = x.contiguous().view(torch.int32)
    bits = torch.where(bits == -2147483648, 0, bits)
    ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits).to(torch.int64)
    positions = torch.arange(x.shape[1], dtype=torch.int64, device=x.device)
    return (ordered << 32) | (0xffffffff - positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = args.out.parent / 'build'
    build.mkdir(parents=True, exist_ok=True)
    source = build / 'topk.cu'
    source.write_text('#include "dsv41_ops.cu"\n#include "dsv41/topk_radix.cuh"\n')
    cubin = build / 'topk.cubin'
    result = subprocess.run(['/usr/local/cuda/bin/nvcc', '-cubin', '-arch=sm_121', '-std=c++20', '-O3',
        '--fmad=false', '--ftz=false', '-lineinfo', '-Xptxas=-v', '-I' + str(root / 'zig/kernels/cuda'),
        str(source), '-o', str(cubin)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (build / 'compile.log').write_text(result.stdout)
    if result.returncode:
        print(result.stdout, flush=True)
        result.check_returncode()
    torch.cuda.set_per_process_memory_fraction(0.008)
    torch.manual_seed(1028)
    torch.empty(1, device='cuda')
    kernels = {(kind, variant): Kernel(cubin, f'tf_ds_topk_{kind}{suffix}_kernel', block)
               for kind in ('i64', 'indices')
               for variant, suffix, block in [('base', '', 1024),
                                              ('radix', '_radix', 1024), ('parts', '_parts', 1024)]}
    kernels['indices', 'merge'] = Kernel(cubin, 'tf_ds_topk_indices_merge_kernel', 1024)
    report = {'gpu': torch.cuda.get_device_name(), 'seed': 1028, 'cases': [], 'passed': False,
              'timing': 'Median of five CUDA-event timings over replayed graphs; microseconds per launch'}
    shapes = [(1, n, 512) for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 524288, 1048320)]
    shapes += [(r, n, 512) for r in (6, 16, 128) for n in (16384, 32768)]
    shapes += [(1, n, k) for n, k in ((512, 512), (513, 511), (16387, 512), (131072, 2048))]
    if args.quick:
        shapes = [(1, 16384, 512), (1, 32768, 512), (6, 32768, 512), (1, 131072, 2048)]
    if args.sanitize:
        shapes = [(1, 513, 511), (3, 16387, 512), (1, 131072, 2048)]
    for rows, n, k in shapes:
        for distribution in ('random', 'ties', 'ascending', 'descending', 'masked', 'extremes'):
            host = torch.randn(rows, n + 7, dtype=torch.float32)
            if distribution == 'ties':
                host = torch.randint(-2, 3, (rows, n + 7)).float()
            elif distribution in ('ascending', 'descending'):
                host = torch.arange(n + 7).float().expand(rows, -1).clone()
                if distribution == 'descending':
                    host.neg_()
            elif distribution == 'masked':
                host[:, n // 3:] = -float('inf')
            elif distribution == 'extremes':
                values = torch.tensor([0., -0., float('inf'), -float('inf'), 1e-40, -1e-40, 3e38, -3e38])
                host = values.repeat((n + 14) // 8)[:n + 7].expand(rows, -1).clone()
            scores = host.to('cuda')[:, :n]
            host = host[:, :n]
            visible = torch.tensor([max(0, n - i * n // max(rows, 1)) for i in range(rows)],
                                   dtype=torch.int64, device='cuda')
            ordered = score_keys(host)
            expected_keys = torch.sort(ordered, descending=True, stable=True).values[:, :k]
            expected_idx = torch.sort(torch.argsort(ordered, descending=True, stable=True)[:, :k]).values
            expected_idx = torch.where(expected_idx < visible.cpu()[:, None], expected_idx, -1)
            for kind in ('i64', 'indices'):
                data = ordered.to('cuda') if kind == 'i64' else scores
                expected = expected_keys if kind == 'i64' else expected_idx
                vis = None if kind == 'i64' else visible
                variants = ['base'] + (['radix'] if n <= 32768 else []) + ['split']
                outputs = [torch.empty((rows, k), dtype=torch.int64, device='cuda') for _ in variants]
                width = 4096 if rows <= 8 else 16384
                parts = min(16, max(2, (n + width - 1) // width))
                if (n + parts - 1) // parts > 32768:
                    parts = (n + 32767) // 32768
                scratch = torch.empty((rows, parts * k), dtype=torch.int64, device='cuda')
                calls = []
                for variant, output in zip(variants, outputs):
                    if variant != 'split':
                        calls.append(lambda v=variant, o=output: kernels[kind, v].launch(data, o, k, vis))
                    else:
                        def split(output=output):
                            kernels[kind, 'parts'].launch(data, scratch, k, parts=parts)
                            merge = kernels[kind, 'radix' if kind == 'i64' else 'merge']
                            merge.launch(scratch, output, k, vis)
                        calls.append(split)
                for call in calls:
                    call()
                torch.cuda.synchronize()
                results = [out.cpu() for out in outputs]
                if kind == 'i64':
                    results = [torch.sort(out, descending=True).values for out in results]
                for variant, result in zip(variants, results):
                    assert torch.equal(result, expected), (kind, rows, n, k, distribution, variant)
                times = [timed(call, 10 if args.quick else 30) for call in calls] if not args.sanitize else [None] * len(calls)
                row = dict(kind=kind, rows=rows, n=n, k=k, distribution=distribution,
                           **{v + '_us': t for v, t in zip(variants, times)}, exact=True)
                report['cases'].append(row)
                if distribution == 'random':
                    print(json.dumps(row), flush=True)
                args.out.write_text(json.dumps(report, indent=2) + '\n')
            del data, scores, outputs, scratch
    report['passed'] = True
    report['case_count'] = len(report['cases'])
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'case_count': report['case_count']}), flush=True)


if __name__ == '__main__':
    main()
