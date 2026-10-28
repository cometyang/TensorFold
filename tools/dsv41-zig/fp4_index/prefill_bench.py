"""Compare prefill cubins with alternating CUDA-graph timings and exact output checks."""
import argparse
import json
import math
from pathlib import Path
import statistics

import torch
from driver import Kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261028)
    torch.empty(1, device='cuda')
    baseline = json.loads((args.baseline / 'aot.json').read_text())['kernels']
    candidate = json.loads((args.candidate / 'aot.json').read_text())['kernels']
    old_map = {x['original_hash']: x['native_hash'] for x in
               json.loads((args.baseline / 'index-fp4.json').read_text())['variants']}
    new_map = {x['original_hash']: x['native_hash'] for x in
               json.loads((args.candidate / 'index-fp4.json').read_text())['variants']}
    old_by_hash, new_by_hash = ({x['hash']: x for x in entries} for entries in (baseline, candidate))
    quant = Kernel(args.candidate, next(x for x in candidate if x['fn'] == '_fp4qd_p2'))
    report = {'passed': False, 'cases': [], 'method': 'seven alternating AB/BA graph-replay pairs; median CUDA events'}
    for keys in (False, True):
        entry = next(x for x in baseline if x['fn'] == '_index_score_rows'
                     and bool(x['consts'].get('KEYS', {}).get('int', 0)) == keys
                     and bool(x['consts'].get('TREE', {}).get('int', 0))
                     and bool(x['consts'].get('HAS_TMAX', {}).get('int', 0)) == keys
                     and 'n' not in x['consts'] and 'rows' not in x['consts'])
        original = next(k for k, v in old_map.items() if v == entry['hash'])
        new_entry = new_by_hash[new_map[original]]
        kernels = [Kernel(args.baseline, entry), Kernel(args.candidate, new_entry)]
        for rows, n in ((17, 65), (128, 1024), (128, 32768), (128, 524160), (2048, 2048), (2048, 32768)):
            query = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device='cuda')
            packed = torch.empty_like(query)
            quant.launch({'X': query, 'OUT': packed, 'nblk': query.numel() // 32},
                         ((query.numel() // 32 + 31) // 32, 1, 1), torch.cuda.current_stream().cuda_stream)
            nt = math.ceil(n / 64)
            values = {'Q': packed, 'K': torch.randint(0, 256, (n, 64), dtype=torch.uint8, device='cuda'),
                      'KS': torch.randint(123, 130, (n, 4), dtype=torch.uint8, device='cuda'),
                      'Wt': torch.randn((rows, 32), dtype=torch.bfloat16, device='cuda'),
                      'VIS': torch.full((rows,), n, dtype=torch.int64, device='cuda'), 'n': n, 'rows': rows, 'nt': nt}
            if rows == 17:
                values['VIS'][:4] = 0
                values['VIS'][4:8] = 1
                values['VIS'][8:12] = 64
                query.zero_()
                quant.launch({'X': query, 'OUT': packed, 'nblk': query.numel() // 32},
                             ((query.numel() // 32 + 31) // 32, 1, 1), torch.cuda.current_stream().cuda_stream)
            outputs = [torch.empty((rows, n), dtype=torch.int64 if keys else torch.float32, device='cuda') for _ in kernels]
            maxima = [torch.zeros((rows, nt), dtype=torch.int64, device='cuda') for _ in kernels]
            graph_list = []
            repeats = 5 if rows * n > 10000000 else 30
            for i, kernel in enumerate(kernels):
                arguments = {**values, 'OUT': outputs[i], 'TMAX': maxima[i]}
                rb = entry['consts'].get('RB', {}).get('int', 1)
                grid = (math.ceil(rows / rb), nt, 1)
                def call():
                    kernel.launch(arguments, grid, torch.cuda.current_stream().cuda_stream)
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                reference = outputs[i].clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(repeats):
                        call()
                outputs[i].fill_(19)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(outputs[i], reference), 'graph capture did not reproduce the kernel output'
                del reference
                graph_list.append(graph)
            equal = torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
            equal_max = torch.equal(maxima[0], maxima[1])
            assert equal and equal_max, (keys, rows, n, 'score/key or tile-maximum mismatch')
            samples = [[], []]
            for repeat in range(7):
                for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    graph_list[i].replay()
                    b.record()
                    b.synchronize()
                    samples[i].append(a.elapsed_time(b) / repeats)
            old_ms, new_ms = map(statistics.median, samples)
            row = {'keys': keys, 'rows': rows, 'scanned_keys': n, 'exact_output_equal': equal,
                   'tile_maxima_equal': equal_max, 'baseline_ms': old_ms, 'candidate_ms': new_ms,
                   'speedup': old_ms / new_ms, 'samples_ms': samples}
            report['cases'].append(row)
            args.out.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            del graph_list, graph, query, packed, values, outputs, maxima, arguments
            torch.cuda.empty_cache()
        for kernel in kernels:
            kernel.close()
    quant.close()
    report['passed'] = True
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
