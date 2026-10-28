"""Compare native FP4 cubins with the deployed cubins and an independent FP64 score reference."""
import argparse
import gc
import json
from pathlib import Path

import torch
from driver import Kernel


def constants(entry):
    return {name: value['int'] for name, value in entry['consts'].items()}


def decoded_query(packed, rows):
    raw = packed.view(torch.uint8).reshape(rows, 8192)
    codes = raw[:, :2048].reshape(rows, 32, 64).long()
    unpacked = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(rows, 32, 128)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         device='cuda', dtype=torch.float64)
    scale = torch.exp2(raw[:, 2048:2176].reshape(rows, 32, 4).double() - 127).repeat_interleave(32, -1)
    return (table[unpacked] * scale).to(torch.bfloat16)


def key_scores(keys):
    ordered = (keys >> 32).to(torch.int32)
    return torch.where(ordered < 0, ordered ^ 0x7FFFFFFF, ordered).view(torch.float32)


def timed(call, repetitions):
    for _ in range(2):
        call()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--all-variants', action='store_true')
    parser.add_argument('--same-base', action='store_true')
    parser.add_argument('--rows', type=int, choices=(1, 5, 6, 16), default=16)
    parser.add_argument('--prefill-rows', type=int, choices=(16, 256, 2048), default=16)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.012)
    torch.manual_seed(641)
    original = json.loads((args.base / 'aot.json').read_text())['kernels']
    native = json.loads((args.native / 'aot.json').read_text())['kernels']
    marker = json.loads((args.native / 'index-fp4.json').read_text())
    mapping = {row['original_hash']: row['native_hash'] for row in marker['variants']}
    by_hash = {entry['hash']: entry for entry in native}
    stream = torch.cuda.current_stream().cuda_stream
    torch.empty(1, device='cuda')
    report = {'quantization': [], 'scores': [], 'passed': False, 'gpu': torch.cuda.get_device_name(),
              'precision_reference': 'Independent FP64 QK, ReLU and head-weight sum on sampled columns'}
    quant = next(entry for entry in original if entry['fn'] == '_fp4qd_p2')
    qbase, qnative = Kernel(args.base, quant), Kernel(args.native, by_hash[mapping[quant['hash']]])
    try:
        for rows in (1, 5, 16, 33):
            query = torch.randn((rows, 32, 128), device='cuda', dtype=torch.bfloat16)
            query[0, 1] *= 1e-30
            query[0, 2] *= 1e4
            query[0, 3] = 0
            query[0, 0, :16] = torch.tensor([0., -0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.,
                                            -.25, -.75, -1.25, -1.75, -3.5, -6.], device='cuda')
            ordinary, packed = torch.empty_like(query), torch.empty_like(query)
            values = {'X': query, 'nblk': query.numel() // 32}
            grid = ((values['nblk'] + 31) // 32, 1, 1)
            qbase.launch({**values, 'OUT': ordinary}, grid, stream)
            qnative.launch({**values, 'OUT': packed}, grid, stream)
            reconstructed = decoded_query(packed, rows)
            equal = torch.equal(ordinary.view(torch.int16), reconstructed.view(torch.int16))
            report['quantization'].append({'rows': rows, 'represented_bf16_bytes_equal': equal})
            if not equal:
                raise AssertionError('query representation differs')
        names = ('_index_score_rows', '_index_score', '_index_score_cand')
        contexts = (2048,) if args.quick or args.all_variants else (2048, 65536, 1048320)
        for name in names:
            entries = [entry for entry in original if entry['fn'] == name]
            entry = next((entry for entry in entries if not constants(entry).get('KEYS', name.endswith('cand'))
                          and not constants(entry).get('HAS_BASE', 0)
                          and not constants(entry).get('HAS_CAND', 0)), entries[0])
            selected = entries if args.all_variants else [entry]
            for entry in selected:
                const = constants(entry)
                first, second = Kernel(args.base, entry), Kernel(args.native, by_hash[mapping[entry['hash']]])
                try:
                    for requested_n in contexts:
                        n = const.get('n', const.get('nt', 0) * 64 or requested_n)
                        rows = args.prefill_rows if name == '_index_score_rows' else args.rows
                        if 'rows' in const:
                            rows = const['rows']
                        query = torch.randn((rows, 32, 128), device='cuda', dtype=torch.bfloat16)
                        ordinary, packed = torch.empty_like(query), torch.empty_like(query)
                        qvalues = {'X': query, 'nblk': query.numel() // 32}
                        qgrid = ((qvalues['nblk'] + 31) // 32, 1, 1)
                        qbase.launch({**qvalues, 'OUT': ordinary}, qgrid, stream)
                        qnative.launch({**qvalues, 'OUT': packed}, qgrid, stream)
                        codes = torch.randint(0, 256, (n + 128, 64), dtype=torch.uint8, device='cuda')
                        scales = torch.randint(123, 130, (n + 128, 4), dtype=torch.uint8, device='cuda')
                        weights = torch.randn((rows, 32), dtype=torch.bfloat16, device='cuda')
                        visible = torch.full((rows,), n, dtype=torch.int64, device='cuda')
                        if rows > 1:
                            visible[0] = 0
                            visible[1] = n - 17
                        bases = (torch.arange(rows, device='cuda', dtype=torch.int64) % 2) * 64
                        if args.same_base:
                            bases.zero_()
                        candidate_count = 2048
                        blocks = torch.arange(candidate_count, device='cuda', dtype=torch.int32).repeat(rows, 1)
                        blocks[blocks >= n // 8] = -1
                        mask = torch.ones((rows, (n + 7) // 8), dtype=torch.uint8, device='cuda')
                        mask[:, 1::3] = 0
                        candidate = name.endswith('cand')
                        width = candidate_count * 8 if candidate else n
                        keys = bool(const.get('KEYS', candidate))
                        dtype = torch.int64 if keys else torch.float32
                        output1 = torch.full((rows, width), -19, dtype=dtype, device='cuda')
                        output2 = output1.clone()
                        nt = (n + 63) // 64
                        tmax1 = torch.zeros((rows, nt), dtype=torch.int64, device='cuda')
                        tmax2 = tmax1.clone()
                        values = {'K': codes, 'KS': scales, 'Wt': weights, 'VIS': visible, 'n': n, 'rows': rows,
                                  'KB': bases, 'CAND': mask, 'cs': mask.shape[1], 'TMAX': tmax1, 'nt': nt,
                                  'CBLK': blocks, 'cbs': candidate_count, 'nout': width}
                        grid = ((rows + const.get('RB', 1) - 1) // const.get('RB', 1), (width + 63) // 64, 1)
                        run1 = lambda: first.launch({**values, 'Q': ordinary, 'OUT': output1}, grid, stream)
                        run2 = lambda: second.launch({**values, 'Q': packed, 'OUT': output2, 'TMAX': tmax2}, grid, stream)
                        run1()
                        run2()
                        torch.cuda.synchronize()
                        equal = torch.equal(output1, output2)
                        maxima_equal = torch.equal(tmax1, tmax2)
                        score1, score2 = (key_scores(output1), key_scores(output2)) if keys else (output1, output2)
                        finite = torch.isfinite(score1) & torch.isfinite(score2)
                        max_difference = (score1[finite] - score2[finite]).abs().max().item() if finite.any() else 0
                        sample = torch.arange(min(n, 256), device='cuda', dtype=torch.int64)
                        row = min(2, rows - 1)
                        physical = sample + (bases[row] if const.get('HAS_BASE', 0) else 0)
                        raw = codes[physical].long()
                        elements = torch.cat((raw & 15, raw >> 4), dim=-1)
                        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5,
                                              -2., -3., -4., -6.], dtype=torch.float64, device='cuda')
                        kvalues = table[elements] * torch.exp2(scales[physical].double() - 127).repeat_interleave(32, -1)
                        reference = ((ordinary[row].double() @ kvalues.T).clamp_min(0)
                                     * weights[row].double()[:, None]).sum(0)
                        eligible = finite[row, sample]
                        if const.get('HAS_CAND', 0):
                            eligible &= mask[row, sample // 8] != 0
                        err1 = (score1[row, sample].double() - reference).abs()[eligible]
                        err2 = (score2[row, sample].double() - reference).abs()[eligible]
                        no_worse = bool(torch.all(err2 <= err1 + 1e-10))
                        row_report = {'function': name, 'original_variant': entry['hash'], 'context': n, 'rows': rows,
                            'exact_output_equal': equal, 'tile_maxima_equal': maxima_equal,
                            'maximum_score_difference': max_difference, 'native_error_no_worse_than_baseline': no_worse,
                            'baseline_fp64_max_error': err1.max().item() if err1.numel() else 0,
                            'native_fp64_max_error': err2.max().item() if err2.numel() else 0,
                            'baseline_ms': timed(run1, 3), 'native_ms': timed(run2, 3)}
                        report['scores'].append(row_report)
                        print(json.dumps(row_report), flush=True)
                        args.out.write_text(json.dumps(report, indent=2) + '\n')
                        if not equal or not maxima_equal or not no_worse:
                            bad = torch.nonzero((output1 != output2) & finite)[:64]
                            if bad.numel() and not candidate:
                                physical_bad = bad[:, 1] + (bases[bad[:, 0]] if const.get('HAS_BASE', 0) else 0)
                                torch.save({'q': ordinary.cpu(), 'weights': weights.cpu(), 'at': bad.cpu(),
                                    'codes': codes[physical_bad].cpu(), 'scales': scales[physical_bad].cpu(),
                                    'baseline': score1[bad[:, 0], bad[:, 1]].cpu(),
                                    'native': score2[bad[:, 0], bad[:, 1]].cpu(), 'constants': const},
                                    args.out.with_suffix('.failure.pt'))
                            raise AssertionError('native score parity or FP64 gate failed')
                        del query, ordinary, packed, codes, scales, output1, output2, tmax1, tmax2
                        gc.collect()
                        torch.cuda.empty_cache()
                finally:
                    first.close()
                    second.close()
        report['passed'] = True
    finally:
        torch.cuda.synchronize()
        qbase.close()
        qnative.close()
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'score_cases': len(report['scores'])}))


if __name__ == '__main__':
    main()
