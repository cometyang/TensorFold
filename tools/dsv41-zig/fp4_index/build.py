"""Compile a complete opt-in FP4 indexer catalog against an existing immutable kernel kit."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from cuda_build import compile_score

FUNCTIONS = {'_fp4qd_p2', '_index_score', '_index_score_rows', '_index_score_cand'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Compile-only probe; never emits a usable catalog')
    args = parser.parse_args()
    if not args.base_runtime.is_absolute() or args.out.resolve() == args.base.resolve():
        parser.error('runtime base must be absolute; output must differ from the immutable base')
    source = Path(__file__).with_name('kernels.py')
    spec = importlib.util.spec_from_file_location('fp4_index_kernels', source)
    kernels = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernels
    spec.loader.exec_module(kernels)
    baseline_bytes = (args.base / 'aot.json').read_bytes()
    catalog = json.loads(baseline_bytes)
    out = args.out
    (out / 'cubins').mkdir(parents=True, exist_ok=True)
    (out / 'ptx').mkdir(exist_ok=True)
    native, changed, compiled_count = [], [], 0
    for entry in catalog['kernels']:
        if entry['fn'] not in FUNCTIONS:
            target = out / 'cubins' / (entry['hash'] + '.cubin')
            if not target.exists() and not target.is_symlink():
                target.symlink_to(args.base_runtime / 'cubins' / target.name)
            native.append(entry)
            continue
        if args.limit and compiled_count >= args.limit:
            continue
        is_score = entry['fn'] != '_fp4qd_p2'
        if is_score:
            binary, ptx, metadata = compile_score(entry, out)
            md = SimpleNamespace(**metadata)
        else:
            fn = kernels._fp4qd_p2
            constants = {name: value['int'] for name, value in entry['consts'].items()}
            signature = {p['name']: p['type'] for p in entry['params']}
            signature.update({name: 'constexpr' for name in constants})
            signature = {name: signature[name] for name in fn.arg_names}
            attrs = {(fn.arg_names.index(p['name']),): [['tt.divisibility', 16]]
                     for p in entry['params'] if p['div16']}
            kernel = triton.compile(ASTSource(fn, signature, constants, attrs),
                target=GPUTarget('cuda', 121, 32), options={'num_warps': 4, 'enable_fp_fusion': False})
            binary, ptx, md = kernel.asm['cubin'], kernel.asm['ptx'], kernel.metadata
        (out / 'ptx' / (entry['fn'] + '-latest.ptx')).write_text(ptx)
        if is_score and 'mma.sync.aligned.kind::mxf4nvf4.block_scale' not in ptx:
            raise RuntimeError('compiler did not emit native FP4 MMA for ' + entry['fn'])
        if is_score and re.search(r'mma\.sync[^;]*\.bf16\.', ptx):
            raise RuntimeError('unexpected BF16 MMA emulation in ' + entry['fn'])
        digest = hashlib.sha256(binary).hexdigest()
        (out / 'cubins' / (digest + '.cubin')).write_bytes(binary)
        (out / 'ptx' / (digest + '.ptx')).write_text(ptx)
        replacement = copy.deepcopy(entry)
        replacement.update(hash=digest, name=md.name, num_warps=md.num_warps, num_ctas=md.num_ctas,
                           shared=md.shared, global_scratch=md.global_scratch_size,
                           global_align=md.global_scratch_align, profile_scratch=md.profile_scratch_size,
                           pdl=bool(md.launch_pdl))
        replacement['grid_x_divisor'] = getattr(md, 'grid_x_divisor', 1)
        if replacement['global_scratch'] or replacement['profile_scratch']:
            raise RuntimeError('compiled kernel requires unsupported scratch allocations')
        native.append(replacement)
        changed.append({'function': entry['fn'], 'original_hash': entry['hash'], 'native_hash': digest,
                        'warps': md.num_warps, 'shared': md.shared, 'native_fp4_mma': is_score})
        compiled_count += 1
        print(json.dumps(changed[-1]), flush=True)
    if args.limit:
        (out / 'probe.json').write_text(json.dumps(changed, indent=2) + '\n')
        return
    if {row['function'] for row in changed} != FUNCTIONS:
        raise RuntimeError('incomplete native indexer replacement set')
    metadata = {'schema_version': 2, 'index_heads': 32, 'index_head_dim': 128, 'query_stride_bytes': 8192,
                'base_manifest_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
                'kernel_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'cuda_source_sha256': hashlib.sha256((Path(__file__).resolve().parents[3] /
                    'zig/kernels/cuda/dsv41/index_fp4.cuh').read_bytes()).hexdigest(),
                'cuda_prefill_source_sha256': hashlib.sha256((Path(__file__).resolve().parents[3] /
                    'zig/kernels/cuda/dsv41/index_fp4_prefill.cuh').read_bytes()).hexdigest(),
                'triton_version': triton.__version__, 'arch': 121, 'variants': changed}
    (out / 'index-fp4.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'aot.json').write_text(json.dumps({'generator': 'fp4_index/build.py', 'kernels': native}, indent=1) + '\n')
    print(json.dumps({'compiled_variants': len(changed), 'catalog_kernels': len(native)}), flush=True)


if __name__ == '__main__':
    main()
