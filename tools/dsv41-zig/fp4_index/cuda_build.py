"""Specialize native SM121 CUDA scoring to each existing catalog launch ABI."""
from pathlib import Path
import subprocess


def compile_score(entry, out):
    root = Path(__file__).resolve().parents[3]
    directory = out / 'compiler'
    directory.mkdir(exist_ok=True)
    name = 'tf_index_' + entry['hash'][:20]
    constants = {key: value['int'] for key, value in entry['consts'].items()}
    if (constants.get('IH'), constants.get('ID'), constants.get('BN'), constants.get('PACKED')) != (32, 128, 64, 1):
        raise ValueError('unsupported indexer geometry or cache format')
    runtime = {p['name'] for p in entry['params']}
    types = {'*bf16': '__nv_bfloat16*', '*u8': 'uint8_t*', '*i64': 'int64_t*',
             '*i32': 'int32_t*', '*fp32': 'float*', 'i32': 'int32_t', 'u64': 'uint64_t'}
    parameters = ', '.join(types[p['type']] + ' ' + p['name'] for p in entry['params'])

    def value(key, default=0):
        return key if key in runtime else str(constants.get(key, default))

    rows = entry['fn'] == '_index_score_rows'
    candidates = entry['fn'] == '_index_score_cand'
    group = 1 if candidates else 4
    if group > 1:
        parameters += ', uint32_t launch_grid_x'
    flags = [constants.get('HAS_BASE', 0), constants.get('HAS_CAND', 0),
             1 if candidates else constants.get('KEYS', 0), constants.get('HAS_TMAX', 0),
             constants.get('SKIP_DEAD', 0), constants.get('DEAD_PAD', 0), int(candidates), constants.get('CB', 8),
             int(rows and constants.get('TREE', 0))]
    template = ','.join(str(v) for v in flags)
    arguments = ['sm', value('Q'), value('K'), value('KS'), value('Wt'), value('VIS'), value('OUT'), value('n'),
                 'reinterpret_cast<const int64_t*>(' + value('KB') + ')',
                 'reinterpret_cast<const uint8_t*>(' + value('CAND') + ')', value('cs'),
                 'reinterpret_cast<int64_t*>(' + value('TMAX') + ')', value('nt'),
                 'reinterpret_cast<const int32_t*>(' + value('CBLK') + ')', value('cbs'),
                 value('nout') if candidates else value('n'), 'r', 'reload', 'key_vis']
    call = 'tf_index_fp4::score<' + template + '>(' + ','.join(arguments) + ');'
    core_call = call
    if group > 1:
        rb = constants.get('RB', 1)
        count = value('rows') if rows else 'int(launch_grid_x)'
        start = f'int(blockIdx.x)*{group * rb}'
        end = f'min({start}+{group * rb},{count})'
        reuse = 'true'
        if constants.get('HAS_BASE', 0):
            reuse = f'({value("KB")}[first]=={value("KB")}[r])'
        prefix = f'const int first={start}, end={end}; bool reuse=true; int64_t maximum_vis=0;'
        prefix += f'for(int r=first;r<end;++r) {{ reuse=reuse && {reuse}; '
        prefix += f'maximum_vis=max(maximum_vis,{value("VIS")}[r]); }} '
        call = prefix + f'for(int r=first;r<end;++r) {{ bool reload=!reuse || r==first; '
        call += f'int64_t key_vis=reuse?maximum_vis:{value("VIS")}[r]; ' + core_call + ' }'
    else:
        call = f'const int r=blockIdx.x; const bool reload=true; const int64_t key_vis={value("VIS")}[r]; ' + call
    pdl = ('asm volatile("griddepcontrol.wait;"); asm volatile("griddepcontrol.launch_dependents;");'
           if constants.get('PDL', 0) else '')
    prefill = rows and bool(constants.get('TREE', 0))
    if prefill:
        if constants.get('HAS_BASE', 0) or constants.get('HAS_CAND', 0):
            raise ValueError('prefill specialization does not support pool bases or candidate masks')
        call = f'const int first={start}, end={end}; '
        call += 'tf_index_fp4::prefill<' + str(constants.get('KEYS', 0)) + ','
        call += str(constants.get('HAS_TMAX', 0)) + '>('
        call += ','.join(['sm', value('Q'), value('K'), value('KS'), value('Wt'), value('VIS'), value('OUT'),
                         value('n'), 'reinterpret_cast<int64_t*>(' + value('TMAX') + ')', value('nt'),
                         'first', 'end']) + ');'
    header = 'index_fp4_prefill.cuh' if prefill else 'index_fp4.cuh'
    storage = 'PrefillShared' if prefill else 'Shared'
    source = '#include "' + header + '"\nextern "C" __global__ __launch_bounds__(128) void '
    source += name + '(' + parameters + ') { ' + pdl + ' __shared__ tf_index_fp4::' + storage + ' sm; ' + call + ' }\n'
    path = directory / (name + '.cu')
    path.write_text(source)
    cubin = directory / (name + '.cubin')
    subprocess.run(['/usr/local/cuda/bin/nvcc', '-cubin', '-arch=sm_121a', '-std=c++17', '-O3',
                    '--fmad=false', '--ftz=false', '--keep', '--keep-dir', str(directory),
                    '-I' + str(root / 'zig/kernels/cuda/dsv41'), str(path), '-o', str(cubin)], check=True)
    ptx = (directory / (name + '.ptx')).read_text()
    if 'mma.sync.aligned.kind::mxf4nvf4.block_scale' not in ptx:
        raise RuntimeError('native FP4 MMA missing from CUDA output')
    return cubin.read_bytes(), ptx, {'name': name, 'num_warps': 4, 'num_ctas': 1, 'shared': 0,
        'global_scratch_size': 0, 'global_scratch_align': 1, 'profile_scratch_size': 0,
        'launch_pdl': bool(constants.get('PDL', 0)), 'grid_x_divisor': group}
