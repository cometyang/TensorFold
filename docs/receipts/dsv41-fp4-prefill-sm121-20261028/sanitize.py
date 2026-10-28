"""Exercise all grouped FP4 scoring variants under Compute Sanitizer without loading PyTorch."""
import ctypes as C
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, '/work/tools/dsv41-zig/fp4_index')
from driver import Kernel

cuda = C.CDLL('libcuda.so.1')


def check(result):
    if result:
        raise RuntimeError('CUDA driver result ' + str(result))


class Buffer:
    def __init__(self, size, fill=0, content=None):
        self.address = C.c_uint64()
        check(cuda.cuMemAlloc_v2(C.byref(self.address), C.c_size_t(size)))
        check(cuda.cuMemsetD8_v2(self.address, C.c_ubyte(fill), C.c_size_t(size)))
        if content is not None:
            check(cuda.cuMemcpyHtoD_v2(self.address, C.c_char_p(content), C.c_size_t(len(content))))

    def data_ptr(self):
        return self.address.value

    def close(self):
        check(cuda.cuMemFree_v2(self.address))


check(cuda.cuInit(0))
context = C.c_void_p()
check(cuda.cuDevicePrimaryCtxRetain(C.byref(context), 0))
check(cuda.cuCtxSetCurrent(context))
directory = Path('/work/native-prefill')
entries = json.loads((directory / 'aot.json').read_text())['kernels']
quant = next(e for e in entries if e['fn'] == '_fp4qd_p2')
qkernel = Kernel(directory, quant)
count = 0
for entry in entries:
    if entry['fn'] != '_index_score_rows':
        continue
    const = {k: v['int'] for k, v in entry['consts'].items()}
    n = const.get('n', const.get('nt', 0) * 64 or 2048)
    rows = const.get('rows', 17 if entry['fn'] == '_index_score_rows' else 5)
    candidate = entry['fn'] == '_index_score_cand'
    width = 2048 * 8 if candidate else n
    nt = (n + 63) // 64
    owned = []

    def buffer(size, fill=0, content=None):
        value = Buffer(size, fill, content)
        owned.append(value)
        return value

    x = buffer(rows * 8192, content=struct.pack('<H', 0x3f80) * (rows * 4096))
    q = buffer(rows * 8192)
    qkernel.launch({'X': x, 'OUT': q, 'nblk': rows * 128}, (rows * 4, 1, 1), None)
    visible = [0, max(0, n - 17)] + [n] * max(0, rows - 2)
    base = buffer(rows * 8)
    values = {'Q': q, 'K': buffer((n + 128) * 64, 0x32), 'KS': buffer((n + 128) * 4, 127),
        'Wt': buffer(rows * 64, content=struct.pack('<H', 0x3f80) * (rows * 32)),
        'VIS': buffer(rows * 8, content=struct.pack('<' + 'q' * rows, *visible[:rows])),
        'OUT': buffer(rows * width * 8), 'n': n, 'rows': rows, 'KB': base,
        'CAND': buffer(rows * ((n + 7) // 8), 1), 'cs': (n + 7) // 8,
        'TMAX': buffer(rows * nt * 8), 'nt': nt,
        'CBLK': buffer(rows * 2048 * 4, content=struct.pack('<' + 'i' * (rows * 2048),
            *([i if i < n // 8 else -1 for i in range(2048)] * rows))), 'cbs': 2048, 'nout': width}
    kernel = Kernel(directory, entry)
    grid = ((rows + const.get('RB', 1) - 1) // const.get('RB', 1), (width + 63) // 64, 1)
    for mixed in (False, True):
        bases = struct.pack('<' + 'q' * rows, *[(i % 2) * 64 if mixed else 0 for i in range(rows)])
        check(cuda.cuMemcpyHtoD_v2(base.address, C.c_char_p(bases), C.c_size_t(len(bases))))
        kernel.launch(values, grid, None)
        check(cuda.cuCtxSynchronize())
    kernel.close()
    for value in owned:
        value.close()
    count += 1
    print(json.dumps({'variant': count, 'function': entry['fn'], 'passed': True}), flush=True)
qkernel.close()
print(json.dumps({'passed': True, 'score_variants': count, 'base_patterns': 2}), flush=True)
