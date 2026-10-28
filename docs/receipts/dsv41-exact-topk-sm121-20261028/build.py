"""Build only the changed family fatbin and native server using the qualified remaining assets."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path.home() / 'codex-experiments'
repo = root / 'tensorfold-exact-topk'
base = root / 'tensorfold-native-fp4'
fatbins = repo / 'fatbins'
fatbins.mkdir(exist_ok=True)
found = {}
for source in (base / '.zig-cache/o').glob('*/*.fatbin'):
    if source.name not in found or source.stat().st_mtime > found[source.name].stat().st_mtime:
        found[source.name] = source
assert len(found) >= 21
hashes = {}
for name, source in found.items():
    shutil.copy2(source, fatbins / name)
    hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
(repo / 'baseline-fatbins.json').write_text(json.dumps(hashes, indent=2) + '\n')
image = 'tensorfold-dsv41:codex-native-fp4-73ee55ad604e'
common = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--memory=4g', '--memory-swap=4g',
          '--cpuset-cpus=0-3', '--user', '1000:1000', '-e', 'HOME=/tmp', '-v', str(repo) + ':/work', '-w', '/work']


def run(label, program, args):
    with (repo / (label + '.log')).open('w') as log:
        result = subprocess.run(common + ['--entrypoint', program, image] + args,
                                stdout=log, stderr=subprocess.STDOUT)
    print(label, result.returncode, flush=True)
    if result.returncode:
        print((repo / (label + '.log')).read_text()[-7000:], flush=True)
        result.check_returncode()


run('format', '/opt/zig/zig', ['fmt', 'zig/src/families/dsv41/topk_exact.zig'])
run('topk-fatbin', '/usr/local/cuda/bin/nvcc', ['-fatbin', '-gencode=arch=compute_121,code=sm_121', '-std=c++20',
    '-O3', '--fmad=false', '--ftz=false', '--expt-relaxed-constexpr', '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
    'zig/kernels/cuda/dsv41_ops.cu', '-o', 'fatbins/dsv41_ops.fatbin'])
run('host-tests', '/opt/zig/zig', ['build', 'test', '-j2', '--summary', 'all'])
run('server-build', '/opt/zig/zig', ['build', 'dsv41-serve', '-Doptimize=ReleaseFast', '-Dfatbins=/work/fatbins',
                                    '-j2', '--summary', 'all'])
run('sanitize-build', '/usr/local/cuda/bin/nvcc', ['-arch=sm_121', '-std=c++20', '-O3', '--fmad=false',
    '--ftz=false', '-lineinfo', '-Izig/kernels/cuda', 'tools/dsv41-zig/topk_exact/sanitize.cu', '-o', 'build/sanitize'])
print(json.dumps({'build_complete': True}), flush=True)
