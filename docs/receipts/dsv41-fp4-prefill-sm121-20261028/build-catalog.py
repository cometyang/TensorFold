"""Rebuild changed scoring variants over the qualified catalog, retaining other cubins exactly."""
import copy,hashlib,json,pathlib,shutil,sys
root=pathlib.Path('/work')
sys.path.insert(0,str(root/'tools/dsv41-zig/fp4_index'))
from cuda_build import compile_score
base=pathlib.Path('/hf/tensorfold-dsv41/kit-abe07c7d/kit/aot')
old=pathlib.Path('/opt/tensorfold/share/dsv41-index-fp4')
out=root/'native-prefill';(out/'cubins').mkdir(parents=True,exist_ok=True);(out/'ptx').mkdir(exist_ok=True)
catalog=json.loads((old/'aot.json').read_text());proof=json.loads((old/'index-fp4.json').read_text())
original={e['hash']:e for e in json.loads((base/'aot.json').read_text())['kernels']}
entries={e['hash']:e for e in catalog['kernels']};changes={}
for row in proof['variants']:
 if row['function']!='_index_score_rows':continue
 prior=entries[row['native_hash']]
 binary,ptx,md=compile_score(original[row['original_hash']],out)
 digest=hashlib.sha256(binary).hexdigest();(out/'cubins'/(digest+'.cubin')).write_bytes(binary)
 (out/'ptx'/(digest+'.ptx')).write_text(ptx)
 replacement=copy.deepcopy(prior)
 replacement.update(hash=digest,name=md['name'],num_warps=md['num_warps'],shared=md['shared'],grid_x_divisor=md['grid_x_divisor'])
 changes[prior['hash']]=replacement
 row.update(native_hash=digest,shared=md['shared'])
 print(json.dumps({'function':row['function'],'original':row['original_hash'],'native':digest}),flush=True)
for i,entry in enumerate(catalog['kernels']):
 if entry['hash'] in changes:catalog['kernels'][i]=changes[entry['hash']]
 else:shutil.copy2(old/'cubins'/(entry['hash']+'.cubin'),out/'cubins'/(entry['hash']+'.cubin'))
proof['cuda_prefill_source_sha256']=hashlib.sha256((root/'zig/kernels/cuda/dsv41/index_fp4_prefill.cuh').read_bytes()).hexdigest()
(out/'aot.json').write_text(json.dumps(catalog,indent=1)+'\n')
(out/'index-fp4.json').write_text(json.dumps(proof,indent=2)+'\n')
assert changes
print(json.dumps({'changed_variants':len(changes),'unchanged_cubins':len(catalog['kernels'])-len(changes)}),flush=True)
