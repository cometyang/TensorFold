"""Recompute descriptive phase summaries and verify the saved exact-token and clock evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import struct

root = Path(__file__).resolve().parent
for folder in (root, root / 'confirmation'):
    if not (folder / 'results.json').exists():
        continue
    report = json.loads((folder / 'results.json').read_text())
    assert report['quality_passed'] and report['restored'] and report['lifecycle']['passed']
    reference = {(x['name'], x['draft']): x for x in report['phases']['A1']['requests']}
    for phase in report['phases'].values():
        for row in phase['requests']:
            ids = row['token_ids']
            assert row['complete'] and row['tokens'] == len(ids) == 128
            assert hashlib.sha256(struct.pack('<128I', *ids)).hexdigest() == row['token_ids_sha256']
            assert ids == reference[row['name'], row['draft']]['token_ids']
            assert ids == reference[row['name'], not row['draft']]['token_ids']
            if row['draft'] and row['name'].startswith('context-'):
                assert row['cache_state'] == 'cold' and row['cached_tokens'] == 0
        for row in phase['concurrent']:
            assert row['complete'] and row['token_ids'] == reference[row['prompt'], True]['token_ids']
    for row in report['lifecycle']['checks']:
        assert row['same_full_token_ids'] and row['token_ids'] == reference[row['fixture'], row['draft']]['token_ids']
    clock = [json.loads(x) for x in gzip.decompress((folder / 'clocks.jsonl.gz').read_bytes()).splitlines()]
    samples = [x for x in clock if 'devices' in x]
    assert samples
    for sample in samples:
        assert not sample.get('errors') and {d['rank'] for d in sample['devices']} == {'rank0', 'rank1'}
        assert all(max(d['graphics_mhz'], d['sm_mhz']) <= 2200 for d in sample['devices'])
    print(folder.name)
    for count in report['contexts']:
        result = {'context': count}
        for title, key, draft in [('ttft_s', 'ttft_s', True), ('plain_tps', 'decode_tps', False), ('draft_tps', 'decode_tps', True)]:
            phases = {label: statistics.median(x[key] for x in phase['requests'] if x['name'].startswith(f'context-{count}-') and x['draft'] == draft)
                      for label, phase in report['phases'].items()}
            baseline = (phases['A1'] + phases['A2']) / 2
            result[title] = {**phases, 'baseline_mean': baseline, 'candidate_change_pct': 100 * (phases['B'] / baseline - 1),
                             'baseline_drift_pct': 100 * (phases['A2'] / phases['A1'] - 1)}
        print(json.dumps(result))
    print(json.dumps({'token_checks_passed': True, 'clock_samples': len(samples), 'observed_max_mhz':
        {rank: max(max(d['graphics_mhz'], d['sm_mhz']) for x in samples for d in x['devices'] if d['rank'] == rank)
         for rank in ('rank0', 'rank1')}}))
for fixture in json.loads((root / 'fixtures.json').read_text()):
    data = gzip.decompress((root / fixture['path']).read_bytes())
    assert hashlib.sha256(data).hexdigest() == fixture['json_sha256']
    assert len(json.loads(data)) == fixture['tokens']
print('Fixture hashes and lengths passed.')
