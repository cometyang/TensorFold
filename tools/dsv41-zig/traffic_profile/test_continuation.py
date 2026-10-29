"""Test continuation lineage, failed-run retention, invalid metrics, explicit aliases and separate counter roots."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import analyze


def request(tps, fixture='short', draft=True, warmup=False, valid=True, rep=1):
    return {'rep': rep, 'warmup': warmup, 'metrics': {'valid': valid, 'error': None if valid else 'unattributed_generation'},
            'runs': [{'fixture': fixture, 'context': 2048 if fixture == 'short' else 1047040, 'draft': draft,
                      'decode_tps': tps, 'complete': True, 'token_ids': [1, 2], 'tokens': 2}]}


def phase(*requests, complete=True):
    return {'mode': 'speed', 'complete': complete, 'requests': list(requests)}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.base = Path.cwd().resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix='traffic-continuation-', dir=self.base)
        self.root = Path(self.temporary.name).resolve()
        self.primary = {'failure': {'type': 'RuntimeError', 'message': 'incomplete or unattributed request'},
                        'restored': True, 'ended_utc': '2026-10-29T01:00:00Z',
                        'plan': {'phases': [{'label': x} for x in ('A1', 'B115', 'B60', 'A2')]}, 'phases': {
            'A1': phase(request(70), request(40, draft=False), request(71, 'long'), request(41, 'long', False)),
            'B115': phase(request(75), request(999, 'long', warmup=True, valid=False), complete=False)}}
        self.continuation = {'restored': True, 'ended_utc': '2026-10-29T02:00:00Z', 'phases': {
            'R115': phase(request(999, 'long', warmup=True), request(77, 'long'), request(42, 'long', False)),
            'B350': phase(request(76), request(78, 'long')),
            'A2': phase(request(72), request(44, draft=False), request(73, 'long'), request(45, 'long', False))}}
        self.reference = {'phases': {'REF': phase(request(40, draft=False), request(41, 'long', False))}}
        self.a = self.write('initial.json', self.primary)
        self.b = self.write('continuation.json', self.continuation)
        self.ref = self.write('reference.json', self.reference)

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(self.base))
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def combined(self, **kwargs):
        return analyze.analyze(self.a, self.ref, continuation_paths=[self.b], phase_aliases=['R115=B115'], **kwargs)

    def test_explicit_alias_preserves_failed_source_and_builds_bracket(self):
        result = self.combined()
        self.assertTrue(result['original_service_restored'])
        self.assertTrue(result['composition']['any_input_failed'])
        self.assertTrue(result['composition']['baseline_spans_inputs'])
        self.assertEqual(result['inputs'][0]['failure'], self.primary['failure'])
        self.assertEqual(result['inputs'][0]['unstarted_phases'], ['B60', 'A2'])
        self.assertEqual(result['receipt_failure'], [{'input_index': 0, 'failure': self.primary['failure']}])
        self.assertEqual(result['inputs'][1]['receipt_basename'], 'continuation.json')
        self.assertFalse(next(p for p in result['inputs'][0]['phases'] if p['source_phase'] == 'B115')['complete'])
        self.assertIn('restoration', ' '.join(result['speed']['limitations']))
        cells = {(x['context'], x['phase']): x for x in result['speed']['contexts']}
        self.assertEqual(cells[(2048, 'B115')]['baseline_tps'], 71)
        self.assertEqual(cells[(1047040, 'B115')]['baseline_tps'], 72)
        self.assertEqual(cells[(1047040, 'B115')]['candidate_tps'], 77)
        rejected = result['speed']['rejected_requests']
        self.assertEqual(len(rejected), 1)
        self.assertTrue(rejected[0]['warmup'])
        self.assertEqual(rejected[0]['source_input'], 0)
        self.assertEqual(rejected[0]['metric_errors'], ['unattributed_generation'])
        self.assertEqual(result['exactness']['counts'], {'matched': 15})
        joined = [x for x in result['exactness']['comparisons'] if x.get('source_phase') == 'R115']
        self.assertEqual({x['phase'] for x in joined}, {'B115'})
        self.assertEqual({x['source_input'] for x in joined}, {1})
        self.assertEqual(json.loads(self.a.read_text(encoding='utf-8')), self.primary)

    def test_duplicate_phase_and_overlapping_alias_are_rejected(self):
        duplicate = copy.deepcopy(self.continuation)
        duplicate['phases']['A1'] = phase(request(72))
        self.write('continuation.json', duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicate phase A1'):
            self.combined()
        duplicate = copy.deepcopy(self.continuation)
        duplicate['phases']['R115']['requests'].append(request(88))
        self.write('continuation.json', duplicate)
        with self.assertRaisesRegex(ValueError, 'overlaps valid measurements'):
            self.combined()

    def test_alias_typos_and_repeated_receipt_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'alias source is absent'):
            analyze.analyze(self.a, continuation_paths=[self.b], phase_aliases=['TYPO=B115'])
        with self.assertRaisesRegex(ValueError, 'unique SOURCE=TARGET'):
            analyze.analyze(self.a, continuation_paths=[self.b], phase_aliases=['R115=B115', 'R115=OTHER'])
        with self.assertRaisesRegex(ValueError, 'duplicate receipt content'):
            analyze.analyze(self.a, continuation_paths=[self.a])

    def test_invalid_measured_rows_and_warmups_never_affect_speed(self):
        data = {'phases': {'A1': phase(request(70)), 'A2': phase(request(72)),
                          'B': phase(request(80), request(1000, valid=False),
                                     request(2000, warmup=True, valid=False))}}
        data['phases']['B']['requests'][1]['runs'][0]['tokens'] = 99
        result = analyze.speed(data, ('A1', 'A2'), 80)
        self.assertEqual(result['contexts'][0]['candidate_tps'], 80)
        self.assertEqual(result['settings_conflicts'], {})
        self.assertEqual(len(result['rejected_requests']), 2)

    def test_gzip_continuations_keep_decompressed_hashes_and_results(self):
        original = self.combined()
        packed = []
        for path in (self.a, self.b, self.ref):
            output = path.with_name(path.name + '.gz')
            output.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            packed.append(output)
        result = analyze.analyze(packed[0], packed[2], continuation_paths=[packed[1]], phase_aliases=['R115=B115'])
        self.assertEqual(result['speed'], original['speed'])
        self.assertEqual(result['exactness'], original['exactness'])
        self.assertEqual([x['decompressed_sha256'] for x in result['inputs']],
                         [x['decompressed_sha256'] for x in original['inputs']])
        self.assertEqual(result['inputs'][1]['receipt_basename'], 'continuation.json.gz')

    def test_counter_files_stay_relative_to_their_original_input(self):
        paths = []
        for index, fill in enumerate((10, 20)):
            directory = self.root / str(index)
            directory.mkdir()
            records = [{'type': 'configuration', 'passes': 1, 'rank': 0},
                       {'type': 'traffic_round', 'rank': 0, 'range': 'rank0_b0_round0', 'rows': 1,
                        'context_tokens': 2048, 'values': {analyze.FILL: fill, analyze.READ: 30,
                                                         analyze.WRITE: 5, analyze.TIME: 1000}},
                       {'type': 'window_complete', 'bucket': 0, 'ranges': 1}]
            (directory / 'traffic.jsonl').write_text('\n'.join(json.dumps(x) for x in records), encoding='utf-8')
            value = {'phases': {f'P{index}': {'mode': 'profile', 'requests': [{'runs': [{'fixture': 'short', 'context': 2048}]}],
                     'activation': [{'rank': 0, 'activation_env': {'TF_TRAFFIC_SKIP': '0', 'TF_TRAFFIC_RANGES': '1'}}],
                     'counter_files': [{'name': 'traffic.jsonl', 'file': 'traffic.jsonl', 'rank': 0}]}}}
            paths.append(self.write(f'{index}/receipt.json', value))
        result = analyze.analyze(paths[0], continuation_paths=[paths[1]])
        self.assertEqual(result['issues'], [])
        self.assertEqual([g['source_input'] for g in result['traffic']['groups']], [0, 1])
        self.assertEqual([g['sysmem_fill_GB']['mean'] for g in result['traffic']['groups']], [320 / 1e9, 640 / 1e9])


if __name__ == '__main__':
    unittest.main()
