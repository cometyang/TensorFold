"""Host tests for counter units, trace attribution, exactness and speed comparisons."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('traffic_analyze', Path(__file__).with_name('analyze.py'))
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


def batch(tps, draft=True, warmup=False):
    return {'warmup': warmup, 'runs': [{'fixture': 'fixture', 'context': 2048, 'draft': draft,
            'decode_tps': tps, 'complete': True, 'token_ids': [1, 2], 'tokens': 2}]}


def verify(sequence, context, rows, accepted=0):
    return {'event': 'verify', 'sequence': sequence, 'wire_step': sequence + 1, 'rows': rows,
            'max_context_end': context + 1, 'verify_ns': 2_000_000, 'total_ns': 2_100_000,
            'windows': [{'context': context + 1 - rows, 'rows': rows, 'proposed_context_end': context + 1,
                         'selected_drafts': rows - 1, 'potential_accepted_drafts': accepted,
                         'first_round': sequence == 0, 'bucket_changed': False}]}


def counter(rank, bucket, ordinal, context, rows, fill=2_097_152):
    return {'type': 'traffic_round', 'range': f'rank{rank}_b{bucket}_round{ordinal}', 'context_tokens': context,
            'rows': rows, 'rank': rank, 'values': {analyze.FILL: fill, analyze.READ: fill * 2,
                                                 analyze.WRITE: fill // 2, analyze.TIME: 1_000_000}}


class SpeedTests(unittest.TestCase):
    def test_bracket_excludes_warmup_and_profile_runs(self):
        receipt = {'phases': {
            'A1': {'requests': [batch(999, warmup=True), batch(70), batch(72), batch(74), batch(40, False)]},
            'A2': {'requests': [batch(74), batch(76), batch(78), batch(44, False)]},
            'B': {'requests': [batch(78), batch(80), batch(82), batch(43, False)]},
            'P': {'mode': 'profile', 'requests': [batch(1000)]}}}
        result = analyze.speed(receipt, ('A1', 'A2'), 80)
        self.assertEqual(len(result['contexts']), 1)
        group = result['contexts'][0]
        self.assertEqual(group['baseline_tps'], 74)
        self.assertEqual(group['candidate_tps'], 80)
        self.assertTrue(group['target_met'])
        self.assertAlmostEqual(group['speculative_gain_pct'], (80 / 74 - 1) * 100)
        self.assertAlmostEqual(group['plain_gain_pct'], (43 / 42 - 1) * 100)
        del receipt['phases']['A2']
        incomplete = analyze.speed(receipt, ('A1', 'A2'), 80)
        self.assertEqual(incomplete['contexts'], [])
        self.assertIsNone(incomplete['fixtures'][0]['baseline_bracket_tps'])

    def test_exactness_compares_arrays_and_keeps_unknowns(self):
        reference = {'phases': {'A1': {'requests': [batch(40, False)]}}}
        receipt = {'phases': {'B': {'requests': [batch(80)]}}}
        self.assertEqual(analyze.accuracy(receipt, reference)['counts'], {'matched': 1})
        receipt['phases']['B']['requests'][0]['exact_plain_reference'] = True
        receipt['phases']['B']['requests'][0]['runs'][0]['token_ids'] = [1, 3]
        self.assertEqual(analyze.accuracy(receipt, reference)['counts'], {'failed': 1})
        self.assertEqual(analyze.accuracy(receipt, None)['counts'], {'unknown': 1})


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.base = Path.cwd().resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix='traffic-analyze-test-', dir=self.base)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.base))
        self.events = [verify(0, 100, 17), verify(1, 2048, 2), verify(2, 2049, 2, 1),
                       verify(3, 2052, 3, 1), verify(4, 65536, 1), verify(5, 65538, 2, 1)]
        self.write_trace()
        files = [{'name': 'spec-trace.jsonl', 'file': 'spec-trace.jsonl', 'rank': 0}]
        for rank in (0, 1):
            records = [{'type': 'configuration', 'rank': rank, 'passes': 1},
                       counter(rank, 0, 1, 2049, 2), counter(rank, 0, 2, 2052, 3),
                       {'type': 'window_complete', 'bucket': 0, 'ranges': 2},
                       counter(rank, 1, 1, 65538, 2),
                       {'type': 'window_complete', 'bucket': 1, 'ranges': 1}]
            name = f'traffic-{rank}.jsonl'
            self.write_records(name, records)
            files.append({'name': name, 'file': name, 'rank': rank})
        self.receipt = {'phases': {'P': {'mode': 'profile', 'counter_files': files,
                        'requests': [{'runs': [{'context': 2048}]}, {'runs': [{'context': 65536}]}]}}}

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.base))
        self.temporary.cleanup()

    def write_records(self, name, records):
        (self.root / name).write_text('\n'.join(json.dumps(x) for x in records) + '\n', encoding='utf-8')

    def write_trace(self):
        self.write_records('spec-trace.jsonl', self.events)

    def test_sectors_and_yield_join_are_distinct_and_exact(self):
        issues = []
        result = analyze.traffic(self.receipt, self.root, issues, True)
        self.assertEqual(issues, [])
        self.assertEqual(len(result['groups']), 6)
        for group in result['groups']:
            self.assertEqual(group['sysmem_fill_GB']['mean'], 67_108_864 / 1e9)
            self.assertEqual(group['l2_sysmem_read_request_GB']['mean'], 134_217_728 / 1e9)
            self.assertEqual(group['l2_sysmem_write_request_GB']['mean'], 33_554_432 / 1e9)
            self.assertEqual(group['gpu_range_ms']['mean'], 1)
            self.assertEqual(group['sysmem_fill_inferred_GBps']['mean'], 67.108864)
            joined = group['trace_join']
            self.assertEqual(joined['matched_count'], 1)
            self.assertEqual(joined['estimated_potential_emitted_tokens']['mean'], 2)
            self.assertAlmostEqual(joined['sysmem_fill_bytes_per_potential_emitted_token_weighted'], 33_554_432)

    def test_wrong_context_does_not_silently_shift_trace_join(self):
        self.events[2]['max_context_end'] += 1
        self.events[2]['windows'][0]['context'] += 1
        self.events[2]['windows'][0]['proposed_context_end'] += 1
        self.write_trace()
        issues = []
        result = analyze.traffic(self.receipt, self.root, issues, True)
        self.assertEqual(len(issues), 2)
        self.assertTrue(all('exact trace rows/context join failed' in x for x in issues))
        self.assertEqual(sum(g['trace_join']['matched_count'] for g in result['groups']), 4)

    def test_digest_and_path_mismatches_are_rejected(self):
        bad = copy.deepcopy(self.receipt)
        bad['phases']['P']['counter_files'][1]['sha256'] = 'incorrect'
        bad['phases']['P']['counter_files'][2]['file'] = '../traffic.jsonl'
        issues = []
        result = analyze.traffic(bad, self.root, issues)
        self.assertEqual(result['groups'], [])
        self.assertTrue(any('digest mismatch' in x for x in issues))
        self.assertTrue(any('escapes receipt directory' in x for x in issues))


if __name__ == '__main__':
    unittest.main()
