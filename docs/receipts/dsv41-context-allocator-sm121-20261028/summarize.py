"""Summarize complete A1/B/A2 receipts with equal fixture weights and separate plain controls."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import statistics
import struct

LABELS = ('A1', 'B', 'A2')
COUNTERS = ('generation_tokens_total', 'prompt_tokens_total', 'mtp_drafted_total',
            'mtp_accepted_total', 'decode_rounds_total', 'request_decode_seconds_sum')
TPOT = 'request_time_per_output_token_seconds'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def geomean(values):
    require(bool(values) and all(positive(value) for value in values), 'invalid geometric-mean inputs')
    return math.exp(statistics.mean(math.log(value) for value in values))


def validate_batch(batch, fixture, tokens=1024, temperature=0):
    require(len(batch.get('runs', [])) == 1, 'main benchmark must have exactly one request per batch')
    row = batch['runs'][0]
    tag = f"{batch.get('label', 'sampled')}/{row.get('fixture')}/{batch.get('rep')}"
    require(row.get('complete') is True and not row.get('error'), tag + ': incomplete request')
    require(row.get('tokens') == tokens and row.get('max_tokens') == tokens, tag + ': wrong output count')
    ids = row.get('token_ids')
    require(isinstance(ids, list) and len(ids) == tokens
            and all(type(token) is int and 0 <= token < 2**32 for token in ids), tag + ': invalid full IDs')
    sha = hashlib.sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest()
    require(row.get('token_sha256') == sha, tag + ': output digest mismatch')
    require(row.get('fixture') == fixture['name'] and row.get('context') == fixture['context']
            and row.get('prompt_sha256') == fixture['prompt_sha256']
            and row.get('split') == 'holdout' and row.get('kind') == fixture['kind'],
            tag + ': fixture identity mismatch')
    require(row.get('temperature') == temperature and row.get('seed') == 1234
            and row.get('ignore_eos') is True and type(row.get('draft')) is bool, tag + ': protocol mismatch')
    require(positive(row.get('decode_tps')), tag + ': invalid client rate')
    pieces = row.get('pieces', [])
    require(len(pieces) >= 4 and all(isinstance(p, (list, tuple)) and len(p) == 2
            and type(p[0]) in (int, float) and math.isfinite(p[0]) and p[0] >= 0 for p in pieces),
            tag + ': invalid client arrival trace')
    require(all(b[0] >= a[0] for a, b in zip(pieces, pieces[1:]))
            and pieces[-1][0] > pieces[0][0], tag + ': invalid client timing interval')
    require(math.isclose(row['decode_tps'], (tokens - 1) / (pieces[-1][0] - pieces[0][0]), rel_tol=1e-10),
            tag + ': client rate does not match arrival trace')
    require(row.get('usage', {}).get('prompt_tokens') == fixture['context']
            and row.get('usage', {}).get('completion_tokens') == tokens, tag + ': inconsistent usage')
    retrieval = row.get('retrieval', {})
    require(set(retrieval) == set(fixture.get('expected', []))
            and all(type(value) is bool for value in retrieval.values()), tag + ': invalid retrieval evidence')
    metrics = batch.get('metrics', {})
    delta = metrics.get('delta', {})
    require(metrics.get('valid') is True, tag + ': unattributed server metrics')
    for counter in COUNTERS:
        value = delta.get(counter)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                tag + ': invalid counter ' + counter)
        before = batch.get('metrics_before', {}).get(counter)
        after = batch.get('metrics_after', {}).get(counter)
        require(type(before) in (int, float) and type(after) in (int, float)
                and math.isfinite(before) and math.isfinite(after)
                and math.isclose(after - before, value, rel_tol=1e-10, abs_tol=1e-10),
                tag + ': inconsistent counter delta ' + counter)
    require(delta['generation_tokens_total'] == tokens, tag + ': wrong attributed generation')
    seconds = delta['request_decode_seconds_sum']
    require(positive(seconds) and positive(batch.get('server_decode_tps_per_stream'))
            and math.isclose(batch['server_decode_tps_per_stream'], (tokens - 1) / seconds, rel_tol=1e-10),
            tag + ': invalid server completion rate')
    server_first_last = None
    names = (TPOT + '_sum', TPOT + '_count')
    before, after = batch['metrics_before'], batch['metrics_after']
    if all(name in before and name in after for name in names):
        count = after[names[1]] - before[names[1]]
        tpot = after[names[0]] - before[names[0]]
        require(count == 1 and positive(tpot), tag + ': incomplete TPOT attribution')
        server_first_last = 1 / tpot
    return row, server_first_last


def fixture_performance(records):
    values = {label: [batch['runs'][0]['decode_tps'] for batch, _ in records[label]
                      if batch['runs'][0]['draft'] and not batch['warmup']] for label in LABELS}
    medians = {label: statistics.median(data) for label, data in values.items()}
    servers = {label: statistics.median(batch['server_decode_tps_per_stream'] for batch, _ in records[label]
               if batch['runs'][0]['draft'] and not batch['warmup']) for label in LABELS}
    first_last_values = {label: [rate for batch, rate in records[label]
                        if batch['runs'][0]['draft'] and not batch['warmup']] for label in LABELS}
    available = all(rate is not None for values_ in first_last_values.values() for rate in values_)
    first_last = {label: statistics.median(values_) for label, values_ in first_last_values.items()} if available else None
    plain = {label: next(batch['runs'][0]['decode_tps'] for batch, _ in records[label]
                        if not batch['runs'][0]['draft']) for label in LABELS}
    plain_baseline = (plain['A1'] + plain['A2']) / 2
    baseline = (medians['A1'] + medians['A2']) / 2
    server_baseline = (servers['A1'] + servers['A2']) / 2
    result = {'client_samples_tps': values, 'client_medians_tps': medians, 'baseline_tps': baseline,
              'candidate_tps': medians['B'], 'improvement_pct': 100 * (medians['B'] / baseline - 1),
              'target_tps': baseline * 1.2, 'target_met': medians['B'] / baseline >= 1.2 - 1e-12,
              'baseline_drift_pct': 100 * (medians['A2'] / medians['A1'] - 1),
              'plain_control': {'requests_per_phase': 1, 'client_tps': plain,
                                'baseline_tps': plain_baseline, 'candidate_tps': plain['B'],
                                'improvement_pct': 100 * (plain['B'] / plain_baseline - 1),
                                'baseline_drift_pct': 100 * (plain['A2'] / plain['A1'] - 1),
                                'scope': 'single plain request per phase; diagnostic, excluded from primary speedup'},
              'speculative_vs_plain_gain_pct_by_phase': {label: 100 * (medians[label] / plain[label] - 1)
                                                          for label in LABELS},
              'client_observed_min_max_tps': {label: [min(data), max(data)] for label, data in values.items()},
              'server_request_completion_medians_tps': servers,
              'server_request_completion_improvement_pct': 100 * (servers['B'] / server_baseline - 1),
              'server_first_to_last_medians_tps': first_last,
              'server_first_to_last_samples_available': {label: sum(v is not None for v in data)
                                                        for label, data in first_last_values.items()}}
    if first_last:
        result['server_first_to_last_improvement_pct'] = 100 * (
            first_last['B'] / ((first_last['A1'] + first_last['A2']) / 2) - 1)
    return result


def context_performance(fixtures):
    baseline = geomean([value['baseline_tps'] for value in fixtures.values()])
    candidate = geomean([value['candidate_tps'] for value in fixtures.values()])
    pooled = {label: statistics.median([sample for value in fixtures.values()
              for sample in value['client_samples_tps'][label]]) for label in LABELS}
    phase_geomeans = {label: geomean([value['client_medians_tps'][label] for value in fixtures.values()])
                     for label in LABELS}
    result = {'aggregation': 'equal-weight geometric mean of per-fixture median-TPS ratios',
              'display_tps_statistic': 'geometric mean across fixture median TPS; A1/A2 bracket within each fixture',
              'fixture_count': len(fixtures), 'fixtures': list(fixtures), 'measured_requests_per_phase': 3 * len(fixtures),
              'baseline_tps': baseline, 'candidate_tps': candidate,
              'improvement_pct': 100 * (candidate / baseline - 1), 'target_tps': baseline * 1.2,
              'target_met': candidate / baseline >= 1.2 - 1e-12,
              'baseline_drift_pct': 100 * (phase_geomeans['A2'] / phase_geomeans['A1'] - 1),
              'client_phase_geomean_fixture_medians_tps': phase_geomeans,
              'client_pooled_medians_tps_descriptive_only': pooled,
              'fixture_improvement_range_pct': [min(v['improvement_pct'] for v in fixtures.values()),
                                                max(v['improvement_pct'] for v in fixtures.values())],
              'fixtures_meeting_target': sum(v['target_met'] for v in fixtures.values())}
    if all('plain_control' in value for value in fixtures.values()):
        plain_a = geomean([value['plain_control']['baseline_tps'] for value in fixtures.values()])
        plain_b = geomean([value['plain_control']['candidate_tps'] for value in fixtures.values()])
        result['plain_control'] = {'baseline_tps': plain_a, 'candidate_tps': plain_b,
                                   'improvement_pct': 100 * (plain_b / plain_a - 1),
                                   'requests_per_fixture_phase': 1,
                                   'scope': 'equal-weight geometric mean of single plain controls; diagnostic only'}
    for metric in ('request_completion', 'first_to_last'):
        field = 'server_' + metric + '_medians_tps'
        if all(value[field] is not None for value in fixtures.values()):
            a = geomean([(v[field]['A1'] + v[field]['A2']) / 2 for v in fixtures.values()])
            b = geomean([v[field]['B'] for v in fixtures.values()])
            result['server_' + metric] = {'baseline_tps': a, 'candidate_tps': b,
                                         'improvement_pct': 100 * (b / a - 1)}
        else:
            result['server_' + metric] = None
    return result


def summarize(receipt):
    holdout = [item for item in receipt['fixtures'] if item['split'] == 'holdout']
    names = {item['name']: item for item in holdout}
    require(len(names) == len(holdout), 'duplicate fixture names')
    contexts = receipt['contexts']
    require(len(contexts) == 3 and len(set(contexts)) == 3
            and {2048, 65536}.issubset(contexts) and 1000000 <= max(contexts) < 1048576,
            'expected 2K, 64K and near-1M contexts')
    require(set(item['context'] for item in holdout) == set(contexts), 'fixture contexts do not match receipt')
    for context in contexts:
        expected = {'retrieval'} if context >= 1000000 else {'prose', 'code', 'retrieval'}
        selected = [item for item in holdout if item['context'] == context]
        require(len(selected) == len(expected) and {item['kind'] for item in selected} == expected,
                f'{context}: incomplete workload coverage')
    records = {name: {label: [] for label in LABELS} for name in names}
    expected_keys = {(name, draft, rep) for name in names for draft in (True, False)
                     for rep in (range(4) if draft else (0,))}
    for label in LABELS:
        seen = set()
        for batch in receipt['phases'][label]['requests']:
            require(len(batch.get('runs', [])) == 1, label + ': expected single-request batches')
            name = batch['runs'][0].get('fixture')
            require(name in names, label + ': unknown holdout fixture')
            row, first_last = validate_batch(batch, names[name])
            rep = batch.get('rep')
            require(type(rep) is int and batch.get('label') == label
                    and type(batch.get('warmup')) is bool
                    and batch['warmup'] == bool(row['draft'] and rep == 0), label + ': wrong phase/rep/warmup')
            key = (name, row['draft'], rep)
            require(key in expected_keys and key not in seen, label + ': duplicate or unexpected request')
            seen.add(key)
            records[name][label].append((batch, first_last))
        require(seen == expected_keys, label + ': incomplete request matrix')
    summary = {'schema_version': 2, 'source_commit': 'bd6fa1255bb1a9f37cedb045d770b5f94207c3df',
               'baseline_commit': '6af4fd8a083323644651f65397f21ca61b4cdd2f',
               'clock_ceiling_mhz': receipt['clock_ceiling_mhz'], 'contexts': {}, 'fixtures': {},
               'greedy_comparisons': [], 'target_decode_throughput_gain_pct': 20,
               'restored': receipt.get('restored'), 'receipt_failure': receipt.get('failure'),
               'protocol': {'tokens': 1024, 'temperature': 0, 'seed': 1234, 'ignore_eos': True,
                            'warmup_requests_per_fixture_phase': 1, 'measured_reps': 3,
                            'near_1m_holdout_fixtures': 1,
                            'primary_metric': 'client post-first-text-arrival decode tokens/s',
                            'server_completion_metric': '(N-1)/(request completion minus first generated-token arrival)',
                            'server_first_last_metric': 'inverse single-request TPOT histogram delta, when count delta is 1',
                            'uncertainty': 'three repeats per fixture; observed ranges are not confidence intervals'}}
    require(summary['clock_ceiling_mhz'] == 2200, 'receipt clock ceiling does not match protocol')
    for name, item in names.items():
        summary['fixtures'][name] = fixture_performance(records[name])
        summary['fixtures'][name].update(context=item['context'], kind=item['kind'])
        reference = next(batch['runs'][0] for batch, _ in records[name]['A1'] if not batch['runs'][0]['draft'])
        for label in LABELS:
            for batch, _ in records[name][label]:
                row = batch['runs'][0]
                if label == 'A1' and not row['draft']:
                    continue
                summary['greedy_comparisons'].append({'fixture': name, 'phase': label, 'rep': batch['rep'],
                    'draft': row['draft'], 'warmup': batch['warmup'],
                    'equal_full_token_ids': row['token_ids'] == reference['token_ids']})
    for context in contexts:
        summary['contexts'][str(context)] = context_performance({name: value for name, value in
            summary['fixtures'].items() if value['context'] == context})
    summary['greedy_compared'] = len(summary['greedy_comparisons'])
    summary['greedy_matched'] = sum(row['equal_full_token_ids'] for row in summary['greedy_comparisons'])
    summary['greedy_completed_requests'] = sum(len(receipt['phases'][label]['requests']) for label in LABELS)
    long_fixture = next(item for item in holdout if item['context'] == max(contexts))
    sampled = receipt.get('near_1m_sampled', [])
    require(len(sampled) == 2, 'missing near-1M sampled plain/spec pair')
    sampled_rows = [validate_batch(batch, long_fixture, 128, .7)[0] for batch in sampled]
    require([row['draft'] for row in sampled_rows] == [False, True], 'invalid sampled plain/spec pair')
    summary['sampled_near_1m_exact'] = sampled_rows[0]['token_ids'] == sampled_rows[1]['token_ids']
    require(receipt.get('near_1m_sampled_exact') == summary['sampled_near_1m_exact'], 'stale sampled exactness flag')
    lifecycle = receipt.get('lifecycle', {})
    checks, cancellations = lifecycle.get('checks', []), lifecycle.get('cancellations', [])
    summary['lifecycle_passed'] = bool(lifecycle.get('passed') is True and checks and len(cancellations) == 2
                                      and all(item.get('passed') is True for item in checks + cancellations))
    summary['unique_retrieval_records'] = {label: [] for label in LABELS}
    for label in LABELS:
        for name, item in names.items():
            if item['kind'] != 'retrieval':
                continue
            row = next(batch['runs'][0] for batch, _ in records[name][label]
                       if batch['rep'] == 1 and batch['runs'][0]['draft'])
            summary['unique_retrieval_records'][label].append({'context': row['context'], 'fixture': name,
                'correct': sum(row['retrieval'].values()), 'total': len(row['retrieval'])})
    summary['exactness_passed'] = bool(summary['greedy_matched'] == summary['greedy_compared']
                                     and summary['sampled_near_1m_exact'] and summary['lifecycle_passed'])
    summary['validation'] = {'complete_request_matrix': True, 'all_main_and_sampled_batches_revalidated': True,
                             'clock_telemetry_checked_here': False}
    return summary


def self_test():
    # Synthetic workload imbalance: pooled median hides the third fixture's regression.
    def perf(a, b):
        return {'baseline_tps': a, 'candidate_tps': b, 'improvement_pct': 100 * (b/a-1),
                'target_met': b/a >= 1.2, 'client_medians_tps': {'A1': a, 'B': b, 'A2': a},
                'client_samples_tps': {'A1': [a]*3, 'B': [b]*3, 'A2': [a]*3},
                'server_request_completion_medians_tps': {'A1': a, 'B': b, 'A2': a},
                'server_first_to_last_medians_tps': None}
    context = context_performance({'a': perf(10, 11), 'b': perf(100, 110), 'c': perf(1000, 500)})
    require(math.isclose(context['improvement_pct'], 100*((1.1*1.1*.5)**(1/3)-1)), 'bad equal-weight aggregate')
    require(context['improvement_pct'] < 0 and context['client_pooled_medians_tps_descriptive_only']['B'] == 110,
            'pooling regression test failed')
    fixtures = [{'name': f'{kind}-{context}', 'split': 'holdout', 'kind': kind, 'context': context,
                 'prompt_sha256': 'a'*64, 'expected': ['secret'] if kind == 'retrieval' else []}
                for context in (2048, 65536, 1047040)
                for kind in (('retrieval',) if context > 1000000 else ('prose', 'code', 'retrieval'))]
    def batch(item, label, draft, rep, tokens=1024, temperature=0):
        ids = [1]*tokens
        delta = dict.fromkeys(COUNTERS, 0)
        delta.update(generation_tokens_total=tokens, prompt_tokens_total=item['context'], request_decode_seconds_sum=10)
        before = dict.fromkeys(COUNTERS, 0)
        before.update({TPOT+'_count': 0, TPOT+'_sum': 0})
        after = dict(delta)
        after.update({TPOT+'_count': 1, TPOT+'_sum': 9/(tokens-1)})
        row = dict(fixture=item['name'], split='holdout', kind=item['kind'], context=item['context'],
                   prompt_sha256=item['prompt_sha256'], tokens=tokens, max_tokens=tokens, token_ids=ids,
                   token_sha256=hashlib.sha256(struct.pack(f'<{tokens}I', *ids)).hexdigest(), complete=True,
                   temperature=temperature, seed=1234, ignore_eos=True, draft=draft,
                   decode_tps=(tokens-1)/9, pieces=[[1, 1], [3, 1], [6, 1], [10, 1]],
                   usage={'prompt_tokens': item['context'], 'completion_tokens': tokens},
                   retrieval={value: True for value in item['expected']})
        return dict(runs=[row], label=label, rep=rep, warmup=bool(draft and rep == 0),
                    metrics={'valid': True, 'delta': delta}, metrics_before=before, metrics_after=after,
                    server_decode_tps_per_stream=(tokens-1)/10)
    receipt = {'contexts': [2048, 65536, 1047040], 'fixtures': fixtures, 'clock_ceiling_mhz': 2200,
               'phases': {label: {'requests': [batch(item, label, draft, rep) for item in fixtures
                           for draft in (True, False) for rep in (range(4) if draft else (0,))]} for label in LABELS},
               'near_1m_sampled': [batch(fixtures[-1], 'sampled', draft, 0, 128, .7) for draft in (False, True)],
               'near_1m_sampled_exact': True, 'lifecycle': {'passed': True, 'checks': [{'passed': True}],
                                                          'cancellations': [{'passed': True}]*2}, 'restored': True}
    result = summarize(receipt)
    require(result['greedy_compared'] == 98 and result['greedy_completed_requests'] == 105
            and result['exactness_passed'] and result['contexts']['2048']['server_first_to_last'] is not None,
            'complete receipt check failed')
    cases = []
    damaged = copy.deepcopy(receipt); damaged['phases']['A2']['requests'].pop(); cases.append(damaged)
    damaged = copy.deepcopy(receipt); damaged['phases']['B']['requests'].append(damaged['phases']['B']['requests'][0]); cases.append(damaged)
    damaged = copy.deepcopy(receipt); damaged['phases']['A1']['requests'][4]['runs'][0]['complete'] = False; cases.append(damaged)
    damaged = copy.deepcopy(receipt); damaged['phases']['B']['requests'][0]['metrics']['valid'] = False; cases.append(damaged)
    damaged = copy.deepcopy(receipt); damaged['near_1m_sampled'][1]['runs'][0]['token_ids'].pop(); cases.append(damaged)
    damaged = copy.deepcopy(receipt); damaged['phases']['B']['requests'][1]['server_decode_tps_per_stream'] = math.nan; cases.append(damaged)
    damaged = copy.deepcopy(receipt); damaged['phases']['B']['requests'][1]['metrics_after'][TPOT+'_count'] = 2; cases.append(damaged)
    for damaged in cases:
        try:
            summarize(damaged)
        except ValueError:
            continue
        raise AssertionError('invalid receipt accepted')
    # A complete divergent output fails exactness instead of being mistaken for missing evidence.
    damaged = copy.deepcopy(receipt)
    changed = damaged['phases']['B']['requests'][0]['runs'][0]
    changed['token_ids'][10] = 2
    changed['token_sha256'] = hashlib.sha256(struct.pack('<1024I', *changed['token_ids'])).hexdigest()
    require(not summarize(damaged)['exactness_passed'], 'complete token divergence hidden')
    # Missing optional TPOT metrics must remove corroboration, not invalidate primary timing.
    damaged = copy.deepcopy(receipt)
    for side in ('metrics_before', 'metrics_after'):
        for suffix in ('_sum', '_count'):
            del damaged['phases']['B']['requests'][1][side][TPOT+suffix]
    require(summarize(damaged)['contexts']['2048']['server_first_to_last'] is None, 'missing TPOT hidden')
    print('self-test passed: equal-weight aggregation, complete matrix, timing/IDs, 7 rejection cases, optional TPOT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--receipt', type=Path, default=Path(__file__).resolve().parent/'collected/receipt.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'summary.json')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    summary = summarize(json.loads(args.receipt.read_text()))
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: summary[key] for key in ['contexts', 'greedy_matched', 'greedy_compared',
                      'exactness_passed', 'unique_retrieval_records', 'restored']}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
