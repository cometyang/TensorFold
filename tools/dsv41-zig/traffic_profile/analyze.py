"""Analyze verification traffic receipts without treating L2 requests as DRAM traffic."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zlib

FILL = 'lts__d_sectors_fill_sysmem.sum'
READ = 'lts__t_sectors_aperture_sysmem_op_read.sum'
WRITE = 'lts__t_sectors_aperture_sysmem_op_write.sum'
TIME = 'gpu__time_duration.sum'
METRICS = {FILL: ('sysmem_fill_GB', 32 / 1e9), READ: ('l2_sysmem_read_request_GB', 32 / 1e9),
           WRITE: ('l2_sysmem_write_request_GB', 32 / 1e9), TIME: ('gpu_range_ms', 1 / 1e6)}


def input_bytes(path):
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix.lower() == '.gz' else raw


def read_json(path):
    return json.loads(input_bytes(path).decode('utf-8-sig'))


def confined_input(directory, relative):
    root = directory.resolve()
    path = (directory / str(relative).replace('\\', '/')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('counter path escapes receipt directory')
    if not path.is_file() and path.suffix.lower() != '.gz':
        path = Path(str(path) + '.gz').resolve()
        if not path.is_relative_to(root):
            raise ValueError('compressed counter path escapes receipt directory')
    return path


def bucket(context):
    return 0 if context <= 4096 else 1 if context <= 131072 else 2


def stats(values):
    if not values:
        return {'count': 0, 'median': None, 'mean': None, 'min': None, 'max': None}
    return {'count': len(values), 'median': statistics.median(values), 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values)}


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def geomean(values):
    values = list(values)
    return values[0] if all(v == values[0] for v in values) else statistics.geometric_mean(values)


def requests(receipt):
    for label, phase in receipt.get('phases', {}).items():
        for batch in phase.get('requests', []):
            for row in batch.get('runs', []):
                yield label, phase.get('mode', batch.get('mode', 'speed')), batch, row


def lineage(batch):
    return {key: batch[key] for key in ('source_input', 'source_phase', 'source_batch') if key in batch}


def metrics_valid(batch, row):
    return all(not isinstance(owner.get('metrics'), dict) or owner['metrics'].get('valid') is not False
               for owner in (batch, row))


def speed_eligible(batch, row):
    return (metrics_valid(batch, row) and row.get('complete') is True and not row.get('error')
            and finite(row.get('decode_tps'), True))


def plain_references(receipt, preferred='A1'):
    candidates = list(requests(receipt))
    candidates.sort(key=lambda item: item[0] != preferred)
    references = {}
    for label, _, _, row in candidates:
        ids = row.get('token_ids')
        if row.get('draft') is False and row.get('complete') is True and isinstance(ids, list) and ids:
            references.setdefault(row['fixture'], (label, ids))
    return references


def accuracy(receipt, reference):
    refs = plain_references(reference) if reference is not None else {}
    refs.update({k: v for k, v in plain_references(receipt).items() if k not in refs})
    details = []
    for label, mode, batch, row in requests(receipt):
        fixture = row.get('fixture')
        ids = row.get('token_ids')
        ref = refs.get(fixture)
        match = None
        independent = False
        if ref is not None and isinstance(ids, list) and ids and len(ids) <= len(ref[1]):
            match = ids == ref[1][:len(ids)]
            independent = reference is not None or row.get('draft') is not False or label != ref[0]
        valid_output = row.get('complete') is True and not row.get('error')
        if isinstance(ids, list):
            valid_output = valid_output and len(ids) == row.get('tokens')
        claim = batch.get('exact_plain_reference')
        status = 'failed' if match is False or claim is False or not valid_output else (
            'matched' if match is True and independent else 'self_reference' if match is True else 'unknown')
        details.append({'phase': label, 'mode': mode, 'fixture': fixture, 'draft': row.get('draft'),
                        'rep': batch.get('rep'), 'warmup': batch.get('warmup', False), 'tokens': row.get('tokens'),
                        'status': status, 'token_ids_compared': match is not None,
                        'controller_exact_claim': claim, 'complete': valid_output, **lineage(batch)})
    counts = dict(Counter(item['status'] for item in details))
    return {'counts': counts, 'all_independent_comparisons_match': (
        counts.get('matched', 0) > 0 and counts.get('failed', 0) == 0 and counts.get('unknown', 0) == 0),
        'comparisons': details}


def speed(receipt, baselines, target):
    grouped = defaultdict(list)
    rejected = []
    eligible = []
    settings = defaultdict(lambda: defaultdict(list))
    for label, mode, batch, row in requests(receipt):
        if mode != 'speed':
            continue
        if not metrics_valid(batch, row):
            rejected.append({'phase': label, 'fixture': row.get('fixture'), 'rep': batch.get('rep'),
                             'warmup': batch.get('warmup', False), 'reason': 'invalid request metrics',
                             'metric_errors': [owner['metrics'].get('error') for owner in (batch, row)
                                               if isinstance(owner.get('metrics'), dict)
                                               and owner['metrics'].get('valid') is False], **lineage(batch)})
            continue
        if batch.get('warmup', False):
            continue
        if not speed_eligible(batch, row):
            rejected.append({'phase': label, 'fixture': row.get('fixture'), 'rep': batch.get('rep'), **lineage(batch)})
            continue
        fixture = row.get('fixture')
        values = {key: row.get(key) for key in (
            'tokens', 'max_tokens', 'temperature', 'seed', 'context', 'prompt_sha256', 'ignore_eos')}
        if isinstance(row.get('token_ids'), list):
            values['token_ids_length'] = len(row['token_ids'])
        for key, value in values.items():
            if value is not None and value not in settings[fixture][key]:
                settings[fixture][key].append(value)
        eligible.append((label, batch, row))
    conflicts = {fixture: {key: values for key, values in fields.items() if len(values) > 1}
                 for fixture, fields in settings.items()}
    conflicts = {fixture: fields for fixture, fields in conflicts.items() if fields}
    for label, batch, row in eligible:
        fixture = row.get('fixture')
        if fixture in conflicts:
            rejected.append({'phase': label, 'fixture': fixture, 'rep': batch.get('rep'),
                             'reason': 'inconsistent fixture settings', 'fields': list(conflicts[fixture]), **lineage(batch)})
            continue
        grouped[(fixture, label, bool(row.get('draft')))].append(row['decode_tps'])
    contexts = {row['fixture']: row.get('context') for _, _, _, row in requests(receipt)}
    fixtures = []
    summaries = defaultdict(list)
    for fixture in sorted({key[0] for key in grouped}):
        phases = sorted({label for name, label, _ in grouped if name == fixture})
        distributions = {label: {'speculative': stats(grouped[(fixture, label, True)]),
                                  'plain_control': stats(grouped[(fixture, label, False)])} for label in phases}
        a, b = baselines
        medians = [distributions.get(label, {}).get('speculative', {}).get('median') for label in baselines]
        plain = [distributions.get(label, {}).get('plain_control', {}).get('median') for label in baselines]
        bracket = statistics.mean(medians) if all(v is not None for v in medians) else None
        plain_bracket = statistics.mean(plain) if all(v is not None for v in plain) else None
        drift = 100 * (medians[1] / medians[0] - 1) if bracket else None
        comparisons = []
        for label in phases:
            if label in baselines:
                continue
            candidate = distributions[label]['speculative']['median']
            control = distributions[label]['plain_control']['median']
            item = {'phase': label, 'candidate_tps': candidate, 'baseline_tps': bracket,
                    'speculative_gain_pct': 100 * (candidate / bracket - 1) if candidate and bracket else None,
                    'plain_candidate_tps': control, 'plain_baseline_tps': plain_bracket,
                    'plain_gain_pct': 100 * (control / plain_bracket - 1) if control and plain_bracket else None,
                    'target_tps': target, 'target_met': candidate >= target if candidate is not None else None,
                    'additional_tps_to_target': max(0, target - candidate) if candidate is not None else None,
                    'additional_speedup_pct_to_target': 100 * max(0, target / candidate - 1) if candidate else None,
                    'latency_reduction_pct_to_target': 100 * max(0, 1 - candidate / target) if candidate else None}
            comparisons.append(item)
            if bracket and candidate:
                summaries[(contexts[fixture], label)].append((fixture, item))
        fixtures.append({'fixture': fixture, 'context': contexts[fixture], 'phases': distributions,
                         'baseline_bracket_tps': bracket, 'baseline_drift_pct': drift, 'comparisons': comparisons})
    aggregate = []
    for (context, phase), entries in sorted(summaries.items()):
        baseline = geomean(x['baseline_tps'] for _, x in entries)
        candidate = geomean(x['candidate_tps'] for _, x in entries)
        controls = [x for _, x in entries if x['plain_baseline_tps'] and x['plain_candidate_tps']]
        aggregate.append({'context': context, 'phase': phase, 'fixtures': [name for name, _ in entries],
                          'baseline_tps': baseline, 'candidate_tps': candidate,
                          'speculative_gain_pct': 100 * (candidate / baseline - 1),
                          'plain_gain_pct': 100 * (geomean(
                              x['plain_candidate_tps'] / x['plain_baseline_tps'] for x in controls) - 1) if controls else None,
                          'plain_fixture_count': len(controls), 'target_met': candidate >= target,
                          'additional_tps_to_target': max(0, target - candidate),
                          'additional_speedup_pct_to_target': 100 * max(0, target / candidate - 1),
                          'latency_reduction_pct_to_target': 100 * max(0, 1 - candidate / target)})
    return {'baselines': baselines, 'bracket_method': 'arithmetic mean of A1/A2 per-fixture median TPS',
            'context_aggregation': 'equal-weight geometric mean across per-fixture median TPS',
            'plain_scope': 'diagnostic controls, not statistically independent performance claims',
            'profiled_requests_excluded': True, 'rejected_requests': rejected,
            'settings_conflicts': conflicts,
            'settings_scope': 'available token lengths, requested lengths, sampling and prompt identity across each fixture',
            'fixtures': fixtures, 'contexts': aggregate}


def trace_index(phase, label, directory, issues):
    files = [x for x in phase.get('counter_files', []) if x.get('rank') == 0
             and x.get('name', '').removesuffix('.gz') == 'spec-trace.jsonl']
    if len(files) != 1:
        if phase.get('mode') == 'profile':
            issues.append(f'{label}: trace join requires exactly one rank0 spec-trace file')
        return {}, {'available': False}
    item = files[0]
    try:
        path = confined_input(directory, item['file'])
    except ValueError:
        issues.append(f'{label}: unsafe rank0 trace path')
        return {}, {'available': False}
    if not path.is_file():
        issues.append(f'{label}: missing or unsafe rank0 trace path')
        return {}, {'available': False}
    try:
        raw = input_bytes(path)
    except (OSError, EOFError, zlib.error):
        issues.append(f'{label}: unreadable rank0 trace')
        return {}, {'available': False}
    digest = hashlib.sha256(raw).hexdigest()
    if item.get('sha256') and digest != item['sha256']:
        issues.append(f'{label}: rank0 trace digest mismatch')
        return {}, {'available': False}
    activations = phase.get('activation', [])
    env = activations[0].get('activation_env', {}) if activations else {}
    try:
        low = int(env.get('TF_TRAFFIC_MIN_CONTEXT', 0))
        high = int(env.get('TF_TRAFFIC_MAX_CONTEXT', 2**64 - 1))
        max_rows = int(env.get('TF_TRAFFIC_MAX_ROWS', 16))
    except ValueError:
        issues.append(f'{label}: invalid collector eligibility settings for trace join')
        return {}, {'available': False}
    eligibility_names = ('TF_TRAFFIC_MIN_CONTEXT', 'TF_TRAFFIC_MAX_CONTEXT', 'TF_TRAFFIC_MAX_ROWS')
    defaults = ('0', str(2**64 - 1), '16')
    for node in activations[1:]:
        other = node.get('activation_env', {})
        if any(int(env.get(k, d)) != int(other.get(k, d)) for k, d in zip(eligibility_names, defaults)):
            issues.append(f'{label}: rank eligibility differs; primary trace cannot index all ranks')
            return {}, {'available': False}
    index, counts = {}, Counter()
    last_sequence = -1
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f'{label}: malformed trace line {number}; trace join discarded')
            return {}, {'available': False}
        if event.get('event') != 'verify':
            continue
        if type(event.get('sequence')) is not int or event['sequence'] <= last_sequence:
            issues.append(f'{label}: duplicate or unordered verify trace; trace join discarded')
            return {}, {'available': False}
        last_sequence = event['sequence']
        windows = event.get('windows', [])
        if not windows or any(type(w.get('rows')) is not int or w['rows'] < 1
                              or type(w.get('context')) is not int
                              or w.get('proposed_context_end') != w['context'] + w['rows'] for w in windows):
            issues.append(f'{label}: malformed trace window; trace join discarded')
            return {}, {'available': False}
        context = max(w['proposed_context_end'] for w in windows) - 1
        rows = sum(w['rows'] for w in windows)
        if event.get('max_context_end') != context + 1 or event.get('rows') != rows:
            issues.append(f'{label}: inconsistent trace context/rows; trace join discarded')
            return {}, {'available': False}
        if not (1 <= rows <= max_rows and low <= context <= high):
            continue
        b = bucket(context)
        index[(b, counts[b])] = event
        counts[b] += 1
    return index, {'available': True, 'sha256': digest, 'eligible_verify_counts_by_bucket': dict(counts),
                   'context_formula': 'max(window.proposed_context_end) - 1, matching maximum input position',
                   'indexing': 'zero-based eligible verification index per three collector context buckets'}


def join_trace(record, index, label, issues):
    match = re.fullmatch(r'rank(\d+)_b([012])_round(\d+)', record.get('range', ''))
    if match is None:
        issues.append(f'{label}: unsupported counter range identity for trace join')
        return None
    rank, b, ordinal = map(int, match.groups())
    event = index.get((b, ordinal))
    if (rank != record.get('rank') or b != bucket(record['context_tokens']) or event is None
            or event['rows'] != record['rows'] or event['max_context_end'] - 1 != record['context_tokens']):
        issues.append(f'{label}/{record["range"]}: exact trace rows/context join failed')
        return None
    windows = event['windows']
    if any(type(w.get('potential_accepted_drafts')) is not int
           or not 0 <= w['potential_accepted_drafts'] <= w['rows'] - 1
           or w.get('selected_drafts') != w['rows'] - 1 for w in windows):
        issues.append(f'{label}/{record["range"]}: invalid trace acceptance')
        return None
    accepted = sum(w['potential_accepted_drafts'] for w in windows)
    result = {'range': record['range'], 'sequence': event['sequence'], 'wire_step': event.get('wire_step'),
              'streams': len(windows), 'potential_accepted_drafts': accepted,
              'estimated_potential_emitted_tokens': len(windows) + accepted,
              'contains_first_round': any(w.get('first_round', False) for w in windows),
              'contains_bucket_change': any(w.get('bucket_changed', False) for w in windows)}
    for source, target in (('verify_ns', 'profiled_cpu_verify_wall_ms'), ('total_ns', 'profiled_cpu_round_wall_ms')):
        if finite(event.get(source)):
            result[target] = event[source] / 1e6
    return result


def validated_rounds(records, phase, label, rank, issues):
    env = next((node.get('activation_env', {}) for node in phase.get('activation', [])
                if node.get('rank') == rank), {})
    try:
        limit = int(env.get('TF_TRAFFIC_RANGES', 8))
        skip = int(env.get('TF_TRAFFIC_SKIP', 2))
        if not 1 <= limit <= 1024 or skip < 0:
            raise ValueError
    except (ValueError, TypeError):
        issues.append(f'{label}/rank{rank}: invalid collector range/skip limits')
        return []
    accepted, pending, counts = [], [], Counter()
    for record in records:
        if record['type'] == 'traffic_round':
            pending.append(record)
            continue
        b = record.get('bucket')
        size = record.get('ranges')
        valid = (type(b) is int and b in (0, 1, 2) and type(size) is int
                 and 0 < size == len(pending) and counts[b] + size <= limit)
        ordinals = []
        for sample in pending:
            match = re.fullmatch(r'rank(\d+)_b([012])_round(\d+)', sample.get('range', ''))
            context = sample.get('context_tokens')
            if (match is None or type(context) is not int or context < 0 or sample.get('rank') != rank
                    or int(match[1]) != rank or int(match[2]) != b or bucket(context) != b):
                valid = False
            elif match is not None:
                ordinals.append(int(match[3]))
        if valid and sorted(ordinals) != list(range(skip + counts[b], skip + counts[b] + size)):
            valid = False
        if valid:
            accepted.extend(pending)
            counts[b] += size
        else:
            issues.append(f'{label}/rank{rank}/bucket{b}: incomplete or inconsistent counter window')
        pending = []
    if pending:
        issues.append(f'{label}/rank{rank}: unterminated counter window')
    return accepted


def traffic(receipt, directory, issues, join_spec_trace=False):
    groups = defaultdict(list)
    configurations = []
    metadata = {}
    seen = set()
    sources = []
    trace_sources = []
    for label, phase in receipt.get('phases', {}).items():
        index = {}
        if join_spec_trace:
            index, description = trace_index(phase, label, directory, issues)
            trace_sources.append({'phase': label, **description})
        context_map = defaultdict(set)
        for _, _, _, row in requests({'phases': {label: phase}}):
            if type(row.get('context')) is int:
                context_map[bucket(row['context'])].add(row['context'])
        phase_records = defaultdict(list)
        for item in phase.get('counter_files', []):
            if not item.get('name', Path(item.get('file', '')).name).startswith('traffic'):
                continue
            relative = str(item['file']).replace('\\', '/')
            try:
                path = confined_input(directory, relative)
            except ValueError as error:
                issues.append(f'{label}: {error}')
                continue
            if not path.is_file():
                issues.append(f'{label}: missing counter file {Path(relative).name}')
                continue
            try:
                raw = input_bytes(path)
            except (OSError, EOFError, zlib.error):
                issues.append(f'{label}: unreadable counter file {path.name}')
                continue
            digest = hashlib.sha256(raw).hexdigest()
            if item.get('sha256') and digest != item['sha256']:
                issues.append(f'{label}: counter digest mismatch {path.name}')
                continue
            sources.append({'phase': label, 'file': relative, 'sha256': digest, 'rank': item.get('rank')})
            for line_number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    issues.append(f'{label}/{path.name}:{line_number}: malformed JSON')
                    continue
                kind = record.get('type')
                if kind == 'metric_metadata':
                    metric = record['metric']
                    metadata[metric['name']] = metric
                elif kind == 'configuration':
                    configurations.append({'phase': label, **record})
                    if record.get('passes') != 1:
                        issues.append(f'{label}/{path.name}: counter configuration is not single-pass')
                    if item.get('rank') is not None and record.get('rank') != item['rank']:
                        issues.append(f'{label}/{path.name}: configuration rank mismatch')
                elif kind == 'error':
                    issues.append(f'{label}/{path.name}: collector error {record}')
                elif kind in ('traffic_round', 'window_complete'):
                    phase_records[item.get('rank', record.get('rank'))].append(record)
        for rank, records in phase_records.items():
            rounds = validated_rounds(records, phase, label, rank, issues)
            for b in {bucket(r['context_tokens']) for r in rounds}:
                if len(context_map[b]) > 1:
                    issues.append(f'{label}/bucket{b}: multiple requested contexts cannot be attributed uniquely')
            for record in rounds:
                identity = (label, rank, record.get('range'))
                if identity in seen:
                    issues.append(f'{label}/rank{rank}: duplicate range {record.get("range")}')
                    continue
                seen.add(identity)
                if record.get('rank') != rank or type(record.get('rows')) is not int or record['rows'] < 1:
                    issues.append(f'{label}/rank{rank}: malformed round identity')
                    continue
                values = record.get('values', {})
                if not all(finite(values.get(name)) for name in METRICS) or not finite(values.get(TIME), True):
                    issues.append(f'{label}/rank{rank}/{record.get("range")}: missing or invalid metric')
                    continue
                b = bucket(record['context_tokens'])
                context = next(iter(context_map[b])) if len(context_map[b]) == 1 else None
                sample = {name: values[metric] * scale for metric, (name, scale) in METRICS.items()}
                sample['sysmem_fill_inferred_GBps'] = values[FILL] * 32 / values[TIME]
                sample['context_tokens'] = record['context_tokens']
                if index:
                    joined = join_trace(record, index, label, issues)
                    if joined is not None:
                        sample['trace'] = joined
                        sample['sysmem_fill_bytes_per_potential_emitted_token'] = (
                            values[FILL] * 32 / joined['estimated_potential_emitted_tokens'])
                groups[(label, b, context, rank, record['rows'])].append(sample)
    output = []
    for (label, b, context, rank, rows), samples in sorted(groups.items()):
        summary = {'phase': label, 'bucket': b, 'context': context, 'rank': rank, 'rows': rows,
                   'count': len(samples), 'observed_context_min': min(x['context_tokens'] for x in samples),
                   'observed_context_max': max(x['context_tokens'] for x in samples)}
        for key in ('sysmem_fill_GB', 'l2_sysmem_read_request_GB', 'l2_sysmem_write_request_GB',
                    'gpu_range_ms', 'sysmem_fill_inferred_GBps'):
            summary[key] = stats([x[key] for x in samples])
        summary['sysmem_fill_inferred_weighted_GBps'] = (
            sum(x['sysmem_fill_GB'] for x in samples) / sum(x['gpu_range_ms'] for x in samples) * 1000)
        if join_spec_trace:
            matched = [x for x in samples if 'trace' in x]
            summary['trace_join'] = {'matched_count': len(matched), 'unmatched_count': len(samples) - len(matched),
                                     'rounds': [x['trace'] for x in matched]}
            for name in ('potential_accepted_drafts', 'estimated_potential_emitted_tokens'):
                summary['trace_join'][name] = stats([x['trace'][name] for x in matched])
            summary['trace_join']['sysmem_fill_bytes_per_potential_emitted_token'] = stats([
                x['sysmem_fill_bytes_per_potential_emitted_token'] for x in matched])
            summary['trace_join']['sysmem_fill_bytes_per_potential_emitted_token_weighted'] = (
                sum(x['sysmem_fill_GB'] for x in matched) * 1e9
                / sum(x['trace']['estimated_potential_emitted_tokens'] for x in matched)) if matched else None
        output.append(summary)
    return {'sector_bytes': 32, 'GB_definition': 'decimal 1e9 bytes', 'groups': output,
            'configurations': configurations, 'metric_metadata': metadata, 'sources': sources,
            'trace_join_enabled': join_spec_trace, 'trace_sources': trace_sources,
            'yield_semantics': 'Potential new emissions are 1 + accepted prefix per window, before stop/output budgets; '
                               'the pending input was already emitted and is not counted again. First prefill output '
                               'is outside this range. This is not observed committed output count.',
            'bytes_per_yield_scope': 'Target forward + DSpark tap absorption only; excludes full MTP draft passes, '
                                     'argmax/readback and prefill. Profiled CPU wall times are diagnostics, not GPU time '
                                     'or unprofiled speed. Ratios are per Spark and must not be summed into its bandwidth.',
            'semantics': {
                'sysmem_fill_GB': 'L2 sectors filled from system memory, counted at 32 bytes per sector',
                'l2_sysmem_read_request_GB': 'L2 sysmem-aperture read request sectors; can include L2 hits',
                'l2_sysmem_write_request_GB': 'L2 sysmem-aperture write request sectors; not DRAM writeback bytes',
                'gpu_range_ms': 'GPU duration metric for the instrumented verification range',
                'sysmem_fill_inferred_GBps': 'system-memory L2 fills divided by GPU range duration; not DRAM-total bandwidth'},
            'limitations': ['The three byte counters overlap in scope and must not be added together.',
                            'L2 request-sector counts are not total physical DRAM reads or writes.',
                            'Ranges are instrumented; use separate unprofiled requests for speed claims.',
                            'Do not add both ranks and divide by one rank bandwidth when forming a roofline.',
                            'Grouping by row width does not control the union of selected MoE experts.']}


def analyze_one(receipt_path, reference_path=None, baselines=('A1', 'A2'), target=80, join_spec_trace=False):
    receipt = read_json(receipt_path)
    reference = read_json(reference_path) if reference_path else None
    issues = []
    counters = traffic(receipt, receipt_path.parent, issues, join_spec_trace)
    weights = {'active_weight_bytes_per_rank_one_token': 4334554816,
               'fixed_weight_bytes_per_rank': 2799353536, 'six_routed_experts_bytes_per_rank': 1535201280,
               'decimal_GB_per_rank': 4.334554816, 'layers': 40, 'selected_routed_experts_per_token_per_layer': 6,
               'available_routed_experts_per_layer': 384,
               'meaning': 'Analytical once-through active weight bytes for one plain token, not measured total traffic.',
               'tensor_parallelism': 'Each Spark holds half of every expert intermediate; all six selected experts run on both ranks.',
               'verification_model': 'fixed bytes + sum_over_layers(unique selected expert IDs * that layer expert-slice bytes)',
               'excludes': ['KV/index scans', 'activation traffic', 'extra reads and cache effects', 'drafter passes', 'communication'],
               'not': ['total resident model size', 'per-output-token measured traffic', 'a verification-round constant']}
    assert weights['fixed_weight_bytes_per_rank'] + weights['six_routed_experts_bytes_per_rank'] == 4334554816
    return {'schema_version': 1, 'receipt_sha256': hashlib.sha256(input_bytes(receipt_path)).hexdigest(),
            'target_tps': target, 'original_service_restored': receipt.get('restored'),
            'receipt_failure': receipt.get('failure', receipt.get('error')), 'issues': issues,
            'speed': speed(receipt, baselines, target), 'exactness': accuracy(receipt, reference),
            'traffic': counters, 'weight_model': weights}


def analyze(receipt_path, reference_path=None, baselines=('A1', 'A2'), target=80, join_spec_trace=False,
            continuation_paths=(), phase_aliases=()):
    if not continuation_paths and not phase_aliases:
        return analyze_one(receipt_path, reference_path, baselines, target, join_spec_trace)
    from continuation import compose
    paths = [receipt_path, *continuation_paths]
    receipts = [read_json(path) for path in paths]
    analyses = [analyze_one(path, reference_path, baselines, target, join_spec_trace) for path in paths]
    receipt, result = compose(paths, receipts, analyses, phase_aliases, baselines, speed_eligible)
    result['speed'] = speed(receipt, baselines, target)
    result['speed']['limitations'] = result['composition']['limitations']
    result['exactness'] = accuracy(receipt, read_json(reference_path) if reference_path else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--reference-receipt', type=Path)
    parser.add_argument('--baseline-labels', nargs=2, default=('A1', 'A2'))
    parser.add_argument('--target-tps', type=float, default=80)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--join-spec-trace', action='store_true')
    parser.add_argument('--continuation-receipt', type=Path, action='append', default=[])
    parser.add_argument('--phase-alias', action='append', default=[], metavar='SOURCE=TARGET')
    args = parser.parse_args()
    if not finite(args.target_tps, True):
        parser.error('--target-tps must be finite and positive')
    try:
        result = analyze(args.receipt.resolve(), args.reference_receipt, args.baseline_labels, args.target_tps,
                         args.join_spec_trace, args.continuation_receipt, args.phase_alias)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or args.receipt.with_name('traffic-analysis.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'traffic_groups': len(result['traffic']['groups']),
                      'speed_contexts': result['speed']['contexts'], 'exactness': result['exactness']['counts'],
                      'issues': result['issues'], 'latest_input_restored': result['original_service_restored'],
                      'receipt_failure': result['receipt_failure'], 'inputs': result.get('inputs')}, indent=2))


if __name__ == '__main__':
    main()
