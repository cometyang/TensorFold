"""Compose explicitly related request evidence while retaining each receipt's outcome and counter attribution."""
from collections import defaultdict
import copy


def aliases_from(arguments, present):
    aliases = {}
    for argument in arguments:
        source, separator, target = argument.partition('=')
        source, target = source.strip(), target.strip()
        if not separator or not source or not target or source in aliases:
            raise ValueError('phase aliases must be unique SOURCE=TARGET entries')
        aliases[source] = target
    if set(aliases) - present:
        raise ValueError('phase alias source is absent from the supplied receipts')
    if any(target in aliases and target != source for source, target in aliases.items()):
        raise ValueError('chained phase aliases are not supported')
    return aliases


def measured_keys(phase, eligible):
    return {(row.get('fixture'), bool(row.get('draft')), batch.get('rep'))
            for batch in phase.get('requests', []) if not batch.get('warmup', False)
            and phase.get('mode', batch.get('mode', 'speed')) == 'speed'
            for row in batch.get('runs', []) if eligible(batch, row)}


def compose(paths, receipts, analyses, arguments, baselines, eligible):
    aliases = aliases_from(arguments, {label for receipt in receipts for label in receipt.get('phases', {})})
    phases, lineage, origins = {}, [], defaultdict(set)
    occupied, source_labels = defaultdict(set), defaultdict(set)
    inputs, digests = [], set()
    for index, (path, receipt, analysis) in enumerate(zip(paths, receipts, analyses)):
        digest = analysis['receipt_sha256']
        if digest in digests:
            raise ValueError('duplicate receipt content would count the same evidence twice')
        digests.add(digest)
        source = {'input_index': index, 'receipt_basename': path.name, 'decompressed_sha256': digest,
                  'started_utc': receipt.get('started_utc'), 'ended_utc': receipt.get('ended_utc'),
                  'restored': receipt.get('restored'), 'failure': analysis['receipt_failure'],
                  'exactness_passed': receipt.get('exactness_passed'), 'phases': []}
        planned = [phase['label'] for phase in receipt.get('plan', {}).get('phases', [])]
        source['planned_phases'] = planned
        source['unstarted_phases'] = [label for label in planned if label not in receipt.get('phases', {})]
        for label, phase in receipt.get('phases', {}).items():
            target = aliases.get(label, label)
            mode = phase.get('mode', 'speed')
            keys = measured_keys(phase, eligible)
            if target in phases:
                explicit = label in aliases or any(prior in aliases for prior in source_labels[target])
                if not explicit:
                    raise ValueError(f'duplicate phase {target}; an explicit phase alias is required')
                if phases[target]['mode'] != mode:
                    raise ValueError(f'phase alias mixes modes for {target}')
                if occupied[target] & keys:
                    raise ValueError(f'phase alias overlaps valid measurements for {target}')
            else:
                phases[target] = {'mode': mode, 'requests': []}
            occupied[target].update(keys)
            source_labels[target].add(label)
            origins[target].add(index)
            description = {'source_input': index, 'source_phase': label, 'analysis_phase': target,
                           'complete': phase.get('complete'), 'started_utc': phase.get('started_utc'),
                           'ended_utc': phase.get('ended_utc'), 'request_batches': len(phase.get('requests', []))}
            lineage.append(description)
            source['phases'].append(copy.deepcopy(description))
            for ordinal, batch in enumerate(phase.get('requests', [])):
                entry = copy.deepcopy(batch)
                entry.update(source_input=index, source_phase=label, source_batch=ordinal)
                phases[target]['requests'].append(entry)
        inputs.append(source)
    result = copy.deepcopy(analyses[0])
    result['inputs'] = inputs
    failures = [{'input_index': item['input_index'], 'failure': item['failure']} for item in inputs if item['failure']]
    result['receipt_failure'] = failures or None
    result['original_service_restored'] = inputs[-1]['restored']
    result['issues'] = [f'input{index}: {issue}' for index, analysis in enumerate(analyses) for issue in analysis['issues']]
    spanning = bool(origins[baselines[0]] and origins[baselines[1]]
                    and len(origins[baselines[0]] | origins[baselines[1]]) > 1)
    limitations = ['This is composed request evidence, not a single completed experiment; raw input outcomes are retained.',
                   'Explicit phase aliases identify intended equivalent settings; they do not prove identical runtime conditions.']
    if spanning:
        limitations.append('The baseline bracket spans input runs and may span service restoration/relaunch; drift is not isolated.')
    result['composition'] = {'phase_aliases': aliases, 'phase_lineage': lineage,
                             'baseline_spans_inputs': spanning, 'any_input_failed': bool(failures),
                             'all_inputs_ended': all(item['ended_utc'] is not None for item in inputs),
                             'receipt_sha256_scope': 'primary input; all decompressed input hashes are listed in inputs',
                             'restoration_scope': 'original_service_restored describes the latest supplied input only',
                             'limitations': limitations}
    for field in ('groups', 'configurations', 'sources', 'trace_sources'):
        result['traffic'][field] = []
        for index, analysis in enumerate(analyses):
            for item in analysis['traffic'][field]:
                result['traffic'][field].append({**copy.deepcopy(item), 'source_input': index,
                                                 'source_phase': item['phase'],
                                                 'phase': aliases.get(item['phase'], item['phase'])})
    result['traffic']['aggregation_scope'] = 'Counter groups remain separate by input; relative files belong to their source receipt.'
    for index, analysis in enumerate(analyses[1:], 1):
        for name, metadata in analysis['traffic']['metric_metadata'].items():
            prior = result['traffic']['metric_metadata'].get(name)
            if prior is not None and prior != metadata:
                result['issues'].append(f'input{index}: metric metadata differs for {name}')
            else:
                result['traffic']['metric_metadata'][name] = copy.deepcopy(metadata)
    return {'phases': phases}, result
