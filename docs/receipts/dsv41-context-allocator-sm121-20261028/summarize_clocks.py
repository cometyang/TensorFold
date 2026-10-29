"""Summarize both devices' existing clock telemetry, without querying hardware."""
from pathlib import Path
from datetime import datetime
import json
import statistics
import argparse

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=root/'collected/worker/clocks.jsonl')
parser.add_argument('--output', type=Path, default=root/'clock-summary.json')
args = parser.parse_args()
records = [json.loads(line) for line in args.input.read_text().splitlines()]
samples = [record for record in records if 'devices' in record]
times = [datetime.fromisoformat(record['utc']) for record in samples]
summary = {'ceiling_mhz': 2200, 'cap_acknowledged_both': any(
    item.get('event') == 'cap_acknowledged_on_both_devices' and item.get('requested_max_mhz') == 2200
    for item in records), 'paired_samples': len(samples), 'first_utc': samples[0]['utc'],
    'last_utc': samples[-1]['utc'], 'max_sampling_gap_s': max(
        (after - before).total_seconds() for before, after in zip(times, times[1:])), 'ranks': {}}
for rank in ('rank0', 'rank1'):
    rows = [device for sample in samples for device in sample['devices'] if device['rank'] == rank]
    good = [row for row in rows if all(isinstance(row.get(key), (int, float))
                                      for key in ('graphics_mhz', 'sm_mhz'))]
    active = [row for row in good if row.get('utilization_pct', 0) >= 90]
    summary['ranks'][rank] = {'samples': len(rows), 'valid_clock_samples': len(good),
        'max_graphics_mhz': max(row['graphics_mhz'] for row in good),
        'max_sm_mhz': max(row['sm_mhz'] for row in good),
        'violations': sum(max(row['graphics_mhz'], row['sm_mhz']) > 2200 for row in good),
        'active_sample_median_sm_mhz': statistics.median(row['sm_mhz'] for row in active) if active else None,
        'max_temperature_c': max(row['temperature_c'] for row in rows if isinstance(row.get('temperature_c'), (int, float))),
        'max_power_w': max(row['power_w'] for row in rows if isinstance(row.get('power_w'), (int, float)))}
summary['passed'] = (summary['cap_acknowledged_both'] and all(
    row['samples'] == row['valid_clock_samples'] == len(samples) and row['violations'] == 0
    for row in summary['ranks'].values()))
args.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
