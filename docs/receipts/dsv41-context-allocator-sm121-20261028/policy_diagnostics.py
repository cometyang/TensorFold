"""Compare measured speculative decisions through attributed per-request counters."""
from pathlib import Path
import json
import statistics

root = Path(__file__).resolve().parent
receipt = json.loads((root/'collected/receipt.json').read_text())
out = {}
for fixture in receipt['fixtures']:
    if fixture['split'] != 'holdout':
        continue
    phases = {}
    for label in ('A1', 'B', 'A2'):
        selected = [b for b in receipt['phases'][label]['requests'] if b['runs'][0]['fixture'] == fixture['name']
                    and b['runs'][0]['draft'] and not b['warmup']]
        assert len(selected) == 3 and all(b['metrics']['valid'] for b in selected)
        deltas = [b['metrics']['delta'] for b in selected]
        def median(fn):
            return statistics.median(fn(d) for d in deltas)
        phases[label] = {
            'decode_rounds': median(lambda d: d['decode_rounds_total']),
            'generated_tokens_per_round': median(lambda d: d['generation_tokens_total']/d['decode_rounds_total']),
            'post_first_token_outputs_per_round': median(lambda d: (d['generation_tokens_total']-1)/d['decode_rounds_total']),
            'verified_drafts_per_round': median(lambda d: d['mtp_drafted_total']/d['decode_rounds_total']),
            'accepted_drafts_per_round': median(lambda d: d['mtp_accepted_total']/d['decode_rounds_total']),
            'accepted_drafts_fraction': median(lambda d: d['mtp_accepted_total']/d['mtp_drafted_total']
                                               if d['mtp_drafted_total'] else 0),
        }
        plain = [b for b in receipt['phases'][label]['requests'] if b['runs'][0]['fixture'] == fixture['name']
                 and not b['runs'][0]['draft']]
        assert len(plain) == 1 and plain[0]['metrics']['valid']
        phases[label]['plain_decode_tps'] = plain[0]['runs'][0]['decode_tps']
        phases[label]['plain_server_request_completion_tps'] = plain[0]['server_decode_tps_per_stream']
    out[fixture['name']] = {'context': fixture['context'], 'kind': fixture['kind'], 'median_counters': phases}
(root/'policy-diagnostics.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
