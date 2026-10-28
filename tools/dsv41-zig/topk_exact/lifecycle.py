"""Check scratch reuse after cancellation and retained-prefix replay on the already-running candidate."""
import http.client
import json
import time
import urllib.request


def check(bench, run_dir, reference):
    expected = {(row['name'], row['draft']): row['token_ids'] for row in reference['requests']}
    result = {'checks': [], 'cancellations': [], 'passed': False}

    def save():
        (run_dir / 'lifecycle.json').write_text(json.dumps(result, indent=2) + '\n')

    def body(name, draft=True, stream=False, tokens=128):
        ids = json.loads((run_dir / (name + '.json')).read_text())
        return {'model': bench.MODEL, 'prompt': ids, 'max_tokens': tokens, 'temperature': 0,
                'seed': 1234, 'ignore_eos': True, 'return_token_ids': True, 'draft': draft, 'stream': stream}

    def complete(name, draft, label):
        bench.idle(90)
        request = urllib.request.Request(bench.URL + '/v1/completions',
            data=json.dumps(body(name, draft)).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=1800) as response:
            reply = json.load(response)
        ids = reply.get('tensorfold', {}).get('token_ids')
        row = {'label': label, 'fixture': name, 'draft': draft, 'token_ids': ids,
               'same_full_token_ids': ids == expected[name, draft], 'usage': reply.get('usage')}
        result['checks'].append(row)
        save()
        assert row['same_full_token_ids'], label

    def cancel(name, decode):
        bench.idle(90)
        request = body(name, stream=True, tokens=1024)
        if not decode:
            request['prompt'][0] = 999
        connection = http.client.HTTPConnection('127.0.0.1', 8888, timeout=90)
        connection.request('POST', '/v1/completions', json.dumps(request), {'Content-Type': 'application/json'})
        seen = False
        if decode:
            response = connection.getresponse()
            assert response.status == 200
            while line := response.readline():
                if line.startswith(b'data:') and line.strip() != b'data: [DONE]':
                    chunk = json.loads(line[5:])
                    if any(item.get('text') for item in chunk.get('choices', [])):
                        seen = True
                        break
            response.close()
        else:
            end = time.monotonic() + 5
            while time.monotonic() < end:
                seen = bench.api('/health').get('live', {}).get('connections', 0) > 0
                if seen:
                    break
                time.sleep(0.05)
            time.sleep(0.3)
        connection.close()
        assert seen
        bench.idle(90)
        result['cancellations'].append({'fixture': name, 'phase': 'decode' if decode else 'prefill',
                                       'admission_observed': seen, 'connection_closed': True})
        save()

    complete('context-2048-rep-0', True, 'before cancellation')
    cancel('context-65536-rep-0', False)
    complete('context-2048-rep-1', False, 'after prefill cancellation')
    cancel('context-2048-rep-2', True)
    complete('context-2048-rep-0', True, 'after decode cancellation')
    complete('context-65536-rep-0', True, 'full history after cancellation')
    complete('context-65536-rep-0', False, 'retained prefix after cancellation')
    result['passed'] = True
    save()
    print(json.dumps({'lifecycle_passed': True, 'checks': len(result['checks'])}), flush=True)
    return result
