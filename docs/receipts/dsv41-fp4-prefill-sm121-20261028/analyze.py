"""Summarize the A/B/A receipt without confusing kernel speed with whole-model speed."""
import collections,json,pathlib,statistics
root=pathlib.Path(__file__).resolve().parent
data=root;r=json.loads((data/'receipt.json').read_text())
summary={'quality_passed':r.get('quality_passed'),'restored':r['restored'],'sanitizers_passed':r.get('sanitizers_passed'),'rows':[]}
for count in r['contexts']:
 row={'context':count}
 for kind,field,draft in [('cold_ttft','ttft_s',True),('plain_decode','decode_tps',False),('speculative_decode','decode_tps',True)]:
  values={phase:statistics.median(x[field] for x in p['requests'] if x['name'].startswith(f'context-{count}-') and x['draft']==draft) for phase,p in r['phases'].items()}
  reference=(values['A1']+values['A2'])/2
  row[kind]={'phases':values,'bracketed_baseline':reference,'change_pct':100*(values['B']/reference-1),'baseline_drift_pct':100*(values['A2']/values['A1']-1)}
  if kind=='cold_ttft':row[kind]['throughput_speedup']=reference/values['B']
 summary['rows'].append(row)
summary['comparisons']={'total':len(r.get('comparisons',[])),'exact':sum(x['same_full_token_ids'] and x['same_output'] for x in r.get('comparisons',[]))}
summary['retrieval_accuracy']=r.get('retrieval_accuracy')
summary['lifecycle']=r.get('lifecycle')
summary['concurrent']={k:len(v.get('concurrent',[])) for k,v in r['phases'].items()}
clock_summaries=[json.loads(p.read_text()) for p in data.glob('*clocks-*.summary.json')]
summary['clocks']=clock_summaries
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',newline='\n')
print(json.dumps({k:v for k,v in summary.items() if k!='lifecycle'},indent=2))
