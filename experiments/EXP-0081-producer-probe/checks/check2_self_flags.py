import csv, collections
from pathlib import Path
from tokenomics.train import producer_es as pe
from tokenomics.train.order_es import a0_values, disqualifying
D = Path('/srv/research/tokenomics/experiments/EXP-0081-producer-probe/probe')
S = Path('/srv/research/wt-exp0081/tokenomics/_pod_ckpts/exp0081')
N = Path('/srv/research/tokenomics/experiments/EXP-0080-producer-bar-noise/noise_floor/noise_episodes.csv')
bar, base_rows, _ = pe.load_bar(Path('/srv/research/tokenomics/experiments/EXP-0080-producer-bar-noise/bar')); margin, _ = pe.load_margin(Path('/srv/research/tokenomics/experiments/EXP-0080-producer-bar-noise/noise_floor/margins.json'))
refs = {k: v[0] for k, v in a0_values(base_rows).items()}
def kinds(r):
    k=(r['world'],int(r['episode'])); return sorted({x.split('(')[0].split(':')[0] for x in disqualifying(r, bar[k], margin, refs[k])})
probe = list(csv.DictReader(open(D/'probe_episodes.csv')))
noise = [r for r in csv.DictReader(open(N)) if r['replicate']!='0']
print('bar deaths', [(k, v.a0_units) for k, v in bar.items() if v.a0_dead])
for b in ('0','1'):
    print('== batch', b)
    wes = sorted({(r['world'], r['episode']) for r in probe if r['batch']==b})
    tot=collections.Counter()
    for w,e in wes:
        pr=[r for r in probe if r['batch']==b and (r['world'],r['episode'])==(w,e)]
        nr=[r for r in noise if (r['world'],r['episode'])==(w,e)]
        pf=collections.Counter(k for r in pr for k in kinds(r)); nf=collections.Counter(k for r in nr for k in kinds(r))
        pfl=sum(bool(kinds(r)) for r in pr); nfl=sum(bool(kinds(r)) for r in nr)
        tot['probe']+=pfl; tot['probe_n']+=len(pr); tot['noise']+=nfl; tot['noise_n']+=len(nr)
        if pfl or nfl: print(w[-7:], e, f'members flagged {pfl}/{len(pr)} {dict(pf)} | rf3 rekeys flagged {nfl}/{len(nr)} {dict(nf)} | dead members {sum(r["dead"]=="true" for r in pr)} rekeys {sum(r["dead"]=="true" for r in nr)}')
    print(dict(tot))
print('== rf3 re-keys as members (any flag over the batch => disqualified)')
for b in ('0','1'):
    wes = {(r['world'], r['episode']) for r in probe if r['batch']==b}
    for rep in ('1','2','3','4'):
        nr=[r for r in noise if (r['world'],r['episode']) in wes and r['replicate']==rep]
        print(b, rep, len(nr), 'disqualified' if any(kinds(r) for r in nr) else 'clean', [ (r['world'][-4:], kinds(r)) for r in nr if kinds(r)])
print('== 0016 deaths by member')
d=[r['member'] for r in probe if r['world'].endswith('12-0016') and r['batch']=='1' and r['dead']=='true']
print(sorted(d))
d=[r['member'] for r in probe if r['world'].endswith('12-0004') and r['batch']=='1' and r['dead']=='true']
print(sorted(d))
