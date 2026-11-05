import csv, collections, json
import numpy as np
from pathlib import Path
from tokenomics.train import producer_es as pe
from tokenomics.train.order_es import a0_values, noise, rank_and_estimate, cosine
D = Path('/srv/research/tokenomics/experiments/EXP-0081-producer-probe/probe')
S = Path('/srv/research/wt-exp0081/tokenomics/_pod_ckpts/exp0081')
bar, base_rows, _ = pe.load_bar(Path('/srv/research/tokenomics/experiments/EXP-0080-producer-bar-noise/bar')); margin, _ = pe.load_margin(Path('/srv/research/tokenomics/experiments/EXP-0080-producer-bar-noise/noise_floor/margins.json'))
refs = {k: v[0] for k, v in a0_values(base_rows).items()}
rows = list(csv.DictReader(open(D / 'probe_episodes.csv')))
eps = noise(0, 0, pe.PAIRS, pe.THETA_DIM, pe.NOISE_STREAM_PURPOSE["probe"])
spec = json.load(open(D/'probe.json'))
out = {}
for arm, scale in pe.SIGMA_ARMS.items():
    sig = np.full(pe.THETA_DIM, pe.SIGMA * scale); est = {}
    for b in ('0','1'):
        by = collections.defaultdict(list)
        for r in rows:
            if r['batch']==b and r['member'].startswith(arm+':'): by[r['member'].split(':',1)[1]].append(r)
        sc = pe._scores(by, bar, margin, refs)
        for mode in ('on','off'):
            dq = {k: (v.disqualified if mode=='on' else False) for k,v in sc.items()}
            est[(b,mode)] = rank_and_estimate(eps, sig, {k:v.fitness for k,v in sc.items()}, dq).estimate
        # excluding the noise-flagged world-episodes (rf3 re-keys flag) from the rule:
    for mode in ('on','off'):
        e0, e1 = est[('0',mode)], est[('1',mode)]
        out[(arm,mode)] = (cosine(e0,e1), float(np.linalg.norm(e0)), float(np.linalg.norm(e1)))
for k,v in out.items(): print(k, v)
print('recorded', {k: v['cosine'] for k,v in spec['arms'].items()})
