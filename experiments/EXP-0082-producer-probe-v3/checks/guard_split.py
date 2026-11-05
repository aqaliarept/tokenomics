"""EXP-0082 diagnostic: the v3 probe's cosine with each guard clause on / off (offline, recorded rows)."""
import collections, csv, json
import numpy as np
from tokenomics.train import producer_es as pe
from tokenomics.train.order_es import a0_values, cosine, noise, rank_and_estimate, _bool
E = '/srv/research/tokenomics/experiments/'
bar, base, _ = pe.load_bar(E + 'EXP-0080-producer-bar-noise/bar')
M = E + 'EXP-0080-producer-bar-noise/noise_floor/margins.json'
margin, _ = pe.load_margin(M)
fref, _ = pe.load_flag_reference(M, bar, base, margin)
refs = {k: v[0] for k, v in a0_values(base).items()}
rows = list(csv.DictReader(open(E + 'EXP-0082-producer-probe-v3/probe/probe_episodes.csv')))
eps = noise(0, 0, pe.PAIRS, pe.THETA_DIM, pe.NOISE_STREAM_PURPOSE['probe'])
for arm, scale in pe.SIGMA_ARMS.items():
    sig = np.full(pe.THETA_DIM, pe.SIGMA * scale)
    res = collections.defaultdict(dict)
    for b in ('0', '1'):
        by = collections.defaultdict(list)
        for r in rows:
            if r['batch'] == b and r['member'].startswith(arm + ':'):
                by[r['member'].split(':', 1)[1]].append(r)
        sc = {k: pe.score_member_v3(v, bar, margin, refs, fref) for k, v in by.items()}
        fit = {k: v.fitness for k, v in sc.items()}
        dq = {'v3': {k: v.disqualified for k, v in sc.items()},
              'deaths_only': {k: any(x.endswith(':death') for x in v.reasons) for k, v in sc.items()},
              'flags_only': {k: any(x.startswith('flags:') for x in v.reasons) for k, v in sc.items()},
              'off': {k: False for k in sc}}
        for name, d in dq.items():
            res[name][b] = (rank_and_estimate(eps, sig, fit, d).estimate, sum(d.values()))
        dead = {k: [r['world'][-7:] for r in v if _bool(r['dead'])] for k, v in by.items()}
        print(arm, 'batch', b, 'deaths', {k: v for k, v in dead.items() if v})
    for name, d in res.items():
        print(arm, name, 'cosine', round(cosine(d['0'][0], d['1'][0]), 3), 'disqualified A/B', d['0'][1], d['1'][1])
# units lost by the dying members on 12-0016
u = [float(r['units']) for r in rows if r['world'].endswith('12-0016') and r['batch'] == '1']
du = [float(r['units']) for r in rows if r['world'].endswith('12-0016') and r['batch'] == '1' and _bool(r['dead'])]
k = next(k for k in bar if k[0].endswith('12-0016'))
print('12-0016 bar', bar[k].a0_units, 'members mean', round(np.mean(u), 1), 'dying mean', round(np.mean(du), 1))
