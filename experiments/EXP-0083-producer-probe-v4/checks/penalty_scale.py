"""EXP-0083 diagnostic: the v4 death penalty against the members' fitness spread on minibatch B, and the
probe cosine as a function of the penalty P = p x B_we (offline, recorded rows)."""
import collections, csv
import numpy as np
from tokenomics.train import producer_es as pe
from tokenomics.train.order_es import a0_values, cosine, noise, rank_and_estimate, _bool, we_fitness
E = '/srv/research/tokenomics/experiments/'
bar, base, _ = pe.load_bar(E + 'EXP-0080-producer-bar-noise/bar')
M = E + 'EXP-0080-producer-bar-noise/noise_floor/margins.json'
margin, _ = pe.load_margin(M)
fref, _ = pe.load_flag_reference(M, bar, base, margin)
refs = {k: v[0] for k, v in a0_values(base).items()}
rows = list(csv.DictReader(open(E + 'EXP-0083-producer-probe-v4/probe/probe_episodes.csv')))
eps = noise(0, 0, pe.PAIRS, pe.THETA_DIM, pe.NOISE_STREAM_PURPOSE['probe'])
for arm, scale in pe.SIGMA_ARMS.items():
    sig = np.full(pe.THETA_DIM, pe.SIGMA * scale)
    by = {b: collections.defaultdict(list) for b in ('0', '1')}
    for r in rows:
        if r['member'].startswith(arm + ':'):
            by[r['batch']][r['member'].split(':', 1)[1]].append(r)
    def fit(b, p):
        out, dq = {}, {}
        for m, rs in by[b].items():
            sc = pe.score_member_v4(rs, bar, margin, refs, fref)
            fs = [we_fitness(r, bar[(r['world'], int(r['episode']))], 0.0)
                  - (p if _bool(r['dead']) and (r['world'], int(r['episode'])) not in fref.dead else 0.0)
                  for r in rs]
            out[m], dq[m] = float(np.mean(fs)), sc.disqualified
        return out, dq
    f0, _ = fit('1', 0.0)
    living = [v for m, v in f0.items() if not any(_bool(r['dead']) for r in by['1'][m])]
    print(arm, 'B member fitness SD (p=0)', round(float(np.std(list(f0.values()))), 4),
          'range', round(min(f0.values()), 4), round(max(f0.values()), 4),
          '| one death at p=1 moves the mean by', round(1 / 24, 4))
    for p in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        est = [rank_and_estimate(eps, sig, *fit(b, p)).estimate for b in ('0', '1')]
        print(arm, f'p={p}', 'cosine', round(cosine(*est), 3))
