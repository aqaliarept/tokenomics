import csv, math, sys, collections
import numpy as np
def load(d):
    W = {r['world']: r for r in csv.DictReader(open(f'{d}/worlds.csv'))}
    U = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f'{d}/artifacts/episodes.csv')):
        s = r['subject']
        if s in ('entity_planner/trained_orderer', 'entity_planner/round_robin'):
            U[r['world']][s].append(float(r['units']))
    rows = []
    for w, u in U.items():
        if w not in W or len(u) < 2: continue
        tr, rr = np.mean(u['entity_planner/trained_orderer']), np.mean(u['entity_planner/round_robin'])
        if rr <= 0 or not W[w]['rho']: continue
        x = W[w]; size = int(x['size']); ch = x['choice']
        E = (4 if size == 12 else 14) if ch.startswith('dense') else int(ch.split('_f')[1])
        rows.append(dict(world=w, size=size, E=E, choice=ch, rho=float(x['rho']), pattern=x['pattern'],
                         shocks=x['shocks'] == 'true', d=(tr - rr) / rr, diff=tr - rr))
    return rows
def report(name, rows):
    d = np.array([r['d'] for r in rows]); n = len(rows)
    print(f'\n=== {name}: {n} worlds; trained beats RR on {np.mean(d>0):.0%}; median d {np.median(d):+.3f}')
    X = np.column_stack([np.ones(n), [math.log(r['E']) for r in rows], [math.log(r['rho']) for r in rows],
                         [r['size'] == 42 for r in rows], [r['pattern'] == 'burst' for r in rows], [r['shocks'] for r in rows]]).astype(float)
    b, *_ = np.linalg.lstsq(X, d, rcond=None); res = d - X @ b
    s2 = res @ res / (n - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    for k, bi, si in zip(['const', 'log E (makers)', 'log rho (load)', 'size 42', 'burst', 'shocks'], b, se):
        print(f'  {k:15s} {bi:+.3f}  (se {si:.3f}, t {bi/si:+.1f})')
    print('  median d by makers x load tercile:')
    q = np.quantile([r['rho'] for r in rows], [1/3, 2/3])
    tab = collections.defaultdict(list)
    for r in rows:
        lt = 'light' if r['rho'] <= q[0] else ('mid' if r['rho'] <= q[1] else 'heavy')
        tab[(r['size'], r['E'], lt)].append(r['d'])
    for k in sorted(tab):
        v = tab[k]; print(f'   P{k[0]:>2} E{k[1]:>2} {k[2]:5s} n={len(v):3d} median {np.median(v):+.3f} win {np.mean(np.array(v)>0):.0%}')
for d, name in [(sys.argv[1], 'EXP-0051 screen'), (sys.argv[2], 'EXP-0052 confirm')]:
    report(name, load(d))

rows = load(sys.argv[1]); d = np.array([r['d'] for r in rows]); n = len(rows)
rho = np.array([r['rho'] for r in rows]); lE = np.log([r['E'] for r in rows]); lr = np.log(rho)
print('\nrho terciles', np.round(np.quantile(rho, [0, 1/3, 2/3, 1]), 3))
X = np.column_stack([np.ones(n), lE, lr, lE * lr, [r['size'] == 42 for r in rows]]).astype(float)
b, *_ = np.linalg.lstsq(X, d, rcond=None); res = d - X @ b; s2 = res @ res / (n - 5); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
for k, bi, si in zip(['const', 'logE', 'logrho', 'logE x logrho', 'size42'], b, se): print(f'  {k:14s} {bi:+.3f} (t {bi/si:+.1f})')
for cut in (0.06, 0.1, 0.2):
    m = rho <= cut; print(f'rho <= {cut}: n={m.sum()}, trained wins {np.mean(d[m] > 0):.0%}, median d {np.median(d[m]):+.3f}')
