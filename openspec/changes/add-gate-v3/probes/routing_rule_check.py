import csv, json, statistics, collections, sys
csv.field_size_limit(10**9)
E='/srv/research/tokenomics/experiments/'
atl=json.load(open(E+'EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json'))
print('spreads keys sample', {k: {kk: (round(vv,1) if isinstance(vv,(int,float)) else vv) for kk,vv in v.items()} for k,v in list(atl['spreads'].items())[:2]})
STRONG=['round_robin','random_whole','makeable_weighted_whole','makeable_split']
BLIND=['blend_0.5','even_split','obs_price','proportional','random','shock_reactive','round_robin','even_split_rotating','random_whole']
REACH=['makeable_weighted_whole','makeable_split']
rows=[r for r in csv.DictReader(open(E+'EXP-0051-headroom-atlas-screen/worlds.csv')) if r['anchor']!='true']
g=collections.defaultdict(list)
for r in rows:
    s=int(r['size']); c=r['choice']; a=json.loads(r['arm_units'])
    get=lambda k: a.get('entity_planner/'+k)
    st=[get(k) for k in STRONG if get(k) is not None]
    bl=[get(k) for k in BLIND if get(k) is not None]; re=[get(k) for k in REACH if get(k) is not None]
    if len(st)<4 or not re: continue
    b=max(st); sp=(b-min(st))/b if b else 0
    rg=(max(re)-max(bl))/max(bl) if max(bl) else 0
    lab=f"{s} {c}"+(' band' if 0.28<float(r['rho'])<=3.1 else ' light' if float(r['rho'])<=0.28 else ' heavy')
    g[lab].append((sp,rg))
for k,v in sorted(g.items()):
    sp=[x for x,_ in v]; rg=[y for _,y in v]
    print(f"{k:22s} n={len(v):3d} strong-spread med {statistics.median(sp):.3f} share>=.05 {sum(x>=.05 for x in sp)/len(v):.2f} share>=.10 {sum(x>=.10 for x in sp)/len(v):.2f} | reach-gain med {statistics.median(rg):+.3f} share>.02 {sum(y>.02 for y in rg)/len(v):.2f} share>.05 {sum(y>.05 for y in rg)/len(v):.2f}")
