"""Run 9: wave-V latency detail from a joint_hs state json (as results_facts 2.5 for hs708): residual = model (lat + t_n)
minus the NNGJ1988 surface 5 + 12.9 x 5^(-L/100) x f^(-0.413); rms overall, by level, by frequency; largest residuals;
latency-intensity slope 40 to 80 dB SPL (ms per 10 dB), model and data.  Usage: latency_detail9.py state.json [out.json]"""
import sys, json, numpy as np
c = json.load(open(sys.argv[1])); tn = c['readout']['a']
K = np.array(c['keys']); f, L = K[:, 0], K[:, 1]; m = np.array(c['lat']) + tn
d = 5 + 12.9*5.0**(-L/100)*f**(-0.413); r = m - d
out = dict(rms=float(np.sqrt(np.mean(r**2))), n=len(r), t_n=tn,
           rms_level={int(l): float(np.sqrt(np.mean(r[L == l]**2))) for l in sorted(set(L))},
           rms_freq={float(q): float(np.sqrt(np.mean(r[f == q]**2))) for q in sorted(set(f))})
o = np.argsort(-np.abs(r)); out['largest'] = [dict(f=float(f[i]), L=float(L[i]), resid=float(r[i]), model=float(m[i]), data=float(d[i])) for i in o[:4]]
sl = {}
for q in sorted(set(f)):
    g = lambda arr, l: float(arr[(f == q) & (L == l)][0])
    sl[float(q)] = dict(model=(g(m, 40) - g(m, 80))/4, data=(g(d, 40) - g(d, 80))/4)
out['slope_40_80_ms_per_10dB'] = sl
print(json.dumps(out, indent=1))
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], 'w'), indent=1)
