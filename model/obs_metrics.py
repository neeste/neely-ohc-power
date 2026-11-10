"""obs_metrics.py: tuning-curve metrics for observations 4 and 5
(tip-to-tail difference and tail slope; second-map place and phase step),
as defined in the Methods. Input is the report dict produced by
exp_encoded1.build_rep / exp_obs46. Standard library only."""
import math
from dataclasses import dataclass, field
TAIL_WINDOW_OCT = -1.0       # points this far below CF and lower are tail
TIP_TO_TAIL_REF_OCT = -1.0   # tail level is read here

@dataclass
class Metric:
    name: str
    value: object

@dataclass
class Observation:
    number: int
    metrics: list = field(default_factory=list)

def _linfit(xs, ys):
    n = len(xs)
    if n < 2: return None
    mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    if sxx == 0: return None
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx
    return sl, my - sl*mx

def _first_crossing(xs, ys, level):
    for i in range(1, len(xs)):
        y0, y1 = ys[i-1], ys[i]
        if (y0 < level) != (y1 < level):
            if y1 == y0: return xs[i]
            t = (level-y0)/(y1-y0); return xs[i-1] + t*(xs[i]-xs[i-1])
    return None

def _median(v):
    s = sorted(v); n = len(s)
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])

def _oct(fr, f): return math.log2(f/fr)

def obs4(rep):
    o = Observation(4); tc = rep.get('tuning_curve') or {}
    f = tc.get('frequency_hz') or []; th = tc.get('threshold_db_spl') or []
    if len(f) < 4 or len(f) != len(th):
        o.metrics = [Metric('tip-to-tail', None), Metric('tail slope', None)]; return o
    tip = min(th); cf = tc.get('place_cf_hz') or f[th.index(tip)]
    tail = [(_oct(cf, x), y) for x, y in zip(f, th) if _oct(cf, x) <= TAIL_WINDOW_OCT]
    fit = _linfit([a for a, _ in tail], [b for _, b in tail]) if len(tail) >= 2 else None
    if fit is None:
        o.metrics = [Metric('tip-to-tail', None), Metric('tail slope', None)]; return o
    sl, ic = fit
    o.metrics = [Metric('tip-to-tail', sl*TIP_TO_TAIL_REF_OCT + ic - tip), Metric('tail slope', -sl)]
    return o

def obs5(rep):
    o = Observation(5); tc = rep.get('tuning_curve') or {}; mp = None
    f = tc.get('frequency_hz') or []; th = tc.get('threshold_db_spl') or []
    cf = tc.get('place_cf_hz') or (f[th.index(min(th))] if th else None)
    if len(f) >= 6 and len(f) == len(th) and cf:
        tail = [(_oct(cf, x), y) for x, y in zip(f, th) if _oct(cf, x) <= -1.0]
        tp = [(_oct(cf, x), y) for x, y in zip(f, th) if -0.35 <= _oct(cf, x) <= 0.0]
        ft = _linfit([p[0] for p in tail], [p[1] for p in tail]) if len(tail) >= 2 else None
        fp = _linfit([p[0] for p in tp], [p[1] for p in tp]) if len(tp) >= 2 else None
        if ft and fp and abs(ft[0]-fp[0]) > 1e-9:
            mp = (fp[1]-ft[1])/(ft[0]-fp[0])
    o.metrics.append(Metric('second map', mp))
    ph = (rep.get('transduction_phase') or {}).get('phase_rad') or []
    o.metrics.append(Metric('phase step', (max(ph)-min(ph)) if len(ph) >= 3 else None))
    return o

def _suppression_slope(*a, **k): raise NotImplementedError('not used by the paper scripts')
def score(*a, **k): raise NotImplementedError('not used by the paper scripts')
def format_report(*a, **k): raise NotImplementedError('not used by the paper scripts')
