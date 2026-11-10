"""Stop a combo start when the approved Step-1 rule is met (brief section 4):
nfev >= 1500, or score >= 0.10 with no objective (= J_ftc) improvement for 150 evaluations,
or no objective improvement for 300 evaluations. Usage: combo_watch.py tag pid"""
import sys, json, os, time, signal
tag, pid = sys.argv[1], int(sys.argv[2])
R = 'out/'
while True:
    try:
        c = json.load(open(R + 'combo_%s_nfev.json' % tag)); b = json.load(open(R + 'combo_%s.json' % tag))
    except Exception:
        time.sleep(2); continue
    n, stall = c['nfev'], c['nfev'] - c['last_improve']
    why = None
    if n >= 1500: why = 'nfev 1500'
    elif b['score'] >= 0.10 and stall >= 150: why = 'score>=0.10 and J_ftc stalled 150'
    elif stall >= 300: why = 'no improvement for 300'
    if why:
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
        c['stopped'] = why; c['stopped_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        json.dump(c, open(R + 'combo_%s_nfev.json' % tag, 'w'))
        with open(R + 'combo_%s.log' % tag, 'a') as f: f.write('STOPPED (%s) at n %d\n' % (why, n))
        break
    try: os.kill(pid, 0)
    except ProcessLookupError: break
    time.sleep(2)
