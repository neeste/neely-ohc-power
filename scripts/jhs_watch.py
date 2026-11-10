"""Stop joint_hs when stalled (no improvement for 150 evaluations) or at 1500 (Step 1b rule). Usage: jhs_watch.py tag pid"""
import sys, json, os, time, signal
tag, pid = sys.argv[1], int(sys.argv[2]); F = 'out/joint_hs_%s_nfev.json' % tag
while True:
    try: c = json.load(open(F))
    except Exception: time.sleep(5); continue
    why = 'nfev 1500' if c['nfev'] >= 1500 else ('stalled 150' if c['nfev'] - c['last_improve'] >= 150 else None)
    if why:
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
        open('out/joint_hs_%s.log' % tag, 'a').write('STOPPED (%s) at n %d\n' % (why, c['nfev']))
        c['stopped'] = why; json.dump(c, open(F + '.stopped', 'w')); break
    try: os.kill(pid, 0)
    except ProcessLookupError: break
    time.sleep(5)
