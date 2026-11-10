"""Build a cat state directory for an earlier combined-fit record (for example earlier/combo_c708.json, the
lambda = 0.7085 state used for Supplement Table SVIII), next to model/ so that ../scripts and ../data resolve.
The directory gets copies of the model code and best_d3.json, base_ckpt.json and base_ckpt_disp.json built
from the record (section p, lambda, profile and read-out vector z; the record's own read-out is used for both
files), as model/ holds them for c856.
Usage (from model/): python3 ../scripts/mkstate.py earlier/combo_c708.json ../state_c708
Then: cd ../state_c708 && python3 ../scripts/run_cat.py ../scripts/<script>.py [args]"""
import sys, os, json, glob, shutil
src, out = sys.argv[1], sys.argv[2]
ck = json.load(open(src))
os.makedirs(os.path.join(out, 'out'), exist_ok=True)
for f in glob.glob('*.py'):
    shutil.copy(f, out)
NAMES = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r', 'gamma', 'q', 'm', 'u_thr', 's_g')
json.dump(dict(score=ck['score'], lam=ck['lam'], source=os.path.basename(src), params=dict(zip(NAMES, ck['p']))),
          open(os.path.join(out, 'best_d3.json'), 'w'), indent=1)
for name in ('base_ckpt.json', 'base_ckpt_disp.json'):
    json.dump(dict(z=ck['z'], J=ck.get('J_ftc'), worst_margin=ck['score'], lam=ck['lam'], source=os.path.basename(src)),
              open(os.path.join(out, name), 'w'), indent=1)
shutil.copy(src, os.path.join(out, os.path.basename(src)))
print('state directory', out, 'from', src, 'lam %.4f' % ck['lam'])
