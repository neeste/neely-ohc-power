"""File names of the adopted states, read from model/ (the working directory of every script).
Cat c856: the combined-fit record holds the section p, lambda and the profile/read-out vector z.
  combo_c856v.json : z with the velocity read-out (default)
  combo_r7b.json   : z with the displacement read-out (FTC_CRIT=disp); otherwise identical
  CAT_COMBO=<file> overrides both (for example an earlier state in earlier/).
Human hs856: joint_hs_hs856.json (identical to abr5_ckpt.json); HUMAN_STATE=<file> overrides."""
import os
CRIT = os.environ.get('FTC_CRIT', 'vel')
COMBO = os.environ.get('CAT_COMBO', 'combo_r7b.json' if CRIT == 'disp' else 'combo_c856v.json')
HUMAN = os.environ.get('HUMAN_STATE', 'joint_hs_hs856.json')
OUT = 'out'
os.makedirs(OUT, exist_ok=True)
