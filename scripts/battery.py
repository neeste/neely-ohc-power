"""Six-observation battery at the cat state (score_base.battery); writes base_battery.json in model/.
At the final state run it through run_cat.py."""
import sys; sys.path.insert(0, '.')
import score_base as SB
SB.battery()
