#!/usr/bin/env python3
"""Export the frozen analytic model; the resulting pair runs without src/."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.utils.model_loader import load_model
from src.control.hereditary_fast import FrozenHereditary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True,help='new deployment directory')
    p.add_argument('--action-scale-kpa',type=float,nargs=4,required=True)
    p.add_argument('--upper-kpa',type=float,nargs=4,required=True)
    p.add_argument('--rate-kpa-s',type=float,nargs=4,required=True)
    p.add_argument('--radius-mm',type=float,required=True)
    p.add_argument('--max-horizon',type=int,default=80)
    a=p.parse_args()
    if any(not np.isfinite(x) or x<=0 for x in a.action_scale_kpa+a.upper_kpa+a.rate_kpa_s+[a.radius_mm]):
        p.error('physical scales and limits must be positive finite values')
    if max(a.upper_kpa)>500 or not 2<=a.max_horizon<=200:p.error('invalid hardware pressure or experimental horizon limit')
    info=load_model(a.checkpoint,device='cpu');model=info['model'].eval();engine=FrozenHereditary(model)
    if info['saved_config']['state_view']['state_length_unit']!='mm':p.error('requires planar mm checkpoint')
    target=Path(a.out);target.mkdir(parents=True,exist_ok=False)
    weights=target/'hereditary.npz'
    np.savez_compressed(weights,**engine.__dict__)
    meta=dict(schema='hereditary_deployment_v1',checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),dt=float(model.dt),
              action_unit_to_kpa=(np.asarray(a.action_scale_kpa)*float(model.action_norm_factor)).tolist(),
              expansion6=info['saved_config']['action_view']['action_expansion6'],lower_kpa=[0.]*4,
              upper_kpa=a.upper_kpa,rate_kpa_s=a.rate_kpa_s,radius_mm=a.radius_mm,max_horizon=a.max_horizon,
              initialization='equilibrium at operator-confirmed zero; prior physical PI history unknown',
              evidence='development limits; not a physical control certification',source_checkpoint=str(Path(a.checkpoint).resolve()))
    weights.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(weights)
if __name__=='__main__':main()
