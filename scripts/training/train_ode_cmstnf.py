"""train_ode_cmstnf.py — ODE-CMSTNF 两阶段训练（Neural ODE 时序编码）。"""

import os
import sys
import argparse
import torch

CUDA_DEVICE = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.training.trainer_ode_cmstnf import ODECMSTNFTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
parser.add_argument("--data_dir", type=str, default="data/sequence_data_1d")
parser.add_argument("--canonical_data_dir", type=str, default="data/canonical_data")
parser.add_argument("--canonical_path", type=str, default=None)
parser.add_argument("--exp_dir", type=str, default=None)
args = parser.parse_args()

print(f"Device: {device}")
trainer = ODECMSTNFTrainer(device=device)

if args.phase == 1:
    trainer.train_phase1(exp_dir=args.exp_dir, data_dir=args.canonical_data_dir)
elif args.phase == 2:
    trainer.train_phase2(exp_dir=args.exp_dir, canonical_path=args.canonical_path,
                         data_dir=args.data_dir)
else:
    trainer.train(data_dir=args.data_dir, canonical_data_dir=args.canonical_data_dir)
