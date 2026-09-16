"""Input-lifted linear latent dynamics and learned-actuation planar PCC."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class KoopmanShape(nn.Module):
    """Controlled Koopman-core adaptation: z+ = A z + B phi(u), S = C z.

    Pressure is nonlinearly lifted; latent evolution and shape readout are linear.
    This action-only adaptation does not identify an observed-state Koopman lift.
    A is constrained to spectral norm below one, with no test-state initialization.
    """
    def __init__(self, hidden=64, latent=8):
        super().__init__()
        self.lift = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, latent))
        self.raw_a = nn.Parameter(torch.eye(latent)*.8)
        self.control = nn.Linear(latent, latent, bias=False)
        self.readout = nn.Linear(latent, 45)

    def forward(self, actions):
        # Frobenius norm bounds the spectral norm and is differentiable at I.
        a = .99*self.raw_a/self.raw_a.norm().clamp_min(1.)
        lifted = self.control(self.lift(actions))
        z = torch.zeros_like(lifted[:, 0])
        for t in range(actions.shape[1]):
            z = F.linear(z, a) + lifted[:, t]
        return self.readout(z).reshape(-1,15,3)


class PCCShape(nn.Module):
    """Two planar constant-curvature sections with learned pressure actuation.

    The pressure MLP predicts two total bend angles and two log length scales.
    Rest lengths/base are fitted to training labels; no shape is observed at test.
    """
    def __init__(self, hidden, metadata, normalization):
        super().__init__()
        self.actuation = nn.Sequential(nn.Linear(4,hidden), nn.Tanh(), nn.Linear(hidden,4))
        nn.init.zeros_(self.actuation[-1].weight)
        nn.init.zeros_(self.actuation[-1].bias)
        self.register_buffer('rest_lengths',torch.tensor(metadata['rest_lengths'],dtype=torch.float32))
        self.register_buffer('base',torch.tensor(metadata['base'],dtype=torch.float32))
        self.register_buffer('center',torch.tensor(normalization[0],dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(normalization[1],dtype=torch.float32))

    def forward(self, actions):
        q = self.actuation(actions[:,-1])
        angles = math.pi*torch.tanh(q[:,:2])
        lengths = self.rest_lengths*torch.exp(.25*torch.tanh(q[:,2:]))
        base = self.base.expand(len(actions),3)
        heading = actions.new_full((len(actions),),math.pi/2)
        grid = torch.linspace(0,1,8,device=actions.device,dtype=actions.dtype)[None,:]
        parts=[]
        for section in range(2):
            phase = angles[:,section,None]*grid
            distance = lengths[:,section,None]*grid*torch.sinc(phase/(2*math.pi))
            tangent = heading[:,None]+phase/2
            delta = torch.stack([distance*torch.cos(tangent), distance*torch.sin(tangent),torch.zeros_like(distance)],-1)
            points=base[:,None,:]+delta
            parts.append(points if section==0 else points[:,1:])
            base=points[:,-1]
            heading=heading+angles[:,section]
        return (torch.cat(parts,1)-self.center)/self.scale


def pcc_priors(sequences, history, stride=1, maximum=None):
    from src.benchmarks.modeling_data import CausalWindows
    windows=CausalWindows(sequences,history,stride,maximum)
    y=np.stack([sequences[i]['positions'][t] for i,t in windows.indices])
    lengths=np.linalg.norm(np.diff(y,axis=1),axis=-1)
    return dict(rest_lengths=[float(lengths[:,:7].sum(1).mean()),float(lengths[:,7:].sum(1).mean())],
                base=y[:,0].mean(0).tolist(), fixed_base_tangent_rad=math.pi/2)
