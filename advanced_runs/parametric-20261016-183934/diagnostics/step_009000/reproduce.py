import sys,io,json,subprocess,csv
from pathlib import Path
import numpy as np
import torch
REPO=next(p for p in Path(__file__).resolve().parents if (p/'run_parametric_pinn.py').exists())
sys.path.insert(0,str(REPO))
from particle_pinn.parametric import ParametricFlow,anchors,sample_cases,map_points,pde_residual
from run_parametric_pinn import objective
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(2)
repo=REPO;out=Path(__file__).resolve().parent;rng=np.random.default_rng(20261016)
sets={}
for name,case in anchors().items(): sets[name]=np.repeat(case.array()[None,:3],256,axis=0)
cases,_=sample_cases(rng,32,span=1.)
sets['global']=np.repeat(cases[:,:3],32,axis=0)
probes={}
for name,values in sets.items():
 p=torch.tensor(values,dtype=torch.float32)
 xy=map_points(torch.tensor(rng.uniform(.002,.998,len(p)),dtype=torch.float32),torch.tensor(rng.uniform(-.995,.995,len(p)),dtype=torch.float32),p).detach()
 probes[name]=(xy,p)

file='advanced_runs/parametric-20261016-183934/checkpoint.pt'
revisions=[('1000','ee377f4'),('5000','a5c9150'),('9000','2fcb458')]
rows=[]
for label,commit in [('0',None)]+revisions:
 torch.manual_seed(42);model=ParametricFlow(128,6).eval()
 if commit:
  data=subprocess.check_output(['git','show',commit+':'+file],cwd=repo)
  ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
  assert ck['step']==int(label),(ck['step'],label)
  model.load_state_dict(ck['model'])
 for name,(xy,p) in probes.items():
  r=pde_residual(model,xy.clone().requires_grad_(True),p).detach().numpy()
  row=dict(step=int(label),probe=name,physics=float(np.mean(r*r)),continuity_rms=float(np.sqrt(np.mean(r[:,0]**2))),momentum_x_rms=float(np.sqrt(np.mean(r[:,1]**2))),momentum_y_rms=float(np.sqrt(np.mean(r[:,2]**2))))
  rows.append(row);print(json.dumps(row),flush=True)
with (out/'fixed_probe.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for name in probes:
 rr=[r for r in rows if r['probe']==name]
 axs[0].semilogy([r['step'] for r in rr],[r['physics'] for r in rr],'-o',label=name.capitalize())
axs[0].set(xlabel='Optimizer step',ylabel='Fixed-probe physics loss',title='Fixed points');axs[0].legend(frameon=False);axs[0].grid(alpha=.15)
root=repo/'advanced_runs/parametric-20261016-183934/snapshots'
for name in ['low','base','high']:
 rr=[(int(d.name.split('_')[1]),json.loads((d/'metrics.json').read_text())[name]) for d in sorted(root.glob('step_*')) if int(d.name.split('_')[1])<=9000]
 axs[1].plot([a for a,b in rr],[100*b['velocity_relative_l2'] for a,b in rr],'-o',label=name.capitalize())
axs[1].set(xlabel='Optimizer step',ylabel='Velocity relative L2 error (%)',title='Independent FEM');axs[1].legend(frameon=False);axs[1].grid(alpha=.15)
for ext in ['png','pdf']: fig.savefig(out/f'training_diagnosis.{ext}',dpi=240)
