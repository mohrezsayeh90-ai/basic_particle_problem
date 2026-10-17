import sys,json
from pathlib import Path
import numpy as np
import torch
REPO=next(p for p in Path(__file__).resolve().parents if (p/'run_parametric_pinn.py').exists())
sys.path.insert(0,str(REPO))
from particle_pinn.parametric import ParametricFlow,Case,map_points
from particle_pinn.fem import solve_flow,relative_error
from particle_pinn.advanced_exports import write_json
from analytic_reference import AnalyticChannel
out=Path(__file__).resolve().parent;root=Path(__file__).resolve().parent.parent/'final_output'
torch.set_num_threads(1);ck=torch.load(root/'best_model.pt',map_location='cpu',weights_only=False)
model=ParametricFlow(128,6).eval();model.load_state_dict(ck['model'])
rows=json.loads((out/'global_spot_checks.json').read_text());new=[];rng=np.random.default_rng(30001)
for row in rows:
 if 'error' not in row: continue
 case=Case(**row['parameters']);c=AnalyticChannel(case.rin,case.rout)
 q=np.linspace(.001,.999,300);e=rng.uniform(-.98,.98,len(q));p=torch.tensor(case.array()[None,:3],dtype=torch.float64).expand(len(q),-1)
 expected=map_points(torch.tensor(q),torch.tensor(e),p).numpy()
 assert np.max(np.abs(expected-c.map(q*c.length,e)))<1e-12
 a,b=c.coordinates(expected);np.testing.assert_allclose(a/c.length,q,atol=1e-12);np.testing.assert_allclose(b,e,atol=1e-12)
 coarse=solve_flow(c,case.re,ns=96,nn=22,max_iter=200)
 fine=solve_flow(c,case.re,ns=128,nn=30,max_iter=200)
 x=c.sample(512,rng);pred=model.evaluate(x,case);aa=coarse.evaluate(x);bb=fine.evaluate(x)
 r=dict(index=row['index'],parameters=row['parameters'],reference_geometry='Exact piecewise-circular S channel; no spline curvature overshoot',
  coarse_ok=coarse.diagnostics['solver_ok'],fine_ok=fine.diagnostics['solver_ok'],
  velocity_error=relative_error(pred[:,:2],bb[:,:2]),pressure_error=relative_error(pred[:,2]-pred[:,2].mean(),bb[:,2]-bb[:,2].mean()),
  fem_mesh_change_velocity=relative_error(aa[:,:2],bb[:,:2]),fem_mesh_change_pressure=relative_error(aa[:,2]-aa[:,2].mean(),bb[:,2]-bb[:,2].mean()))
 new.append(r);print(json.dumps(r),flush=True);write_json(out/'wide_geometry_checks.json',new)
