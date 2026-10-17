import sys,json,csv,hashlib,time
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
REPO=next(p for p in Path(__file__).resolve().parents if (p/'run_parametric_pinn.py').exists())
sys.path.insert(0,str(REPO))
from particle_pinn.parametric import ParametricFlow,Case,anchors,sample_cases,map_points,coordinates,pde_residual
from particle_pinn.geometry import Channel
from particle_pinn.fem import solve_flow,relative_error
from particle_pinn.advanced_exports import trace,write_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from PIL import Image

torch.set_num_threads(1)
root=Path(__file__).resolve().parent.parent/'final_output';out=Path(__file__).resolve().parent;rng=np.random.default_rng(20261017)
ck=torch.load(root/'best_model.pt',map_location='cpu',weights_only=False)
model=ParametricFlow(128,6).eval();model.load_state_dict(ck['model'])
original=json.loads((root/'snapshots/step_192000/metrics.json').read_text())
rows=list(csv.DictReader((root/'snapshots/step_192000/particle_paths.csv').open()))
records=[];fields={};channels={};references={};tracks={}
# Mesh/field checks on fixed fresh points: no training or fitting.
for name,case in anchors().items():
 c=Channel('serpentine',case.rin,case.rout);channels[name]=c
 points=c.sample(512,rng,margin=.02)
 a=model.evaluate(points,case)
 coarse=solve_flow(c,case.re,ns=64,nn=14,max_iter=150)
 fine=solve_flow(c,case.re,ns=96,nn=22,max_iter=200);references[name]=fine
 b=coarse.evaluate(points);f=fine.evaluate(points)
 p=torch.tensor(case.array()[None,:3],dtype=torch.float32).expand(len(points),-1)
 r=pde_residual(model,torch.tensor(points,dtype=torch.float32,requires_grad=True),p).detach().numpy()
 q=torch.tensor(np.repeat([.1,.3,.5,.7,.9],65),dtype=torch.float32)
 eta=torch.tensor(np.tile(np.linspace(-1,1,65),5),dtype=torch.float32)
 pp=p[:1].expand(len(q),-1);x=map_points(q,eta,pp);_,_,t=coordinates(x,pp)
 with torch.no_grad(): u=(model(x,pp)[:,:2]*t).sum(1).reshape(5,65).numpy()
 flux=np.trapezoid(u,np.linspace(-1,1,65),axis=1)/(4/3)
 rec=dict(case=name,parameters=case.json(),coarse_solver_ok=coarse.diagnostics['solver_ok'],fine_solver_ok=fine.diagnostics['solver_ok'],
  velocity_error_coarse=relative_error(a[:,:2],b[:,:2]),velocity_error_fine=relative_error(a[:,:2],f[:,:2]),
  pressure_error_fine=relative_error(a[:,2]-a[:,2].mean(),f[:,2]-f[:,2].mean()),
  fem_mesh_change_velocity=relative_error(b[:,:2],f[:,:2]),fem_mesh_change_pressure=relative_error(b[:,2]-b[:,2].mean(),f[:,2]-f[:,2].mean()),
  continuity_scaled_rms=float(np.sqrt(np.mean(r[:,0]**2))),momentum_scaled_rms=np.sqrt(np.mean(r[:,1:]**2,axis=0)).tolist(),
  section_flux_ratio=flux.tolist(),stokes_response_time=case.density_ratio*case.re*case.diameter**2/18,
  stokes_number_radius=case.density_ratio*case.re*case.diameter**2/(18*case.radius))
 records.append(rec);print('ANCHOR',json.dumps(rec),flush=True)
 tracks[name]=[]
 # Separate coarse-vs-refined reference path tests, preserving original seeds.
 for j in range(7):
  rr=[r for r in rows if r['case']==name and int(r['particle'])==j]
  eta0=float(rr[0]['eta']);path=trace(c,case,fine.fast_field(),eta0)
  ap=dict(t=np.array([float(r['t']) for r in rr]),xy=np.array([[float(r['x']),float(r['y'])] for r in rr]))
  end=min(ap['t'][-1],path['t'][-1]);tt=np.linspace(0,end,300)
  aa=np.column_stack([np.interp(tt,ap['t'],ap['xy'][:,k]) for k in range(2)])
  bb=np.column_stack([np.interp(tt,path['t'],path['xy'][:,k]) for k in range(2)])
  tracks[name].append(dict(particle=j,eta=eta0,pinn=ap,reference=path,
   rms_over_width=float(np.sqrt(np.mean(np.sum((aa-bb)**2,axis=1)))/(2*case.h)),
   common_time_coverage=float(end/max(ap['t'][-1],path['t'][-1]))))
 write_json(out/'anchor_audit.json',records)

# Independent DOP853 in physical coordinates, direct circular contact; no Channel event or shared drag RHS.
case=anchors()['high'];R=case.radius;diam=case.diameter;rho=case.density_ratio;Re=case.re
independent=[];indep_paths=[]
for j,track in enumerate(tracks['high']):
 rr=[r for r in rows if r['case']=='high' and int(r['particle'])==j]
 z0=np.array([float(rr[0][k]) for k in ['x','y','up','vp']])
 def rhs(t,z):
  slip=model.evaluate(z[None,:2],case)[0,:2]-z[2:];speed=np.linalg.norm(slip)
  rep=Re*diam*speed
  if rep==0: acceleration=np.zeros(2)
  else:
   cd=24/rep*(1+.15*rep**.687) if rep<1000 else .44
   acceleration=3*cd/(4*rho*diam)*speed*slip
  return np.r_[z[2:],acceleration]
 def contact(t,z): return case.rout-diam/2-np.hypot(z[0]-R,z[1])
 contact.terminal=True;contact.direction=-1
 sol=solve_ivp(rhs,(0,30*R),z0,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.03*R,events=contact,dense_output=True)
 end=sol.t[-1];tt=np.linspace(0,end,300);xy=sol.sol(tt).T[:,:2]
 ap=track['pinn'];t=np.linspace(0,min(end,ap['t'][-1]),300)
 old=np.column_stack([np.interp(t,ap['t'],ap['xy'][:,k]) for k in range(2)])
 err=np.linalg.norm(old-sol.sol(t).T[:,:2],axis=1)/(2*case.h)
 ballistic_time=(np.sqrt((case.rout-diam/2)**2-(z0[0]-R)**2)-z0[1])/z0[3]
 rec=dict(particle=j,solver_success=sol.success,contact_detected=bool(len(sol.t_events[0])),independent_end_time=end,
  saved_end_time=float(ap['t'][-1]),relative_end_time_difference=float(abs(end-ap['t'][-1])/end),
  max_path_difference_over_width=float(err.max()),horizontal_displacement_over_width=float((xy[-1,0]-xy[0,0])/(2*case.h)),
  ballistic_contact_time=float(ballistic_time),ballistic_relative_time_error=float(abs(end-ballistic_time)/end))
 independent.append(rec);indep_paths.append(dict(t=tt,xy=xy));print('INDEPENDENT',json.dumps(rec),flush=True)
write_json(out/'independent_particle_check.json',independent)

# Broad-domain spot checks, not a full-domain certificate. Reynolds/geometry unobserved by FEM training.
values,_=sample_cases(rng,8,1.)
global_rows=[]
for i,v in enumerate(values):
 case=Case(*v)
 try:
  c=Channel('serpentine',case.rin,case.rout);f=solve_flow(c,case.re,ns=80,nn=18,max_iter=200)
  points=c.sample(256,rng,margin=.03);a=model.evaluate(points,case);b=f.evaluate(points)
  rec=dict(index=i,parameters=case.json(),h_over_R=case.h/case.radius,re_times_radius=case.re*case.radius,
   reference_ok=f.diagnostics['solver_ok'],velocity_error=relative_error(a[:,:2],b[:,:2]),
   pressure_error=relative_error(a[:,2]-a[:,2].mean(),b[:,2]-b[:,2].mean()))
 except Exception as e: rec=dict(index=i,parameters=case.json(),error=type(e).__name__,message=str(e))
 global_rows.append(rec);print('GLOBAL',json.dumps(rec),flush=True)
write_json(out/'global_spot_checks.json',global_rows)

torch.save(dict(records=records,tracks=tracks),out/'plot_cache.pt')

# Visual audit: include fluid streamline/tracer and independently recomputed inertial tracks.
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,(name,case) in zip(axes,anchors().items()):
 c=channels[name];s=np.linspace(0,c.length,400)
 for e in [-1,1]: ax.plot(*c.map(s,np.full(len(s),e)).T,c='.35',lw=.8)
 for j,tr in enumerate(tracks[name]):
  color=plt.get_cmap('viridis')(j/6)
  ax.plot(*tr['pinn']['xy'].T,color=color,lw=1.5,label='PINN particle' if j==0 else None)
  ax.plot(*tr['reference']['xy'].T,color='tab:red',ls='--',lw=.8,label='Refined FEM particle' if j==0 else None)
  ax.scatter(*tr['pinn']['xy'][-1],c=[color],s=16,marker='x' if name=='high' else 'o')
 if name=='high':
  # Massless fluid tracer in the same network field, for distinction from heavy particles.
  start=tracks[name][3]['pinn']['xy'][0]
  def velocity(t,x): return model.evaluate(x[None,:],case)[0,:2]
  def outlet(t,x): return c.length-c.coordinates(x[None,:])[0][0]
  outlet.terminal=True;outlet.direction=-1
  tr=solve_ivp(velocity,(0,40*case.radius),start,rtol=1e-7,atol=1e-9,max_step=.08*case.radius,events=outlet)
  ax.plot(*tr.y,c='tab:blue',ls=':',lw=1.8,label='Fluid tracer')
 ax.set(aspect='equal',xlabel=r'$x/L_0$',ylabel=r'$y/L_0$',title=f'{name.capitalize()} · St = {records[list(anchors()).index(name)]["stokes_number_radius"]:.3g}')
 ax.legend(frameon=False,fontsize=6,loc='lower left')
for ext in ['png','pdf']: fig.savefig(out/f'particle_audit.{ext}',dpi=300)
plt.close(fig)

# Revised display only: fixed domain, impact crosses, explicit clocks and shared elapsed t/R.
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');artists=[]
for ax,(name,case) in zip(axes,anchors().items()):
 c=channels[name];s=np.linspace(0,c.length,400)
 for e in [-1,1]: ax.plot(*c.map(s,np.full(len(s),e)).T,c='.35',lw=.8)
 for tr in tracks[name]:
  ax.plot(*tr['reference']['xy'].T,c='tab:red',ls='--',lw=.6,alpha=.65)
  ax.plot(*tr['pinn']['xy'].T,c='.7',lw=.5)
 ax.set(aspect='equal',xlabel=r'$x/L_0$',ylabel=r'$y/L_0$')
 dots=ax.scatter(np.full(7,np.nan),np.full(7,np.nan),s=22,c=np.arange(7),cmap='viridis',vmin=0,vmax=6)
 impact=ax.scatter([],[],marker='x',s=35,color='black');artists.append((name,case,dots,impact))
 # Set limits before empty animation artists so particle motion cannot trigger rescaling.
 ax.set_xlim(-case.h*1.25,4*case.radius+case.h*1.25);ax.set_ylim(-case.rout*1.1,case.rout*1.1)
end=max(tr['pinn']['t'][-1]/case.radius for name,case in anchors().items() for tr in tracks[name])
def update(frame):
 tau=end*frame/79
 for ax,(name,case,dots,impact) in zip(axes,artists):
  points=[];hits=[]
  for tr in tracks[name]:
   path=tr['pinn'];t=tau*case.radius;terminated=t>=path['t'][-1]
   pos=[np.interp(t,path['t'],path['xy'][:,k]) for k in range(2)]
   if terminated and name!='high': points.append([np.nan,np.nan])
   else:
    points.append(pos)
    if terminated: hits.append(pos)
  dots.set_offsets(np.array(points));impact.set_offsets(np.array(hits).reshape(-1,2))
  ax.set_title(f'{name.capitalize()} · t/R = {tau:.2f}\nSt = {records[list(anchors()).index(name)]["stokes_number_radius"]:.3g}')
 return []
ani=FuncAnimation(fig,update,frames=80,interval=100,blit=False)
ani.save(out/'particle_audit.gif',writer=PillowWriter(fps=12));plt.close(fig)
write_json(out/'trajectory_refined_reference.json',{name:[{k:v for k,v in tr.items() if k not in ['pinn','reference']} for tr in rr] for name,rr in tracks.items()})
print('AUDIT_COMPLETE',flush=True)
