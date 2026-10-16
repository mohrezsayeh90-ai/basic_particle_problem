"""Periodic scientific exports and bounded, event-aware sphere trajectories."""
import csv
import json
import math
import os
from pathlib import Path
import traceback
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/parametric-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/parametric-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from scipy.integrate import solve_ivp
from .geometry import Channel
from .fem import solve_flow,relative_error
from .parametric import anchors,drag_acceleration


def write_json(file,data):
    def convert(x):
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    file.write_text(json.dumps(data,indent=2,default=convert,allow_nan=False)+'\n')


def write_csv(file,rows):
    if not rows: return
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def trace(channel,case,field,eta,times=100):
    """Integrate dimensionless positions in q=x/R and time tau=t/R for scale robustness."""
    r=case.radius;start=channel.map(np.array([1e-5*channel.length]),np.array([eta]))[0]
    initial=np.r_[start/r,field(start[None,:])[0,:2]];evaluations=0
    def rhs(t,z):
        nonlocal evaluations
        evaluations+=1
        if evaluations>6000: raise RuntimeError('Particle RHS budget exhausted')
        flow=field((z[:2]*r)[None,:])[0,:2]
        if not np.isfinite(flow).all(): raise FloatingPointError('Non-finite flow')
        acc=drag_acceleration(flow,z[2:],case.diameter,case.re,case.density_ratio)
        return np.r_[z[2:],r*acc]
    def wall(t,z):
        _,e=channel.coordinates((z[:2]*r)[None,:])
        return channel.half_width*(1-abs(e[0]))-case.diameter/2
    def outlet(t,z): return channel.length-channel.coordinates((z[:2]*r)[None,:])[0][0]
    def inlet(t,z): return channel.coordinates((z[:2]*r)[None,:])[0][0]
    for event in [wall,outlet,inlet]: event.terminal=True;event.direction=-1
    try:
        tau=case.density_ratio*case.re*case.diameter**2/18/r
        sol=solve_ivp(rhs,(0,9*math.pi),initial,method='Radau' if tau<.01 else 'RK45',
                      rtol=1e-5,atol=1e-7,max_step=.08,dense_output=True,events=[wall,outlet,inlet])
        t=np.linspace(0,sol.t[-1],times);z=sol.sol(t).T
        outcome=next((name for name,ev in zip(['stuck','exited','backflow_exit'],sol.t_events) if len(ev)),
                     'time_limit' if sol.success else 'integration_failed')
        if outcome=='stuck': z[-1,2:]=0
        flow=field(z[:,:2]*r)[:,:2]
        rep=case.re*case.diameter*np.linalg.norm(flow-z[:,2:],axis=1)
        return dict(t=t*r,xy=z[:,:2]*r,velocity=z[:,2:],outcome=outcome,
                    max_particle_re=float(rep.max()),message=sol.message)
    except Exception as e:
        return dict(t=np.array([0.]),xy=start[None,:],velocity=initial[None,2:],
                    outcome='integration_failed',max_particle_re=None,message=str(e))


class PeriodicExporter:
    def __init__(self,out,seed=42,particles=5,reference_ns=64,reference_nn=14,animation=True):
        self.out=Path(out);self.particles=particles;self.animation=animation
        self.cases=anchors();self.channels={};self.references={};self.reference_metrics={}
        self.seed_eta={};rng=np.random.default_rng(seed+772)
        seed_fractions=np.sort(rng.uniform(-.7,.7,particles))
        (self.out/'references').mkdir(exist_ok=True)
        for name,case in self.cases.items():
            channel=Channel('serpentine',case.rin,case.rout);self.channels[name]=channel
            margin=1-case.diameter/(2*channel.half_width)
            self.seed_eta[name]=seed_fractions*margin
            try:
                f=solve_flow(channel,case.re,ns=reference_ns,nn=reference_nn,max_iter=150)
                self.reference_metrics[name]=f.diagnostics
                self.references[name]=f if f.diagnostics['solver_ok'] else None
                np.savez_compressed(self.out/'references'/f'{name}.npz',points=f.ub.mesh.p,
                    triangles=f.ub.mesh.t,velocity=f.velocity,pressure=f.pressure)
            except Exception as error:
                self.references[name]=None;self.reference_metrics[name]={'solver_ok':False,'error':str(error)}
        write_json(self.out/'references/diagnostics.json',self.reference_metrics)

    @staticmethod
    def walls(ax,channel):
        s=np.linspace(0,channel.length,300)
        for e in [-1,1]: ax.plot(*channel.map(s,np.full(len(s),e)).T,c='.3',lw=.65)
        ax.set(aspect='equal',xlabel=r'$x/L_0$',ylabel=r'$y/L_0$')

    def export(self,model,step,history):
        folder=self.out/'snapshots'/f'step_{step:06d}';folder.mkdir(parents=True,exist_ok=False)
        fig,axes=plt.subplots(2,3,figsize=(13,6.5),layout='constrained')
        summaries={};all_tracks=[];case_tracks={};errors=[]
        for col,(name,case) in enumerate(self.cases.items()):
            channel=self.channels[name];field=lambda x,c=case:model.evaluate(x,c)
            try:
                xy,s,e=channel.grid(80,25);pred=field(xy);speed=np.linalg.norm(pred[:,:2],axis=1)
                if not np.isfinite(pred).all(): raise FloatingPointError('Non-finite predicted field')
                ax=axes[0,col]
                artist=ax.pcolormesh(xy[:,0].reshape(s.shape),xy[:,1].reshape(s.shape),speed.reshape(s.shape),
                                    shading='gouraud',cmap='viridis',vmin=0)
                fig.colorbar(artist,ax=ax,shrink=.65,label='Speed / U0');self.walls(ax,channel)
                ax.set_title(f'{name.capitalize()} · velocity')
                np.savez_compressed(folder/f'{name}_field.npz',xy=xy,velocity=pred[:,:2],pressure=pred[:,2])
                paths=[trace(channel,case,field,float(eta)) for eta in self.seed_eta[name]];case_tracks[name]=paths
                ax=axes[1,col];self.walls(ax,channel)
                for i,path in enumerate(paths):
                    color=plt.get_cmap('viridis')(i/max(1,len(paths)-1))
                    ax.plot(*path['xy'].T,lw=1.2,label=f'P{i+1}',color=color)
                    ax.scatter(*path['xy'][-1],s=20,marker='x' if path['outcome']=='stuck' else 'o',c=[color])
                    for k,t in enumerate(path['t']):
                        all_tracks.append(dict(case=name,particle=i,eta=self.seed_eta[name][i],t=t,
                            x=path['xy'][k,0],y=path['xy'][k,1],up=path['velocity'][k,0],vp=path['velocity'][k,1],outcome=path['outcome']))
                ax.set_title(f'{name.capitalize()} · particle paths');ax.legend(frameon=False,ncol=3,fontsize=6)
                summary={'parameters':case.json(),'particles':[dict(particle=i,outcome=p['outcome'],
                    max_particle_re=p['max_particle_re'],message=p['message']) for i,p in enumerate(paths)],
                    'reference_solver_ok':self.reference_metrics[name].get('solver_ok',False)}
                ref=self.references.get(name)
                if ref is not None:
                    # Independent interior points, never FEM labels used in training.
                    check=channel.sample(256,np.random.default_rng(891),margin=.04)
                    a,b=field(check),ref.evaluate(check)
                    summary['velocity_relative_l2']=relative_error(a[:,:2],b[:,:2])
                    summary['pressure_relative_l2']=relative_error(a[:,2]-a[:,2].mean(),b[:,2]-b[:,2].mean())
                    rfield=ref.fast_field();rpaths=[trace(channel,case,rfield,float(eta)) for eta in self.seed_eta[name]]
                    comparisons=[]
                    for a,b in zip(paths,rpaths):
                        if len(a['t'])<2 or len(b['t'])<2:
                            comparisons.append({'available':False});continue
                        end=min(a['t'][-1],b['t'][-1]);t=np.linspace(0,end,200)
                        axy=np.column_stack([np.interp(t,a['t'],a['xy'][:,j]) for j in range(2)])
                        bxy=np.column_stack([np.interp(t,b['t'],b['xy'][:,j]) for j in range(2)])
                        comparisons.append(dict(available=True,common_time_coverage=end/max(a['t'][-1],b['t'][-1]),
                            rms_position_over_width=float(np.sqrt(np.mean(np.sum((axy-bxy)**2,axis=1)))/(2*channel.half_width)),
                            outcome_agreement=a['outcome']==b['outcome'],pinn_outcome=a['outcome'],reference_outcome=b['outcome']))
                    summary['trajectory_comparison']=comparisons
                summaries[name]=summary
            except Exception as error:
                errors.append(dict(case=name,error=str(error),traceback=traceback.format_exc()))
                summaries[name]={'parameters':case.json(),'export_error':str(error)}
                for row in range(2): axes[row,col].text(.5,.5,'Data unavailable',transform=axes[row,col].transAxes,ha='center')
        fig.suptitle(f'Flow and particle motion · step {step}')
        for ext in ['png','pdf']: fig.savefig(folder/f'three_cases.{ext}',dpi=220)
        plt.close(fig);write_csv(folder/'particle_paths.csv',all_tracks)
        write_json(folder/'metrics.json',summaries);write_json(folder/'export_errors.json',errors)
        write_csv(folder/'training_history.csv',history)
        fig,ax=plt.subplots(figsize=(6,3),layout='constrained')
        for key in ['total','physics','flux']: ax.semilogy([r['step'] for r in history],[max(r[key],1e-15) for r in history],label=key)
        ax.set(xlabel='Optimizer step',ylabel='Loss',title='Training history');ax.legend(frameon=False)
        fig.savefig(folder/'loss.png',dpi=200);plt.close(fig)
        if self.animation and case_tracks:
            try: self.animate(folder,case_tracks,step)
            except Exception as e:
                errors.append(dict(stage='animation',error=str(e)));write_json(folder/'export_errors.json',errors)
        write_json(folder/'complete.json',dict(step=step,export_errors=len(errors),
            scope='Three anchor cases; not a certificate for the full parameter range'))
        return folder

    def animate(self,folder,tracks,step):
        fig,axes=plt.subplots(1,3,figsize=(12,3.1),layout='constrained');artists=[]
        for ax,(name,case) in zip(axes,self.cases.items()):
            self.walls(ax,self.channels[name]);ax.set_title(name.capitalize())
            paths=tracks.get(name,[])
            for path in paths: ax.plot(*path['xy'].T,c='.8',lw=.5)
            initial=np.array([p['xy'][0] for p in paths]).reshape(-1,2)
            dots=ax.scatter(*initial.T,s=22,c=np.arange(len(paths)),cmap='viridis',vmin=0,vmax=max(1,len(paths)-1))
            artists.append((dots,paths))
        def update(frame):
            fraction=frame/39
            for ax,(dots,paths) in zip(axes,artists):
                end=max((p['t'][-1] for p in paths),default=1.)
                points=[]
                for p in paths:
                    t=fraction*end
                    if t>p['t'][-1] and p['outcome']=='exited': points.append([np.nan,np.nan])
                    else: points.append([np.interp(t,p['t'],p['xy'][:,j]) for j in range(2)])
                dots.set_offsets(np.array(points).reshape(-1,2))
                ax.set_title(f'{list(self.cases)[list(axes).index(ax)].capitalize()} · t/T0 = {fraction*end:.2f}')
            fig.suptitle(f'Particle motion · step {step}');return [d for d,_ in artists]
        ani=FuncAnimation(fig,update,frames=40,interval=80,blit=False)
        ani.save(folder/'particle_motion.gif',writer=PillowWriter(fps=12));plt.close(fig)
