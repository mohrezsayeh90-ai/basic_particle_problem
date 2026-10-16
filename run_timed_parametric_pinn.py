#!/usr/bin/env python3
"""Fresh time-budgeted parametric PINN; all scientific products exported once at the end."""
import argparse,copy,json,math,os,random,time,traceback
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/timed_particle_matplotlib')
import numpy as np
import torch
from particle_pinn.parametric import ParametricFlow,BASE,NAMES,pde_residual,map_points,coordinates
from particle_pinn.advanced_exports import PeriodicExporter,write_json
from run_parametric_pinn import batch,atomic_save


def objective(model,xy,p,cases,continuity_weight):
    r=pde_residual(model,xy,p)
    continuity=r[:,0].square().mean();momentum=r[:,1:].square().mean()
    c=torch.as_tensor(cases[:6,:3],dtype=torch.float32,device=p.device)
    pp=c.repeat_interleave(51,dim=0)
    q=torch.tensor([.2,.5,.8],device=p.device).repeat_interleave(17).repeat(len(c))
    eta=torch.linspace(-1,1,17,device=p.device).repeat(3*len(c))
    xx=map_points(q,eta,pp);_,_,t=coordinates(xx,pp)
    ut=(model(xx,pp)[:,:2]*t).sum(1).reshape(len(c),3,17)
    flux=(torch.trapezoid(ut,torch.linspace(-1,1,17,device=p.device),dim=2)/(4/3)-1).square().mean()
    return continuity_weight*continuity+momentum+5*flux,continuity,momentum,flux


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=6.)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--samples',type=int,default=2048)
    parser.add_argument('--width',type=int,default=128);parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--continuity-weight',type=float,default=20.)
    parser.add_argument('--reference-ns',type=int,default=64);parser.add_argument('--reference-nn',type=int,default=14)
    parser.add_argument('--particles',type=int,default=7);parser.add_argument('--no-animation',action='store_true')
    args=parser.parse_args()
    if args.hours<=0 or args.samples<4: parser.error('Positive time budget and samples >=4 required')
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA required; no CPU fallback')
    if args.out.exists() and any(args.out.iterdir()): parser.error('Output directory must be empty')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed)
    rng=np.random.default_rng(args.seed);model=ParametricFlow(args.width,args.depth).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-6)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(parameters=dict(zip(NAMES,BASE.tolist())),parameter_factors=[.1,10],start_from='random initialization',
      sampling='Full valid log-uniform domain from the first update; no curriculum',
      independent_flow_inputs=['x','y','rin','rout','re'],particle_inputs=['diameter','density_ratio'],
      coupling='One-way Schiller-Naumann drag; diameter and density do not condition the fluid network',
      optimizer='AdamW; wall-time cosine 5e-4 to 1e-5; 5-minute warmup',exports='Final only; no dataset publication')
    write_json(args.out/'config.json',config)
    source=os.environ.get('PARTICLE_SOURCE_MANIFEST')
    if source: (args.out/'source_manifest.json').write_bytes(Path(source).read_bytes())
    check_rng=np.random.default_rng(args.seed+809)
    check_xy,check_p,check_cases=batch(check_rng,512,1,1,args.device)
    check_xy=check_xy.detach();best=copy.deepcopy(model.state_dict());best_score=float('inf');best_step=0
    history=[];validation=[];failures=[];step=0;started=time.monotonic();budget=3600*args.hours
    gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None
    print(json.dumps(dict(event='TRAINING_STARTED',hours=args.hours,gpu=gpu,samples=args.samples,
        domain_factors=[.1,10],start_step=0,final_only=True)),flush=True)
    try:
        while time.monotonic()-started<budget or step==0:
            elapsed=time.monotonic()-started;fraction=min(1.,elapsed/budget)
            lr=(1e-5+(5e-4-1e-5)*.5*(1+math.cos(math.pi*fraction)))*min(1.,.1+.9*elapsed/min(300.,budget*.05))
            for group in opt.param_groups: group['lr']=lr
            model.train();xy,p,cases=batch(rng,args.samples,1,1,args.device)
            opt.zero_grad(set_to_none=True);loss,continuity,momentum,flux=objective(model,xy,p,cases,args.continuity_weight)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(norm): raise FloatingPointError('Non-finite gradient')
            opt.step();step+=1
            if step==1 or step%100==0:
                row=dict(step=step,elapsed_seconds=time.monotonic()-started,total=float(loss.detach()),
                    physics=float((continuity+2*momentum).detach()/3),flux=float(flux.detach()),
                    continuity=float(continuity.detach()),momentum=float(momentum.detach()),lr=lr)
                history.append(row);print(json.dumps(row),flush=True)
            if step==1 or step%1000==0:
                model.eval();probe=objective(model,check_xy.clone().requires_grad_(True),check_p,check_cases,args.continuity_weight)
                score=float(probe[0].detach());validation.append(dict(step=step,fixed_loss=score))
                print(json.dumps(dict(event='FIXED_PROBE',step=step,loss=score)),flush=True)
                if np.isfinite(score) and score<best_score:
                    best_score=score;best_step=step;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                del probe
    except Exception as e:
        failures.append(dict(stage='training',step=step,error=str(e),traceback=traceback.format_exc()))
    training_seconds=time.monotonic()-started
    # A single final evaluation also considers the most recent weights.
    try:
        model.eval();probe=objective(model,check_xy.clone().requires_grad_(True),check_p,check_cases,args.continuity_weight)
        score=float(probe[0].detach());validation.append(dict(step=step,fixed_loss=score))
        if np.isfinite(score) and score<best_score:
            best_score=score;best_step=step;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        del probe
    except Exception as e: failures.append(dict(stage='final_probe',error=str(e)))
    atomic_save(dict(scope='timed-parametric-s-channel-v1',step=step,model=model.state_dict(),optimizer=opt.state_dict(),
        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),config=config,history=history),args.out/'checkpoint.pt')
    atomic_save(dict(step=best_step,model=best,config=config,selection='Lowest weighted residual on fixed held-out probe; FEM unused'),args.out/'best_model.pt')
    write_json(args.out/'fixed_validation.json',validation)
    write_json(args.out/'selection.json',dict(final_optimizer_step=step,selected_step=best_step,fixed_loss=best_score))
    model.load_state_dict(best);model.cpu().eval()
    try:
        exporter=PeriodicExporter(args.out,args.seed,args.particles,args.reference_ns,args.reference_nn,not args.no_animation)
        folder=exporter.export(model,best_step,history)
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
        for name,case in exporter.cases.items():
            ref=exporter.references.get(name)
            if ref is None: continue
            points=exporter.channels[name].sample(400,np.random.default_rng(891),margin=.04)
            pred=model.evaluate(points,case);truth=ref.evaluate(points)
            for j,ax in enumerate(axes):
                a=truth[:,j]-truth[:,j].mean() if j==2 else truth[:,j]
                b=pred[:,j]-pred[:,j].mean() if j==2 else pred[:,j]
                ax.scatter(a,b,s=5,alpha=.4,label=name.capitalize())
        for ax,title in zip(axes,['Axial velocity','Transverse velocity','Pressure']):
            lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],'k--',lw=1)
            ax.set(title=title,xlabel='FEM',ylabel='PINN');ax.legend(frameon=False,fontsize=8)
        for ext in ['png','pdf']: fig.savefig(args.out/f'validation.{ext}',dpi=300)
        plt.close(fig)
        print('FINAL_RESULTS:',str(folder),flush=True)
        print((folder/'metrics.json').read_text(),flush=True)
    except Exception as e: failures.append(dict(stage='final_export',error=str(e),traceback=traceback.format_exc()))
    write_json(args.out/'failures.json',failures)
    write_json(args.out/'run_complete.json',dict(step=step,selected_step=best_step,training_seconds=training_seconds,
        requested_training_seconds=budget,total_seconds=time.monotonic()-started,failures=len(failures),gpu=gpu,
        stop_reason='time_budget' if not failures else 'inspect_failures',exports='Final only'))
    print('RUN_FINISHED:',(args.out/'run_complete.json').read_text(),flush=True)
    return 2 if failures else 0

if __name__=='__main__': raise SystemExit(main())
