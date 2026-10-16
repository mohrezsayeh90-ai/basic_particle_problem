#!/usr/bin/env python3
"""Compose full flow checks and explicitly selected particle trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from matplotlib.lines import Line2D
from particle_pinn.api import load_run
from make_publication_figures import plt, style, read_pairs, csv_out, load_paths, INK, BLUE, ORANGE, GRAY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/final_study'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    out=args.out or run/'trajectory_comparison/validation_total'
    if out.exists() and any(out.iterdir()): parser.error('Choose an empty output directory.')
    torch.set_num_threads(2)
    loaded=load_run(run);channel=loaded['channel']
    rows=read_pairs(run/'trajectory_comparison/data/pairs.csv')
    baseline=sorted([r for r in rows if r['comparison']=='Full model' and r['case']=='baseline'],key=lambda r:r['eta'])
    if len(baseline)!=7: raise ValueError('Expected seven baseline seeds.')
    # Explicit user-selected subset: omit coverage 96.2%, 87.4%, and 73.6%.
    selected=baseline[:4];excluded=baseline[4:]
    paths={p.track:p for p in load_paths(run/'data/trajectories.csv')}
    field=np.genfromtxt(run/'data/field_validation.csv',delimiter=',',names=True)
    out.mkdir(parents=True);(out/'data').mkdir()
    style()
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8})
    fig=plt.figure(figsize=(10.2,9.6),layout='constrained')
    grid=fig.add_gridspec(3,3,height_ratios=[1,1,1.05])
    axes=[[fig.add_subplot(grid[i,j]) for j in range(3)] for i in range(3)]
    fig.suptitle('Validation total',fontsize=18,fontweight='medium')
    profile_rows=[];eta=np.linspace(-.98,.98,100)
    femfield=loaded['reference'].fast_field()
    for j,fraction in enumerate([.2,.5,.8]):
        ax=axes[0][j];s=fraction*channel.length
        xy=channel.map(np.full(len(eta),s),eta);_,t,_=channel.frame(s)
        pred=loaded['flow'].evaluate(xy)[:,:2]@t;ref=femfield(xy)[:,:2]@t
        ax.plot(pred,eta,c=BLUE,lw=1.7,label='PINN')
        ax.plot(ref,eta,c=ORANGE,lw=1.6,ls=(0,(4,2)),label='FEM')
        ax.set(xlabel='Tangential velocity',ylabel=r'$n/h$',ylim=(-1.04,1.04))
        ax.set_title(f'({chr(97+j)}) '+rf'$s/L={fraction:.1f}$',loc='left')
        ax.grid(alpha=.35);ax.legend(frameon=False,fontsize=8,loc='upper right')
        profile_rows.extend(dict(s_over_L=fraction,eta=e,pinn_tangential_velocity=a,fem_tangential_velocity=b)
                            for e,a,b in zip(eta,pred,ref))
    for j,(name,title) in enumerate([('u','Horizontal velocity'),('v','Vertical velocity'),('p','Pressure')]):
        ax=axes[1][j];ref=field['fem_'+name].copy();pred=field['pinn_'+name].copy()
        if name=='p': ref-=ref.mean();pred-=pred.mean()
        lo=min(ref.min(),pred.min());hi=max(ref.max(),pred.max());pad=.055*(hi-lo)
        ax.scatter(ref,pred,c=BLUE,s=5,alpha=.28,edgecolors='none')
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],c=INK,ls=(0,(5,3)),lw=1)
        ax.set(xlabel='FEM',ylabel='PINN',xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),aspect='equal')
        ax.set_title(f'({chr(100+j)}) {title}',loc='left');ax.grid(alpha=.35)
    colors=[plt.get_cmap('viridis')(q) for q in [.1,.32,.55,.77]]
    ax=axes[2][0]
    vals=[r[k] for r in selected for k in ['reference_end_time','predicted_end_time']]
    lo=min(vals)-.14;hi=max(vals)+.18
    ax.plot([lo,hi],[lo,hi],c=INK,ls=(0,(5,3)),lw=1)
    offsets=[(-43,15),(-8,-27),(27,-10),(22,25)]
    for r,color,offset in zip(selected,colors,offsets):
        xy=(r['reference_end_time'],r['predicted_end_time'])
        ax.scatter(*xy,s=45,c=[color],edgecolors='white',linewidths=.7,zorder=3)
        ax.annotate(f"{r['eta']:+.2f}",xy,xytext=offset,textcoords='offset points',fontsize=8,
                    arrowprops=dict(arrowstyle='-',lw=.55,color=GRAY))
    ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal',xlabel=r'FEM end time / $T_0$',ylabel=r'PINN end time / $T_0$')
    ax.set_title('(g) Termination timing',loc='left');ax.grid(alpha=.35)
    ax=axes[2][1];positions=np.arange(4)
    bars=ax.bar(positions,[100*r['common_time_coverage'] for r in selected],color=colors,width=.6)
    ax.axhline(95,c=GRAY,ls='--',lw=.8)
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),
                    xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(positions,[f"{r['eta']:+.2f}" for r in selected])
    ax.set(xlabel=r'Inlet position $\eta_0$',ylabel='Common-time coverage (%)',ylim=(0,113))
    ax.set_title('(h) Comparison coverage',loc='left');ax.grid(axis='y',alpha=.35)
    ax=axes[2][2];s=np.linspace(0,channel.length,500)
    for e in [-1,1]: ax.plot(*channel.map(s,np.full(len(s),e)).T,c='#B5BFC8',lw=1,zorder=0)
    for r,color in zip(selected,colors):
        a=paths[int(r['predicted_track'])];b=paths[int(r['reference_track'])]
        ax.plot(*a.xy.T,c=color,lw=1.4)
        ax.plot(*b.xy.T,c=color,lw=1.1,ls=(0,(3,3)))
        # Mark equal physical-time snapshots, rather than equally spaced path positions.
        times=np.linspace(a.t[0],min(a.t[-1],b.t[-1]),5)[1:-1]
        xy=np.column_stack([np.interp(times,a.t,a.xy[:,k]) for k in range(2)])
        ax.scatter(*xy.T,s=12,c=[color],edgecolors='white',linewidths=.35,zorder=4)
        index=int(.6*(len(a.xy)-1))
        ax.annotate('',a.xy[index+5],xytext=a.xy[index-5],
                    arrowprops=dict(arrowstyle='-|>',color=color,lw=.8,mutation_scale=7))
    ax.set(xlabel=r'$x/L_0$',ylabel=r'$y/L_0$',aspect='equal',ylim=(-.7,.62))
    ax.set_title('(i) Particle paths',loc='left')
    ax.legend(handles=[Line2D([0],[0],c=INK,label='PINN + ForceNet'),
                       Line2D([0],[0],c=INK,ls='--',label='FEM + Stokes')],
              loc='lower left',frameon=False,fontsize=7)
    fig.supxlabel('Trajectory panels: 4 selected baseline seeds of 7; colors match across (g–i).',fontsize=8,color=INK)
    for ext in ['pdf','svg','png','tiff']:
        options={'dpi':600} if ext in ['png','tiff'] else {}
        if ext=='tiff': options['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(out/f'validation_total.{ext}',**options)
    plt.close(fig)
    csv_out(out/'data/velocity_profiles.csv',profile_rows)
    csv_out(out/'data/selected_trajectories.csv',selected)
    csv_out(out/'data/excluded_trajectories.csv',excluded)
    inputs=[run/'data/field_validation.csv',run/'data/fem_reference.npz',run/'data/trajectories.csv',
            run/'models/flow_seed_42.pt',run/'config.json',run/'trajectory_comparison/data/pairs.csv']
    manifest=dict(title='Validation total',trajectory_scope='Four explicitly selected baseline seeds out of seven',
                  selected_eta=[r['eta'] for r in selected],excluded_eta=[r['eta'] for r in excluded],
                  field_parity_samples=len(field),profile_samples=len(profile_rows),raster_dpi=600,
                  source_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  selection_is_not_full_trajectory_validation=True,pressure_gauge='Separate mean subtraction',
                  motion_markers='Three equal-physical-time snapshots on each PINN path',
                  numerical_simulations_modified=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    caption='''# Validation total

**Caption.** Numerical flow comparison and selected particle trajectories. (a–c) Tangential velocity profiles at three channel sections. (d–f) PINN versus FEM velocity components and pressure, using all saved independent field-validation samples; pressure gauges are aligned by separate mean subtraction. Dashed diagonals indicate identity. (g–i) Termination-time parity, common-time coverage, and complete paired particle paths for four selected baseline inlet seeds, η₀ = −0.65, −0.4333, −0.2167, and 0. Colors match across these three panels. Solid trajectories are PINN + ForceNet and dashed trajectories are FEM + Stokes; dots mark three equally spaced physical times on each PINN path and arrows indicate motion. The dashed 95% coverage line is a visual reference.

**Scope.** Particle panels show 4 of 7 baseline seeds, selected explicitly as requested. The other three seeds (coverage 96.2%, 87.4%, and 73.6%) are omitted from this figure only. This selected illustration does not establish accuracy across all particle trajectories; the full comparison remains in the adjacent `paper_figures/` directory. Coverage measures common integration duration, not position accuracy. Selected synchronous RMS errors span 2.29–7.82% of channel width; geometric RMS errors span 0.32–1.56%. No simulation data were changed or trajectories extrapolated. This is a numerical FEM comparison, not experimental validation.

نسخهٔ ترکیبی شامل همهٔ نقاط مقایسهٔ میدان و فقط چهار مسیر منتخب از هفت مسیر پایه است. سه مسیر با پوشش زمانی ۹۶٫۲، ۸۷٫۴ و ۷۳٫۶ درصد فقط از این شکل کنار گذاشته شده‌اند؛ نتایج کامل در مخزن موجود است. پوشش زمانی معیار دقت مکانی نیست. خطای زمانی چهار مسیر منتخب تا ۷٫۸۲٪ عرض کانال است.

Formats: vector PDF/SVG, 600 dpi PNG/TIFF. The plotted profile values and selected/excluded path metrics are in `data/`; input hashes and selection details are in `manifest.json`.

Reproduce from the repository root with a fresh output directory:

```bash
python make_validation_total.py --out results/validation_total_new
```
'''
    (out/'CAPTION.md').write_text(caption,encoding='utf-8')
    print(json.dumps(manifest,indent=2));print(out)


if __name__=='__main__':
    main()
