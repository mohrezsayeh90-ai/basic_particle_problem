#!/usr/bin/env python3
"""Publication layouts for existing trajectory comparisons; no simulation changes.

Exports journal-width PDF/SVG/600-dpi PNG/TIFF, a multi-page figure book,
captions, plotted data and an input/output integrity manifest.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/particle-paper-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/particle-paper-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
from particle_pinn.geometry import Channel
from particle_pinn.path_comparison import load_paths

BLUE='#0072B2'
ORANGE='#E69F00'
RED='#C44E32'
INK='#25313D'
GRAY='#77838F'
PAGE_WIDTH=178/25.4


def read_pairs(file):
    records=[]
    with file.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            for key,value in row.items():
                if value in ('True','False'):
                    row[key]=value=='True'
                else:
                    try: row[key]=float(value)
                    except ValueError: pass
            records.append(row)
    return records


def read_series(file,pairs):
    selected={int(r['pair']):[] for r in pairs}
    with file.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            index=int(row['pair'])
            if index in selected:
                selected[index].append({k:float(v) for k,v in row.items() if k not in ('comparison','case')})
    return {index:{k:np.array([r[k] for r in rows]) for k in rows[0]} for index,rows in selected.items()}


def csv_out(path,records):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)


def style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,
        'xtick.color':INK,'ytick.color':INK,'xtick.major.width':.7,'ytick.major.width':.7,
        'axes.titleweight':'medium','axes.titlepad':9,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'mathtext.fontset':'dejavusans','savefig.facecolor':'white','figure.facecolor':'white',
        'grid.color':'#DDE2E7','grid.linewidth':.55,'axes.axisbelow':True,
        'lines.solid_capstyle':'round','lines.dash_capstyle':'round'})


class PaperPlots:
    def __init__(self,baseline,series,tracks,channel,width):
        self.baseline=baseline;self.series=series;self.tracks=tracks
        self.channel=channel;self.width=width
        self.norm=Normalize(min(r['eta'] for r in baseline),max(r['eta'] for r in baseline))
        self.cmap=plt.get_cmap('viridis')
        self.pool={key:np.concatenate([series[int(r['pair'])][key] for r in baseline])
                   for key in ['reference_x','reference_y','predicted_x','predicted_y']}

    def seed_color(self,row): return self.cmap(self.norm(row['eta']))

    def parity(self,ax,component,title):
        ref=self.pool['reference_'+component];pred=self.pool['predicted_'+component]
        low=min(ref.min(),pred.min());high=max(ref.max(),pred.max())
        pad=.055*(high-low);low-=pad;high+=pad
        diagonal=np.array([low,high])
        ax.fill_between(diagonal,diagonal-.05*self.width,diagonal+.05*self.width,
                        color='#E7EBEF',zorder=0)
        for row in self.baseline:
            data=self.series[int(row['pair'])]
            ax.scatter(data['reference_'+component],data['predicted_'+component],s=3.1,
                       color=self.seed_color(row),alpha=.32,edgecolors='none',zorder=2)
        ax.plot(diagonal,diagonal,color=INK,lw=1.05,ls=(0,(5,3)),zorder=3)
        ax.set(xlim=(low,high),ylim=(low,high),aspect='equal',
               xlabel=rf'FEM ${component}/L_0$',ylabel=rf'PINN ${component}/L_0$')
        ax.set_title(title,loc='left')
        error=100*np.sqrt(np.mean((pred-ref)**2))/self.width
        ax.text(.96,.055,f'Coordinate RMSE / W\n{error:.2f}%',ha='right',va='bottom',
                transform=ax.transAxes,fontsize=7.5,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
        ax.grid(True,alpha=.45)

    def seed_bar(self,fig,axes):
        scalar=matplotlib.cm.ScalarMappable(norm=self.norm,cmap=self.cmap)
        bar=fig.colorbar(scalar,ax=axes,shrink=.82,pad=.025,aspect=27)
        bar.set_label(r'Inlet position $\eta_0$',fontsize=8)
        bar.set_ticks([-.65,0,.65]);bar.outline.set_linewidth(.6)

    def errors(self,ax,title,compact=False):
        x=np.arange(len(self.baseline));bar_width=.34
        first=ax.bar(x-bar_width/2,[r['time_rms_width_percent'] for r in self.baseline],
                     width=bar_width,color=BLUE,label='Synchronous position')
        second=ax.bar(x+bar_width/2,[r['geometric_rms_width_percent'] for r in self.baseline],
                      width=bar_width,color=ORANGE,label='Path geometry')
        for bars in (first,second):
            for bar,row in zip(bars,self.baseline):
                if not row['outcome_agreement']:
                    bar.set_edgecolor(RED);bar.set_linewidth(1.1);bar.set_hatch('///')
                value=bar.get_height()
                ax.annotate(f'{value:.1f}' if compact else f'{value:.2f}',
                            xy=(bar.get_x()+bar.get_width()/2,value),xytext=(0,3),
                            textcoords='offset points',ha='center',va='bottom',
                            fontsize=6.2 if compact else 8,color=INK)
        ax.axhline(5,color=GRAY,lw=.85,ls=(0,(2,2)),zorder=0)
        maximum=max(max(r['time_rms_width_percent'],r['geometric_rms_width_percent']) for r in self.baseline)
        ax.set(ylim=(0,maximum*1.24),xlabel=r'Inlet position $\eta_0$',ylabel=r'RMS distance / $W$ (%)')
        ax.set_xticks(x,[f"{r['eta']:+.2f}" for r in self.baseline])
        if compact: ax.tick_params(axis='x',labelsize=7)
        handles=[Patch(facecolor=BLUE,label='Synchronous position'),Patch(facecolor=ORANGE,label='Path geometry'),
                 Patch(facecolor='white',edgecolor=RED,hatch='///',label='Different endpoint')]
        ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=6.5 if compact else 8)
        ax.grid(axis='y',alpha=.45)
        ax.set_title(title,loc='left')

    def walls(self,ax):
        s=np.linspace(0,self.channel.length,500)
        for eta in [-1.,1.]:
            xy=self.channel.map(s,np.full(len(s),eta))
            ax.plot(*xy.T,c='#B5BFC8',lw=.8,zorder=0)
        ax.set(aspect='equal',xlabel=r'$x/L_0$',ylabel=r'$y/L_0$')

    def critical_path(self,ax,title,compact=False):
        row=next(r for r in self.baseline if not r['outcome_agreement'])
        pred=self.tracks[int(row['predicted_track'])];ref=self.tracks[int(row['reference_track'])]
        self.walls(ax)
        ax.plot(*ref.xy.T,c=INK,lw=1.25,ls=(0,(4,2)),label='FEM + Stokes')
        ax.plot(*pred.xy.T,c=BLUE,lw=1.45,label='PINN + ForceNet')
        ax.scatter(*pred.xy[-1],c=RED,marker='X',s=43,zorder=5,edgecolors='white',linewidths=.5)
        ax.scatter(*ref.xy[-1],facecolor='white',edgecolor=INK,s=30,zorder=5)
        ax.annotate('Sticking',pred.xy[-1],xytext=(-18,28),textcoords='offset points',
                    color=RED,fontsize=7.5,ha='right',
                    arrowprops=dict(arrowstyle='-',color=RED,lw=.8))
        ax.annotate('Exit',ref.xy[-1],xytext=(-3,20),textcoords='offset points',fontsize=7.5,ha='right',
                    arrowprops=dict(arrowstyle='-',color=INK,lw=.8))
        ax.set_title(title,loc='left')
        ax.legend(loc='lower left',frameon=False,fontsize=6.5 if compact else 8)
        ax.text(.99,.02,rf"$\eta_0={row['eta']:.2f}$"+'\n'+f"Common time: {100*row['common_time_coverage']:.1f}%",
                transform=ax.transAxes,ha='right',va='bottom',fontsize=6.6 if compact else 8,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
        ax.margins(x=.06,y=.17)


def export(fig,folder,name,book):
    # Constrained layout and fixed dimensions preserve journal-scale font sizes.
    fig.savefig(folder/f'{name}.pdf',metadata={'Title':name,'Author':'Mohammad Reza Sayah'})
    fig.savefig(folder/f'{name}.svg')
    fig.savefig(folder/f'{name}.png',dpi=600)
    fig.savefig(folder/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    book.savefig(fig)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/final_study'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    comparison=run/'trajectory_comparison';out=args.out or comparison/'paper_figures'
    if out.exists() and any(out.iterdir()): parser.error('Choose a new empty --out to preserve earlier exports.')
    pairs_file=comparison/'data/pairs.csv';series_file=comparison/'data/time_errors.csv'
    raw_file=run/'data/trajectories.csv';cfg_file=run/'config.json'
    rows=read_pairs(pairs_file)
    baseline=sorted([r for r in rows if r['comparison']=='Full model' and r['case']=='baseline'],key=lambda r:r['eta'])
    if len(baseline)!=7: raise ValueError('This publication layout expects the seven stored baseline seeds.')
    if any(r['status']!='compared' for r in baseline): raise ValueError('Unavailable baseline path comparison.')
    if sum(not r['outcome_agreement'] for r in baseline)!=1:
        raise ValueError('Review the critical-path panel: the baseline endpoint pattern has changed.')
    series=read_series(series_file,baseline)
    paths={t.track:t for t in load_paths(raw_file)}
    cfg=json.loads(cfg_file.read_text())
    custom=run/'data/centerline.csv'
    channel=Channel(cfg['geometry'],cfg['rin'],cfg['rout'],str(custom) if custom.exists() else cfg.get('centerline'))
    width=2*channel.half_width
    out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    style();plots=PaperPlots(baseline,series,paths,channel,width)
    rendered=[]
    with PdfPages(out/'Particle_Trajectory_Figures.pdf',metadata={'Title':'Particle trajectory comparison'}) as book:
        fig,axes=plt.subplots(2,2,figsize=(PAGE_WIDTH,6.05),layout='constrained')
        plots.parity(axes[0,0],'x','(a) Horizontal position')
        plots.parity(axes[0,1],'y','(b) Vertical position')
        plots.seed_bar(fig,[axes[0,0],axes[0,1]])
        plots.errors(axes[1,0],'(c) Trajectory errors',compact=True)
        plots.critical_path(axes[1,1],'(d) Endpoint comparison',compact=True)
        name='Fig01_trajectory_comparison';export(fig,out,name,book);rendered.append(name)

        fig,axes=plt.subplots(1,2,figsize=(PAGE_WIDTH,3.25),layout='constrained')
        plots.parity(axes[0],'x','(a) Horizontal position')
        plots.parity(axes[1],'y','(b) Vertical position')
        plots.seed_bar(fig,list(axes))
        name='Fig02_position_parity';export(fig,out,name,book);rendered.append(name)

        fig,ax=plt.subplots(figsize=(PAGE_WIDTH,3.4),layout='constrained')
        plots.errors(ax,'Trajectory errors')
        name='Fig03_error_bars';export(fig,out,name,book);rendered.append(name)

        fig,axes=plt.subplots(1,2,figsize=(PAGE_WIDTH,3.1),layout='constrained')
        ax=axes[0];ends=[max(r['reference_end_time'],r['predicted_end_time']) for r in baseline]
        top=1.08*max(ends);ax.plot([0,top],[0,top],c=INK,ls=(0,(5,3)),lw=1)
        label_offsets=[(-35,28),(-40,-25),(8,-25),(15,6),(8,8),(8,8),(8,8)]
        for row,offset in zip(baseline,label_offsets):
            match=row['outcome_agreement']
            ax.scatter(row['reference_end_time'],row['predicted_end_time'],
                       color=plots.seed_color(row) if match else RED,marker='o' if match else 'X',s=45,
                       edgecolors='white',linewidths=.5,zorder=3)
            ax.annotate(f"{row['eta']:+.2f}",(row['reference_end_time'],row['predicted_end_time']),
                        xytext=offset,textcoords='offset points',fontsize=6.5,
                        arrowprops=dict(arrowstyle='-',color=GRAY,lw=.45,shrinkA=2,shrinkB=4))
        ax.set(xlim=(0,top),ylim=(0,top),aspect='equal',xlabel=r'FEM termination time / $T_0$',
               ylabel=r'PINN termination time / $T_0$')
        ax.set_title('(a) Termination timing',loc='left');ax.grid(alpha=.45)
        ax=axes[1];x=np.arange(7)
        bars=ax.bar(x,[100*r['common_time_coverage'] for r in baseline],width=.64,
                    color=[plots.seed_color(r) for r in baseline])
        for bar,row in zip(bars,baseline):
            if not row['outcome_agreement']: bar.set_hatch('///');bar.set_edgecolor(RED)
            ax.annotate(f"{100*row['common_time_coverage']:.1f}",
                        (bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),
                        textcoords='offset points',ha='center',fontsize=6.8)
        ax.axhline(95,c=GRAY,lw=.8,ls='--');ax.set(ylim=(0,113),xlabel=r'Inlet position $\eta_0$',ylabel='Common-time coverage (%)')
        ax.set_xticks(x,[f"{r['eta']:+.2f}" for r in baseline],rotation=35)
        ax.set_title('(b) Comparison coverage',loc='left');ax.grid(axis='y',alpha=.45)
        name='Fig04_timing_and_coverage';export(fig,out,name,book);rendered.append(name)

        fig,axes=plt.subplots(1,2,figsize=(PAGE_WIDTH,3.05),layout='constrained')
        plots.walls(axes[0])
        for row in baseline:
            a,b=paths[int(row['predicted_track'])],paths[int(row['reference_track'])]
            axes[0].plot(*b.xy.T,c=plots.seed_color(row),lw=.95,ls=(0,(4,2)))
            axes[0].plot(*a.xy.T,c=plots.seed_color(row),lw=1.1)
            if not row['outcome_agreement']: axes[0].scatter(*a.xy[-1],s=30,marker='X',c=RED,zorder=5)
        axes[0].set_title('(a) Paired baseline paths',loc='left')
        axes[0].legend(handles=[Line2D([0],[0],c=INK,lw=1.2,label='PINN + ForceNet'),
                              Line2D([0],[0],c=INK,ls='--',lw=1.2,label='FEM + Stokes')],
                       loc='lower left',frameon=False,fontsize=7)
        plots.critical_path(axes[1],'(b) Endpoint comparison')
        name='Fig05_paired_paths';export(fig,out,name,book);rendered.append(name)

        full=[r for r in rows if r['comparison']=='Full model']
        unique={}
        for r in full:
            unique.setdefault((round(r['eta'],12),round(r['diameter'],12),round(r['density_ratio'],8)),r)
        unique_rows=list(unique.values());cases=list(dict.fromkeys(r['case'] for r in unique_rows))
        eta=sorted(set(r['eta'] for r in unique_rows))
        labels=[]
        for case in cases:
            if case=='baseline': labels.append('Baseline')
            elif case.startswith('diameter_'): labels.append('$d_p='+case.split('_',1)[1]+'$')
            else: labels.append(r'$\rho_r='+case.split('_',1)[1]+'$')
        matrices=[];flags=[]
        for key in ['time_rms_width_percent','geometric_rms_width_percent']:
            matrix=np.empty((len(eta),len(cases)));mismatch=np.zeros_like(matrix,dtype=bool)
            for i,value in enumerate(eta):
                for j,case in enumerate(cases):
                    row=next(r for r in unique_rows if r['case']==case and abs(r['eta']-value)<1e-10)
                    matrix[i,j]=row[key];mismatch[i,j]=not row['outcome_agreement']
            matrices.append(matrix);flags.append(mismatch)
        fig,axes=plt.subplots(1,2,figsize=(PAGE_WIDTH,3.9),layout='constrained')
        maximum=np.ceil(max(m.max() for m in matrices)/10)*10
        for ax,matrix,mismatch,title in zip(axes,matrices,flags,['(a) Synchronous error','(b) Geometric error']):
            artist=ax.imshow(matrix,cmap='cividis',vmin=0,vmax=maximum,aspect='auto')
            for (i,j),value in np.ndenumerate(matrix):
                ax.text(j,i,f'{value:.1f}'+('*' if mismatch[i,j] else ''),ha='center',va='center',
                        fontsize=7.2,color='white' if value<.48*maximum else INK)
                if mismatch[i,j]: ax.add_patch(Rectangle((j-.49,i-.49),.98,.98,fill=False,ec=RED,lw=1.5))
            ax.set_xticks(range(len(cases)),labels,rotation=45,ha='right',fontsize=7.5)
            ax.set_yticks(range(len(eta)),[f'{e:+.2f}' for e in eta]);ax.set_ylabel(r'Inlet position $\eta_0$')
            ax.set_title(title,loc='left')
        fig.colorbar(artist,ax=list(axes),shrink=.9,pad=.025,label=r'RMS distance / $W$ (%)')
        fig.supxlabel('* Different endpoint class; repeated baseline control omitted',fontsize=7.3)
        name='Fig06_case_heatmaps';export(fig,out,name,book);rendered.append(name)

        fig,ax=plt.subplots(figsize=(PAGE_WIDTH,3.15),layout='constrained')
        for row in baseline:
            data=series[int(row['pair'])]
            ax.plot(data['t'],data['distance_width_percent'],c=plots.seed_color(row),lw=1.3,
                    label=rf"$\eta_0={row['eta']:+.2f}$")
            if not row['outcome_agreement']:
                ax.scatter(data['t'][-1],data['distance_width_percent'][-1],marker='X',s=35,c=RED,zorder=4)
        ax.axhline(5,c=GRAY,lw=.8,ls=(0,(2,2)),label='5% W reference')
        ax.set(xlabel=r'Physical time / $T_0$',ylabel=r'Position error / $W$ (%)',ylim=(0,None))
        ax.set_title('Trajectory error history',loc='left');ax.grid(alpha=.4)
        ax.legend(frameon=False,ncol=2,loc='upper left',fontsize=7.4)
        name='Fig07_error_history';export(fig,out,name,book);rendered.append(name)

    selected_keys=['eta','time_rms_width_percent','geometric_rms_width_percent','common_time_coverage',
                   'predicted_end_time','reference_end_time','predicted_outcome','reference_outcome','outcome_agreement']
    csv_out(out/'data/baseline_metrics.csv',[{k:r[k] for k in selected_keys} for r in baseline])
    csv_out(out/'data/unique_case_metrics.csv',unique_rows)
    parity_rows=[]
    for row in baseline:
        data=series[int(row['pair'])]
        for i,t in enumerate(data['t']):
            parity_rows.append(dict(eta=row['eta'],t=t,reference_x=data['reference_x'][i],predicted_x=data['predicted_x'][i],
                                    reference_y=data['reference_y'][i],predicted_y=data['predicted_y'][i]))
    csv_out(out/'data/parity_points.csv',parity_rows)
    inputs=[pairs_file,series_file,raw_file,cfg_file]
    manifest=dict(figure_width_mm=178,raster_dpi=600,figure_count=len(rendered),figures=rendered,
                  plotted_baseline_paths=len(baseline),parity_point_pairs=len(parity_rows),
                  unique_particle_pairs=len(unique_rows),omitted_duplicate_controls=len(full)-len(unique_rows),
                  mismatched_baseline_endpoints=sum(not r['outcome_agreement'] for r in baseline),
                  input_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  numerical_data_modified=False,
                  parity_metrics={component:dict(coordinate_rmse_over_width_percent=float(100*np.sqrt(np.mean(
                      (plots.pool['predicted_'+component]-plots.pool['reference_'+component])**2))/width)) for component in ['x','y']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    make_captions(out,manifest,baseline)
    print(json.dumps(manifest,indent=2))
    print('Article figures:',out)


def make_captions(out,manifest,baseline):
    captions=[
        ('Fig01_trajectory_comparison','Trajectory comparison',
         'Baseline particle comparison between PINN + ForceNet and the independent FEM + Stokes reference. '
         '(a,b) Axial and transverse positions paired at the same physical times within each path pair’s common interval. '
         'Dashed lines indicate identity; gray bands are ±5% of channel width W, not confidence intervals. '
         '(c) Synchronous-position and symmetric geometric RMS errors, normalized by W. Hatched bars indicate different endpoint classes. '
         '(d) The η₀ = 0.65 case retains the complete reference path after the predicted sticking event. '
         'Points are temporally correlated samples from seven trajectories, not independent experimental observations.'),
        ('Fig02_position_parity','Position parity',
         'PINN/FEM position parity for seven baseline inlet seeds (1,001 matched times per path). Color denotes inlet position. '
         'Coordinate RMSE is pooled over these sampled coordinate pairs and differs from the time-integrated vector RMS. '
         'All plotted samples and common-time limits are retained; no time stretching or post-termination extrapolation is used. '
         'Identity lines and ±5% W reference corridors are shown.'),
        ('Fig03_error_bars','Trajectory error',
         'Baseline RMS errors in synchronous position and full-path geometry. All values are percentages of channel width. '
         'The two metrics answer different questions; the geometric metric ignores time ordering. '
         'Hatching identifies the sticking/exit disagreement. The dotted 5% W line is a scale/reference line, not a universal engineering tolerance.'),
        ('Fig04_timing_and_coverage','Timing and coverage',
         'Baseline termination-time parity and common-time coverage. The red cross and hatched bar indicate different terminal events. '
         'Termination times for different event classes are displayed but must not be interpreted as equivalent outlet residence times. '
         'The 95% coverage line provides a visual reference.'),
        ('Fig05_paired_paths','Paired particle paths',
         '(a) Full baseline paths, with matched seed colors and solid PINN / dashed FEM lines. '
         '(b) Enlarged endpoint interpretation for the η₀ = 0.65 case. The red marker identifies predicted sticking; the hollow marker identifies reference exit. '
         'Complete stored paths are shown, including the reference portion outside the common-time interval.'),
        ('Fig06_case_heatmaps','Case comparison',
         'Synchronous and geometric RMS errors for 42 unique parameter/seed pairs. '
         'The density-ratio-1000 control duplicates the baseline and is omitted once (seven repeated pairs). '
         'Each cell is one deterministic inlet seed; stars and red borders identify endpoint-class disagreements. '
         'Both panels share the same linear color scale. Diameter and density ratio are nondimensional.'),
        ('Fig07_error_history','Position error history',
         'Synchronous Euclidean position error divided by width W for each baseline trajectory pair. '
         'Every curve stops at the end of its common-time interval. The red cross marks the pair with different endpoint classes. '
         'The dotted 5% W line is a reference level; no prediction is extrapolated after termination.')]
    lines=['# Article figure pack','',
           '**Recommended main figure: Fig01_trajectory_comparison.pdf.** Standalone panels/alternatives follow as Fig02–Fig07.','',
           'All figures are 178 mm wide, with editable text in SVG and embedded TrueType fonts in PDF. '
           'PNG and LZW-compressed TIFF are exported at 600 dpi. A seven-page PDF figure book is included.','',
           'These are numerical PINN/FEM comparisons, not experimental validation. The saved numerical trajectories and errors were not changed. '
           'The input hashes are in manifest.json.','',
           '## Captions','']
    for name,title,caption in captions:
        lines += [f'### {name}: {title}',caption,'']
    lines += ['## Data and interpretation','',
              '- `data/baseline_metrics.csv`: the seven bar/timing records, copied from the saved comparison table.',
              '- `data/parity_points.csv`: every coordinate pair used in the parity panels.',
              '- `data/unique_case_metrics.csv`: the 42 nonduplicate case/seed records.',
              '- Error percentages divide by channel width W = 0.3; they are not coordinate-relative percentages.',
              '- All 7 baseline paths are included, including the mismatched sticking/exit case.',
              '- Synchronous RMS uses the previously computed time integral on the common interval; geometric RMS uses both complete path polylines.',
              '- A narrow parity corridor or a small average field error alone does not certify all particle trajectories.',
              '- Error bars are intentionally absent: inlet-seed variability is not a measured statistical confidence interval.','']
    (out/'CAPTIONS.md').write_text('\n'.join(lines),encoding='utf-8')
    snippet=r'''% Requires \usepackage{graphicx}. Copy this figure folder into the article.
% Choose the vector PDF for publication; use CAPTIONS.md for the full caption.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Fig01_trajectory_comparison.pdf}
\caption{Trajectory comparison between PINN + ForceNet and FEM + Stokes.
(a,b) Position parity at common physical times; dashed lines indicate identity.
(c) Synchronous and geometric RMS errors normalized by channel width.
(d) Complete paths for the sticking/exit disagreement.
Gray corridors show $\pm5\%$ of channel width, not confidence intervals.
Hatched bars identify different endpoint classes.}
\label{fig:particle-trajectory-comparison}
\end{figure*}
'''
    (out/'article_figure.tex').write_text(snippet,encoding='utf-8')
    (out/'README_FA.md').write_text('''# شکل‌های آمادهٔ استفاده در مقاله

شکل پیشنهادی اصلی **Fig01_trajectory_comparison.pdf** است: دو نمودار نقطه‌ایِ مختصات PINN در برابر FEM، نمودار میله‌ای خطا و مسیر دارای اختلاف برخورد را کنار هم نشان می‌دهد.

- Fig02: نمودار نقطه‌ای مقایسهٔ موقعیت‌ها؛ خط‌چین یعنی تطابق کامل.
- Fig03: نمودار میله‌ای خطای زمانی و هندسی هر هفت مسیر پایه.
- Fig04: مقایسهٔ زمان پایان و درصد پوشش زمانی.
- Fig05: مسیرهای جفت‌شده و نمایش واضح اختلاف چسبیدن/خروج.
- Fig06: نقشهٔ خطای حالت‌های مختلف؛ کنترل تکراری حذف شده است.
- Fig07: تغییر خطای مکانی در زمان.

هر شکل در چهار قالب PDF، SVG، PNG و TIFF موجود است. PDF برای مقاله و SVG برای ویرایش مناسب‌اند؛ تصاویر PNG/TIFF با ۶۰۰ dpi ذخیره شده‌اند. فایل Particle_Trajectory_Figures.pdf همهٔ شکل‌ها را یکجا دارد.

متن کپشن انگلیسی در CAPTIONS.md و کد درج شکل اصلی در article_figure.tex است. داده‌های رسم‌شده در data/ قرار دارند. مقدارها، نقاط دور از خط تطابق و مسیر دارای برخورد متفاوت حذف یا دست‌کاری نشده‌اند. ناحیهٔ خاکستریِ نمودار نقطه‌ای فقط محدودهٔ مرجع ±۵٪ عرض کانال است و فاصلهٔ اطمینان آماری نیست.

برای بازتولید، از ریشهٔ پروژه و با یک پوشهٔ خروجی تازه اجرا کنید:

```bash
python make_publication_figures.py --out results/new_paper_figures
```
''',encoding='utf-8')


if __name__=='__main__':
    main()
