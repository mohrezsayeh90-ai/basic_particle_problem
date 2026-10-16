#!/usr/bin/env python3
"""Compare existing PINN/FEM paths and export dedicated accuracy reports/figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from particle_pinn.geometry import Channel
from particle_pinn.path_comparison import load_paths,pair_paths,compare_pair


def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n')
        writer.writeheader();writer.writerows(clean(rows))


def draw_figures(folder,channel,pairs,rows,series):
    from particle_pinn.figures import plt,save,walls,COLORS
    baseline=[r for r in rows if r['comparison']=='Full model' and r['case']=='baseline']
    lookup={p.track:p for pair in pairs for p in pair}
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for ax,row in zip(axes.flat,baseline):
        a,b=lookup[row['predicted_track']],lookup[row['reference_track']]
        walls(ax,channel)
        ax.plot(*b.xy.T,'k--',lw=1.5,label='FEM + Stokes')
        ax.plot(*a.xy.T,color=COLORS[0],lw=1.1,label='PINN + ForceNet')
        ax.scatter(*a.xy[-1],marker='x' if a.outcome=='stuck' else 'o',color=COLORS[0],s=25)
        ax.set_title(f"Inlet n / h = {row['eta']:.3f}")
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center')
    fig.suptitle('Paired particle paths')
    save(fig,folder,'23_paired_paths')

    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for i,row in enumerate(baseline):
        data=series[row['pair']]
        label=f"n/h={row['eta']:.3f}"
        axes[0].plot(data['t'],100*data['distance']/(2*channel.half_width),c=COLORS[i],label=label)
        axes[1].plot(data['t'],100*np.abs(data['normal'])/(2*channel.half_width),c=COLORS[i],label=label)
    for ax,title in zip(axes,['Position error','Normal error']):
        ax.axhline(5,c='.4',ls=':',label='5% reference line')
        ax.set(title=title,xlabel='Time / T₀',ylabel='Distance / width (%)')
    axes[0].legend(fontsize=8,ncol=2)
    save(fig,folder,'24_error_history')

    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(len(baseline))
    for j,(key,label) in enumerate([('time_rms_width_percent','Synchronous RMS'),
                                   ('tangential_rms_width_percent','Tangential RMS'),
                                   ('normal_rms_width_percent','Normal RMS'),
                                   ('geometric_rms_width_percent','Geometric RMS')]):
        ax.bar(x+(j-1.5)*.19,[r[key] for r in baseline],width=.19,label=label)
    ax.axhline(5,ls=':',c='k')
    ax.set_xticks(x,[f"{r['eta']:.3f}" for r in baseline])
    ax.set(title='Trajectory error components',xlabel='Inlet n / h',ylabel='RMS / width (%)')
    ax.legend(fontsize=8,ncol=2)
    save(fig,folder,'25_error_components')

    full=[r for r in rows if r['comparison']=='Full model' and r['status']=='compared']
    cases=list(dict.fromkeys(r['case'] for r in full))
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,key,title in zip(axes,['time_rms_width_percent','geometric_rms_width_percent'],
                            ['Synchronous path error','Geometric path error']):
        for i,case in enumerate(cases):
            selected=[r for r in full if r['case']==case]
            for row in selected:
                ax.scatter(i+.18*row['eta'],row[key],c=[COLORS[i%10]],
                           marker='o' if row['outcome_agreement'] else 'x',s=30)
        ax.set_xticks(range(len(cases)),[c.replace('_','\n',1) for c in cases],rotation=20,fontsize=8)
        ax.set(title=title,ylabel='RMS / width (%)')
        if key=='time_rms_width_percent': ax.axhline(5,c='k',ls=':')
    fig.text(.5,.01,'Circles: matching endpoint classes · Crosses: different endpoint classes',ha='center',fontsize=9)
    save(fig,folder,'26_case_errors')

    outcomes=['exited','stuck','backflow_exit','time_limit','integration_failed']
    present=[x for x in outcomes if any(r['predicted_outcome']==x or r['reference_outcome']==x for r in full)]
    matrix=np.zeros((len(present),len(present)),dtype=int)
    for row in full: matrix[present.index(row['reference_outcome']),present.index(row['predicted_outcome'])]+=1
    fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(matrix,cmap='Blues',vmin=0)
    for (i,j),value in np.ndenumerate(matrix):
        ax.text(j,i,str(value),ha='center',va='center',fontsize=17,color='white' if value>matrix.max()/2 else 'black')
    ax.set_xticks(range(len(present)),present);ax.set_yticks(range(len(present)),present)
    ax.set(title='Particle endpoint agreement',xlabel='PINN + ForceNet',ylabel='FEM + Stokes')
    save(fig,folder,'27_endpoint_confusion')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for i,case in enumerate(cases):
        for row in [r for r in full if r['case']==case]:
            axes[0].scatter(row['reference_end_time'],row['predicted_end_time'],c=[COLORS[i%10]],
                            marker='o' if row['outcome_agreement'] else 'x',s=25)
    end=max(max(r['reference_end_time'],r['predicted_end_time']) for r in full)
    axes[0].plot([0,end],[0,end],'k--',lw=1)
    axes[0].set(title='Termination times',xlabel='FEM time / T₀',ylabel='PINN time / T₀')
    axes[1].bar(range(len(baseline)),[100*r['common_time_coverage'] for r in baseline])
    axes[1].axhline(95,c='k',ls=':')
    axes[1].set_xticks(range(len(baseline)),[f"{r['eta']:.3f}" for r in baseline],rotation=30)
    axes[1].set(title='Common-time coverage',xlabel='Inlet n / h',ylabel='Coverage (%)',ylim=(0,105))
    save(fig,folder,'28_timing_and_coverage')


def make_report(out,rows,audit,width):
    baseline=[r for r in rows if r['comparison']=='Full model' and r['case']=='baseline']
    full=[r for r in rows if r['comparison']=='Full model' and r['status']=='compared']
    lines=['# مقایسهٔ مستقل مسیرهای ذخیره‌شده','',
           'خطای مسیرهای پایه قبلاً در metrics.json محاسبه شده بود. در این تحلیل، همان اعداد از CSV خام بازتولید و با مقادیر قبلی تطبیق داده شدند؛ سپس مقایسه به همهٔ حالت‌های موجود گسترش یافت. هیچ مسیر تازه‌ای حل یا جایگزین نشده است.','',
           f"تطبیق با معیار قبلی: {audit['matched']} از {audit['expected']} مسیر؛ بیشترین اختلاف عددی RMS = {audit['maximum_absolute_difference']:.3g}.",
           '', '## تعریف خطاها','',
           f'عرض بی‌بُعد کانال W = {width:g} است. درصد خطای مکانی در این گزارش از تقسیم فاصله بر **عرض کانال** و ضرب در ۱۰۰ ساخته می‌شود؛ درصد خطای مختصات x یا y نیست.','',
           '- **خطای هم‌زمان:** فاصلهٔ دو ذره در یک زمان فیزیکی مشترک. RMS جدید با انتگرال زمانی روی مربع فاصله محاسبه شده است.',
           '- **معیار قبلی:** میانگین مربعی ۱۵۰ نمونهٔ زمانی، با وزن یکسان حتی در دو انتها. ستون legacy_sample_rms_150 آن را دقیقاً نگه می‌دارد. اختلاف جزئی آن با RMS انتگرالی جدید ناشی از تعریف وزن‌دهی است.',
           '- **خطای هندسی:** فاصلهٔ نقاط مسیر از نزدیک‌ترین پاره‌خط مسیر دیگر، در هر دو جهت و با نمونه‌برداری یکنواخت طول قوس. این معیار شکل مسیر را بدون الزام به رسیدن هم‌زمان مقایسه می‌کند.',
           '- **مؤلفه‌های مماسی و نرمال:** تصویر اختلاف مکانیِ هم‌زمان بر جهت مماس/نرمال خط مرکزی در محل ذرهٔ مرجع؛ مؤلفهٔ مماسی می‌تواند ناشی از اختلاف پیشروی یا زمان‌بندی باشد.',
           '- **پوشش زمانی:** طول بازهٔ مشترک تقسیم بر بازهٔ طولانی‌تر. پس از پایان یکی از مسیرها، برون‌یابی یا ثابت‌نگه‌داشتن مصنوعی موقعیت انجام نشده است.',
           '- **فاصلهٔ نقاط انتهایی:** دو موقعیت نهایی ممکن است مربوط به دو زمان یا دو رویداد متفاوت باشند؛ این فاصله جایگزین خطای هم‌زمان نیست.',
           '- ستون sampled_hausdorff فاصلهٔ بیشینه در نمونه‌های هندسی است؛ ادعای محاسبهٔ دقیق فاصلهٔ هاسدورف پیوسته نیست.','',
           '## مسیرهای پایه: PINN + ForceNet در برابر FEM + Stokes','',
           '| ورودی η | RMS هم‌زمان / عرض (%) | بیشینه / عرض (%) | RMS هندسی / عرض (%) | پوشش زمان (%) | نتیجهٔ PINN / FEM |',
           '|---:|---:|---:|---:|---:|---|']
    for r in baseline:
        lines.append(f"| {r['eta']:.3f} | {r['time_rms_width_percent']:.3f} | {r['max_time_error_width_percent']:.3f} | {r['geometric_rms_width_percent']:.3f} | {100*r['common_time_coverage']:.2f} | {r['predicted_outcome']} / {r['reference_outcome']} |")
    lines+=['','## همهٔ حالت‌های ذره','',
            '| حالت | تعداد جفت | توافق سرنوشت | میانگین RMS هم‌زمان / عرض (%) | بیشترین RMS / عرض (%) |',
            '|---|---:|---:|---:|---:|']
    for case in dict.fromkeys(r['case'] for r in full):
        group=[r for r in full if r['case']==case]
        lines.append(f"| {case} | {len(group)} | {sum(r['outcome_agreement'] for r in group)}/{len(group)} | {np.mean([r['time_rms_width_percent'] for r in group]):.3f} | {max(r['time_rms_width_percent'] for r in group):.3f} |")
    unique={}
    for r in full:
        unique.setdefault((round(r['eta'],12),round(r['diameter'],12),round(r['density_ratio'],8)),r)
    lines += ['',f"در کل {len(full)} جفت ثبت‌شده، توافق سرنوشت {sum(r['outcome_agreement'] for r in full)}/{len(full)} است. baseline و density_1000 یک کنترل تکراری‌اند؛ پس از حذف تکرار پارامترهای یکسان، {len(unique)} جفت یکتا و توافق {sum(r['outcome_agreement'] for r in unique.values())}/{len(unique)} داریم. این نسبت‌ها نمونهٔ مستقل آزمایشگاهی یا برآورد احتمال عمومی رسوب نیستند.",
              '', '## تفکیک منبع خطا','']
    for comparison in ['Flow only','Force only','Fluid tracers']:
        group=[r for r in rows if r['comparison']==comparison and r['status']=='compared']
        lines.append(f"- {comparison}: تعداد {len(group)}؛ توافق سرنوشت {sum(r['outcome_agreement'] for r in group)}/{len(group)}؛ بیشترین RMS هم‌زمان {max(r['time_rms_width_percent'] for r in group):.3f}٪ عرض.")
    lines+=['','مقایسهٔ Force only از یک میدان جریان PINN مشترک استفاده می‌کند؛ بنابراین اختلاف آن فقط به جایگزینی درگ دقیق با شبکهٔ نیرو مربوط است. این تفکیک به تشخیص سهم خطای جریان کمک می‌کند، ولی علت دقیق موضعی را اثبات نمی‌کند.','',
            '## حدود اعتبار','',
            'مرجع، مسیر محاسبه‌شده با میدان FEM مستقل و قانون درگ استوکس است؛ دادهٔ تجربی نیست. مقایسه بر پاره‌خط‌های ساخته‌شده از ۲۶۰ نمونهٔ ذخیره‌شدهٔ هر مسیر انجام شده است. دو برابر کردن تعداد نقاط محاسبهٔ معیار، فقط حساسیت انتگرال‌گیریِ همین مسیرهای ذخیره‌شده را می‌سنجد؛ استقلال از گام زمانی حل‌گر ODE یا دقت رویداد برخورد را اثبات نمی‌کند.',
            '', 'مسیر هندسی نزدیک می‌تواند با خطای زمانی بزرگ همراه باشد. از سوی دیگر، در مسیری که یکی زودتر به دیواره می‌چسبد، توافق شکل جزئی نباید به‌عنوان صحت کامل مسیر گزارش شود. نتیجهٔ نهایی باید هم‌زمان با خطای مکان، پوشش زمانی و سرنوشت ذره خوانده شود.','',
            '## فایل‌ها','',
            '- `data/pairs.csv`: همهٔ معیارها، شناسهٔ دو مسیر و نتیجهٔ رویدادها.',
            '- `data/time_errors.csv`: زمان، دو موقعیت و خطای هم‌زمان برای هر جفت.',
            '- `summary.json`: تعداد مقایسه‌ها، تطبیق با اعداد قبلی و هش ورودی‌ها.',
            '- `figures/23` تا `figures/28`: تطبیق مسیر، خطای زمانی، مؤلفه‌ها، حالت‌ها، سرنوشت و زمان پایان، در قالب PNG و PDF.','']
    (out/'REPORT_FA.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/final_study'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    out=args.out or run/'trajectory_comparison'
    if out.exists() and any(out.iterdir()): parser.error('Choose a new empty --out directory.')
    cfg=json.loads((run/'config.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    tracks=load_paths(run/'data/trajectories.csv')
    summary=json.loads((run/'data/trajectory_summary.json').read_text())
    assert len(tracks)==len(summary)==metrics['trajectory']['total']
    custom=run/'data/centerline.csv'
    channel=Channel(cfg['geometry'],cfg['rin'],cfg['rout'],str(custom) if custom.exists() else cfg.get('centerline'))
    width=2*channel.half_width
    groups=[('Full model','PINN + ForceNet','FEM + Stokes',None),
            ('Flow only','PINN + Stokes','FEM + Stokes','baseline'),
            ('Force only','PINN + ForceNet','PINN + Stokes','baseline'),
            ('Fluid tracers','PINN tracer','FEM tracer','tracer')]
    rows=[];series={};all_pairs=[]
    for name,source,reference,case in groups:
        pairs=pair_paths([t for t in tracks if case is None or t.case==case],source,reference)
        for a,b in pairs:
            row,data=compare_pair(a,b,width,channel)
            index=len(rows);row.update(pair=index,comparison=name)
            rows.append(row);series[index]=data;all_pairs.append((a,b))
    if any(r['status']!='compared' for r in rows):
        raise ValueError('Some stored paths cannot be compared; inspect their failure records.')
    old=metrics['trajectory']['baseline_comparison'];differences=[]
    for r in rows:
        if r['comparison']=='Full model' and r['case']=='baseline':
            previous=next(x for x in old if abs(x['seed_eta']-r['eta'])<1e-10)
            differences.append(abs(previous['rms_position_error']-r['legacy_sample_rms_150']))
            assert previous['outcome_agreement']==r['outcome_agreement']
    audit=dict(expected=len(old),matched=sum(d<1e-10 for d in differences),maximum_absolute_difference=max(differences))
    if audit['matched']!=audit['expected']: raise ValueError('Previous baseline metrics were not reproduced.')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['data','figures']: (out/name).mkdir(exist_ok=True)
    write_csv(out/'data/pairs.csv',rows)
    records=[]
    for row in rows:
        data=series[row['pair']]
        for t,p,r,d,along,across in zip(data['t'],data['predicted'],data['reference'],data['distance'],data['tangential'],data['normal']):
            records.append(dict(pair=row['pair'],comparison=row['comparison'],case=row['case'],eta=row['eta'],
                                t=t,predicted_x=p[0],predicted_y=p[1],reference_x=r[0],reference_y=r[1],
                                distance=d,distance_width_percent=100*d/width,tangential=along,normal=across))
    write_csv(out/'data/time_errors.csv',records)
    draw_figures(out/'figures',channel,all_pairs,rows,series)
    inputs=[run/'config.json',run/'metrics.json',run/'data/trajectories.csv',run/'data/trajectory_summary.json']
    report=dict(pairs=len(rows),group_counts={name:sum(r['comparison']==name for r in rows) for name,*_ in groups},
                previous_metric_audit=audit,width=width,numerical_trajectories_modified=False,
                maximum_metric_quadrature_change_width_percent=max(r['metric_quadrature_change_width_percent'] for r in rows),
                comparison_time_samples=1001,quadrature_check_samples=2001,geometric_samples_per_direction=501,
                source_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    write_json(out/'summary.json',report)
    make_report(out,rows,audit,width)
    print(json.dumps(clean(report),indent=2))
    print('Report:',out/'REPORT_FA.md')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
