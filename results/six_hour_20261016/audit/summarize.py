import sys,json,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
REPO=next(p for p in Path(__file__).resolve().parents if (p/'run_parametric_pinn.py').exists())
sys.path.insert(0,str(REPO))
from particle_pinn.parametric import anchors
from particle_pinn.path_comparison import geometry_errors
from particle_pinn.advanced_exports import write_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent;root=Path(__file__).resolve().parent.parent/'final_output'
cache=torch.load(out/'plot_cache.pt',map_location='cpu',weights_only=False)
summary=[]
for name,case in anchors().items():
 for track in cache['tracks'][name]:
  p=track['pinn'];r=track['reference'];g=geometry_errors(p['xy'],r['xy'])
  summary.append(dict(case=name,particle=track['particle'],geometric_rms_over_width=g['geometric_rms']/(2*case.h),
   rms_position_at_same_time_over_width=track['rms_over_width'],common_time_coverage=track['common_time_coverage'],
   relative_termination_time_difference=abs(p['t'][-1]-r['t'][-1])/r['t'][-1],
   reference_outcome=r['outcome']))
write_json(out/'trajectory_quality.json',summary)
base=json.loads((out/'anchor_audit.json').read_text());gl=json.loads((out/'global_spot_checks.json').read_text());wide=json.loads((out/'wide_geometry_checks.json').read_text())
combined=[]
for row in gl:
 if row.get('reference_ok'):
  combined.append(dict(index=row['index'],velocity_error=row['velocity_error'],pressure_error=row['pressure_error'],reference='Spline 80x18'))
 else:
  rr=next(x for x in wide if x['index']==row['index'])
  combined.append(dict(index=row['index'],velocity_error=rr['velocity_error'],pressure_error=rr['pressure_error'],reference='Exact circle 128x30'))
write_json(out/'global_combined.json',combined)
fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained');colors=['#20908c','#35618d','#d98728']
x=np.arange(3);w=.34
axes[0].bar(x-w/2,[100*r['velocity_error_fine'] for r in base],w,label='Velocity',color='#35618d')
axes[0].bar(x+w/2,[100*r['pressure_error_fine'] for r in base],w,label='Pressure',color='#20908c')
axes[0].set(xticks=x,xticklabels=['Low','Base','High'],ylabel='Relative L2 error (%)',title='Anchor cases');axes[0].legend(frameon=False)
x=np.arange(len(combined))
axes[1].bar(x-w/2,[100*r['velocity_error'] for r in combined],w,label='Velocity',color='#35618d')
axes[1].bar(x+w/2,[100*r['pressure_error'] for r in combined],w,label='Pressure',color='#20908c')
axes[1].set(xticks=x,xticklabels=[str(i+1) for i in range(len(x))],xlabel='Additional parameter case',ylabel='Relative L2 error (%)',title='Parameter checks')
axes[1].legend(frameon=False)
for a in axes: a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
for ext in ['png','pdf']: fig.savefig(out/f'validation_audit.{ext}',dpi=300)
plt.close(fig)
for name in anchors():
 rows=[r for r in summary if r['case']==name]
 print(name,'geometry rms/width min/max',min(r['geometric_rms_over_width'] for r in rows),max(r['geometric_rms_over_width'] for r in rows),
 'same-time max',max(r['rms_position_at_same_time_over_width'] for r in rows))
