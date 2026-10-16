"""Create a fresh six-hour notebook with final-only scientific exports."""
import argparse,ast,json
from pathlib import Path
from build_notebook import build

def make(repo,out,owner,slug,hours=6.):
    settings=json.loads((repo/'advanced/settings.json').read_text())
    fingerprint=build(repo,out,owner,slug,settings)
    path=out/'advanced.ipynb';nb=json.loads(path.read_text())
    nb['cells'][0]['source']=['# Parametric particle PINN\n','Fresh six-hour GPU study. Full valid 0.1–10× parameter domain. Final results only.\n']
    code=''.join(nb['cells'][2]['source'])
    start=code.index('command=');end=code.index('\nif RESUME:',start)
    command="command=[sys.executable,'-u','run_timed_parametric_pinn.py','--device','cuda','--hours',"+repr(str(hours))+",'--out',str(RUN)]"
    code=code[:start]+command+code[end:]
    code=code.replace("if (RUN/'chunk_complete.json').exists(): print((RUN/'chunk_complete.json').read_text())", "if (RUN/'run_complete.json').exists(): print((RUN/'run_complete.json').read_text())\nfrom IPython.display import display,Image\nfor image_path in sorted((RUN/'snapshots').glob('step_*/three_cases.png')):\n    display(Image(filename=str(image_path)))\nif (RUN/'validation.png').exists(): display(Image(filename=str(RUN/'validation.png')))\nfor gif_path in sorted((RUN/'snapshots').glob('step_*/particle_motion.gif')):\n    display(Image(filename=str(gif_path)))")
    nb['cells'][2]['source']=code.splitlines(True)
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    path.write_text(json.dumps(nb,indent=1));return fingerprint

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--owner',default='mohrezsayeh');p.add_argument('--slug',required=True);a=p.parse_args()
    print(make(Path(__file__).resolve().parents[1],a.out,a.owner,a.slug))
