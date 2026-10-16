#!/usr/bin/env python3
"""Build a private Kaggle notebook with continuous training and live exports."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import zlib


def source_payload(repo):
    files=sorted(set([*repo.glob('*.py'),*repo.glob('particle_pinn/*.py'),*repo.glob('tests/*.py'),*repo.glob('advanced/*.py'),repo/'requirements.txt']))
    blob=io.BytesIO();hashes={}
    with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name=str(p.relative_to(repo));b=p.read_bytes();z.writestr(name,b);hashes[name]=hashlib.sha256(b).hexdigest()
    # Transport/controller edits must not invalidate unchanged numerical weights.
    training={n:v for n,v in hashes.items() if not n.startswith(('advanced/','tests/'))}
    fingerprint=hashlib.sha256(json.dumps(training,sort_keys=True).encode()).hexdigest()
    return blob.getvalue(),fingerprint,hashes


def build(repo,out,owner,slug,settings,resume=None,expected_source=None,resume_url=None,live_dataset=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    payload,fingerprint,hashes=source_payload(repo)
    if expected_source and fingerprint!=expected_source: raise RuntimeError('Training source changed; start a new run instead of silently changing a resumed study.')
    encoded=base64.b64encode(payload).decode();digest=hashlib.sha256(payload).hexdigest()
    if resume and not resume_url: raise RuntimeError('A committed checkpoint URL is required; embedding exceeds Kaggle source limits.')
    checkpoint_hash=hashlib.sha256(resume.read_bytes()).hexdigest() if resume else None
    def cell(kind,text,index):
        c=dict(cell_type=kind,id=f'advanced-{index}',metadata={},source=text.splitlines(True))
        if kind=='code': c.update(execution_count=None,outputs=[])
        return c
    setup=f'''import base64,hashlib,io,json,os,pathlib,subprocess,sys,urllib.request,zipfile
ROOT=pathlib.Path('/tmp/particle_parametric_source')
ROOT.mkdir(parents=True,exist_ok=True)
payload=base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for n in z.namelist():
        if not (ROOT/n).resolve().is_relative_to(ROOT.resolve()): raise RuntimeError('Invalid source member')
    z.extractall(ROOT)
os.environ['MPLCONFIGDIR']='/tmp/particle_matplotlib'
os.environ['PIP_CACHE_DIR']='/tmp/particle_pip_cache'
os.environ['XDG_CACHE_HOME']='/tmp/particle_cache'
import torch
if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; CPU fallback disabled')
print('GPU:',torch.cuda.get_device_name(0),'CUDA:',torch.version.cuda,flush=True)
import importlib.metadata,importlib.util
from packaging.requirements import Requirement
missing=[]
for spec in (ROOT/'requirements.txt').read_text().splitlines():
    if not spec.strip() or spec.startswith('#'): continue
    req=Requirement(spec)
    try: compatible=importlib.metadata.version(req.name) in req.specifier
    except importlib.metadata.PackageNotFoundError: compatible=False
    if not compatible:
        if req.name=='torch': raise RuntimeError('CUDA PyTorch version does not meet project requirements')
        missing.append(spec)
if {bool(live_dataset)!r} and importlib.util.find_spec('kaggle') is None: missing.append('kaggle==2.2.4')
if missing: subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)
source_manifest=pathlib.Path('/tmp/particle_source_manifest.json')
source_manifest.write_text(json.dumps({{'fingerprint':{fingerprint!r},'files':{hashes!r}}}))
os.environ['PARTICLE_SOURCE_MANIFEST']=str(source_manifest)
RESUME=None
checkpoint_url={resume_url!r}
if checkpoint_url:
    with urllib.request.urlopen(checkpoint_url,timeout=120) as response:
        data=response.read()
    assert hashlib.sha256(data).hexdigest()=={checkpoint_hash!r}
    RESUME=pathlib.Path('/tmp/previous_checkpoint.pt');RESUME.write_bytes(data)
'''
    args=[]
    for key in ['total_steps','chunk_steps','snapshot_every','samples','width','depth','curriculum_steps','particles','reference_ns','reference_nn','seed']:
        args.extend(['--'+key.replace('_','-'),str(settings[key])])
    args+=['--async-exports']
    if live_dataset: args+=['--live-dataset',live_dataset]
    run=f'''RUN=pathlib.Path('/kaggle/working/advanced_output')
command=[sys.executable,'-u','run_parametric_pinn.py','--device','cuda','--out',str(RUN)]+{args!r}
if RESUME: command+=['--resume',str(RESUME)]
log=pathlib.Path('/kaggle/working/advanced_training.log')
print('MAIN TRAINING:', ' '.join(command),flush=True)
with log.open('w') as stream:
    with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        for line in process.stdout:
            stream.write(line);stream.flush()
            print(line,end='',flush=True)
        returncode=process.wait()
if RUN.exists():
    (RUN/'source_manifest.json').write_text(json.dumps({{'fingerprint':{fingerprint!r},'files':{hashes!r}}},indent=2))
if (RUN/'chunk_complete.json').exists(): print((RUN/'chunk_complete.json').read_text())
if returncode: raise RuntimeError('Pipeline failed; inspect output files and log. Exit '+str(returncode))
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},
        cells=[cell('markdown','# Global parametric particle PINN\nOne continuous optimizer state, exported every 1,000 steps. All valid parameter combinations are sampled; three anchors are shown. No GitHub token is present in Kaggle.\n',0),cell('code',setup,1),cell('code',run,2)])
    serialized=json.dumps(nb,indent=1)
    if len(serialized.encode())>=950000: raise RuntimeError('Notebook exceeds the safe Kaggle source size budget.')
    (out/'advanced.ipynb').write_text(serialized)
    meta=dict(id=owner+'/'+slug,title=slug.replace('-',' '),code_file='advanced.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    return fingerprint


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--owner',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--slug',default='particle-parametric-advanced');args=parser.parse_args()
    cfg=json.loads((Path(__file__).parent/'settings.json').read_text())
    print(build(Path(__file__).resolve().parents[1],args.out,args.owner,args.slug,cfg))
