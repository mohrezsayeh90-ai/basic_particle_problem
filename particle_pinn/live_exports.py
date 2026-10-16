"""CPU snapshot renderer and private Kaggle publisher, independent of GPU training."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
os.environ.setdefault('MPLCONFIGDIR','/tmp/particle-live-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/particle-live-cache')
import torch
from .parametric import ParametricFlow
from .advanced_exports import PeriodicExporter,write_json


def make_packet(out,checkpoint,step,config):
    stage=out/'.publication'/f'step_{step:06d}';stage.mkdir(parents=True,exist_ok=True)
    folder=out/'snapshots'/f'step_{step:06d}'
    runtime=json.loads((out/'runtime.json').read_text())
    progress=dict(start_step=runtime['start_step'],step=step,target_steps=config['total_steps'],
        reached_chunk_target=step>=config['total_steps'],training_complete=step>=config['total_steps'],
        execution_failures=0,device=config['device'],gpu=runtime['gpu'],timestamp_utc=datetime.now(timezone.utc).isoformat())
    write_json(stage/'chunk_complete.json',progress)
    # A .bin suffix prevents Kaggle from unpacking the zip on upload.
    bundle=stage/'snapshot.bin'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file(): z.write(p,str(p.relative_to(out)))
        for p in [out/'config.json',out/'source_manifest.json',stage/'chunk_complete.json']:
            z.write(p,p.name)
        z.write(checkpoint,'checkpoint.pt');z.writestr('failures.json','[]\n')
    manifest=dict(step=step,bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        source_fingerprint=json.loads((out/'source_manifest.json').read_text())['fingerprint'],
        target_steps=config['total_steps'],training_complete=progress['training_complete'])
    write_json(stage/'publication.json',manifest)
    return stage


class KagglePublisher:
    def __init__(self,identifier):
        os.environ.setdefault('KAGGLE_CONFIG_DIR','/tmp/particle_live_kaggle_config')
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api=KaggleApi();self.api.authenticate();self.identifier=identifier
        if self.api.config_values['username']!=identifier.split('/')[0]: raise RuntimeError('Dataset owner mismatch')

    def publish(self,stage,step):
        meta=dict(id=self.identifier,title=self.identifier.split('/')[1].replace('-',' '),licenses=[{'name':'CC0-1.0'}])
        write_json(stage/'dataset-metadata.json',meta)
        try:
            status=self.api.dataset_status(self.identifier)
            exists=True
        except Exception as e:
            if getattr(getattr(e,'response',None),'status_code',None)!=404: raise
            exists=False;status=None
        if exists:
            until=time.monotonic()+180
            while status!='ready' and time.monotonic()<until:
                time.sleep(5);status=self.api.dataset_status(self.identifier)
            if status!='ready': raise RuntimeError('Previous dataset version is still processing')
            r=self.api.dataset_create_version(str(stage),version_notes=f'Optimizer step {step}',quiet=True,convert_to_csv=False,delete_old_versions=False)
        else:
            r=self.api.dataset_create_new(str(stage),public=False,quiet=True,convert_to_csv=False)
        if r.error: raise RuntimeError('Kaggle publication rejected: '+r.error)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset')
    args=p.parse_args();out=args.out;config=json.loads((out/'config.json').read_text())
    torch.set_num_threads(1)
    exporter=PeriodicExporter(out,config['seed'],config['particles'],config['reference_ns'],config['reference_nn'],not config['no_animation'])
    model=ParametricFlow(config['width'],config['depth']).eval();publisher=None
    rendered=[];published=[];failures=[]
    queue=out/'.export_queue';queue.mkdir(exist_ok=True)
    while True:
        jobs=sorted(queue.glob('step_*.pt'))
        if not jobs:
            if (queue/'done.json').exists(): break
            time.sleep(.25);continue
        job=jobs[0];step=int(job.stem.split('_')[1]);folder=out/'snapshots'/f'step_{step:06d}'
        try:
            checkpoint=torch.load(job,map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model'])
            if not (folder/'complete.json').exists(): exporter.export(model,step,checkpoint['history'])
            rendered.append(step)
            if args.dataset:
                if publisher is None: publisher=KagglePublisher(args.dataset)
                stage=make_packet(out,job,step,config)
                last=None
                for delay in [0,3,10,30]:
                    if delay: time.sleep(delay)
                    try: publisher.publish(stage,step);last=None;break
                    except Exception as e: last=e
                if last is not None: raise last
                published.append(step);print(f'LIVE_PUBLISHED step={step} dataset={args.dataset}',flush=True)
            job.unlink()
        except Exception as e:
            # Keep immutable checkpoints and rendered products for local recovery.
            failures.append(dict(step=step,error_type=type(e).__name__))
            failed=job.with_suffix('.failed');job.replace(failed)
            print(f'EXPORT_WORKER_ERROR step={step} type={type(e).__name__}',flush=True)
        write_json(out/'live_publication_status.json',dict(rendered_steps=rendered,published_steps=published,failures=failures))
    write_json(out/'export_worker_complete.json',dict(rendered_steps=rendered,published_steps=published,failures=failures))
    return 2 if failures else 0


if __name__=='__main__': raise SystemExit(main())
