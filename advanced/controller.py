#!/usr/bin/env python3
"""Advance one private Kaggle run, or collect its finished 1,000-step chunk.

Run by GitHub Actions. Kaggle's token stays in GitHub Actions; GITHUB_TOKEN
is used by actions/checkout and git, and is never sent into Kaggle.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_notebook import build


def write_state(file,state):
    file.parent.mkdir(exist_ok=True,parents=True)
    temp=file.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(file)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true');args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];file=repo/'advanced/run_state.json'
    state=json.loads(file.read_text()) if file.exists() else {'status':'idle'}
    if not args.start and state['status'] not in ['running','ready']:
        print('No active training run:',state['status']);return
    if not os.environ.get('KAGGLE_API_TOKEN'): raise RuntimeError('Set KAGGLE_API_TOKEN in repository Actions secrets.')
    os.environ.setdefault('KAGGLE_CONFIG_DIR',str(Path(os.environ.get('RUNNER_TEMP','/tmp'))/'particle_kaggle_config'))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();owner=api.config_values['username']
    temp=Path(os.environ.get('RUNNER_TEMP','/tmp'))/'particle_kaggle_controller';temp.mkdir(exist_ok=True,parents=True)
    if args.start:
        if state['status'] in ['running','ready']: raise RuntimeError('An active run exists; collect it before starting another.')
        timestamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        state=dict(status='ready',run_id='parametric-'+timestamp,kernel=owner+'/particle-parametric-'+timestamp,
                   settings=json.loads((repo/'advanced/settings.json').read_text()),completed_steps=0)
        write_state(file,state)
    if state['kernel'].split('/')[0]!=owner: raise RuntimeError('Authenticated Kaggle owner does not match run state.')
    output=repo/'advanced_runs'/state['run_id'];output.mkdir(parents=True,exist_ok=True)
    if state['status']=='running':
        result=api.kernels_status(state['kernel']);status=str(result.status).split('.')[-1].upper()
        print('Kaggle status:',status)
        if status in ['QUEUED','RUNNING']: return
        if status!='COMPLETE':
            state.update(status='failed',remote_status=status);write_state(file,state)
            raise RuntimeError('Kaggle execution did not complete: '+status)
        download=temp/('download_'+str(state['completed_steps']));download.mkdir(exist_ok=True)
        # Download only this run's products, avoiding pip caches and source trees.
        api.kernels_output(state['kernel'],str(download),file_pattern=r'^advanced_output/|^advanced_training\.log$',quiet=True)
        root=download/'advanced_output';done=json.loads((root/'chunk_complete.json').read_text())
        expected=min(state['completed_steps']+state['settings']['chunk_steps'],state['settings']['total_steps'])
        if done['step']==state['completed_steps']:
            print('Latest session outputs have not propagated yet; keeping the active run unchanged.')
            return
        if done['start_step']!=state['completed_steps'] or done['step']!=expected or not done['reached_chunk_target']:
            state.update(status='failed',error='Incomplete or stale chunk');write_state(file,state)
            raise RuntimeError('Incomplete or stale chunk; preserving the last committed checkpoint.')
        if done['execution_failures']:
            state.update(status='failed',error='Pipeline failure; inspect downloaded artifacts');write_state(file,state)
            shutil.copytree(root,output/'failed_chunk',dirs_exist_ok=True)
            raise RuntimeError('Execution failure; no next chunk submitted.')
        actual=json.loads((root/'source_manifest.json').read_text())['fingerprint']
        if actual!=state['source_fingerprint']: raise RuntimeError('Remote source mismatch')
        # Copy complete snapshot folders before updating the last checkpoint.
        for folder in (root/'snapshots').glob('step_*'):
            if not (folder/'complete.json').exists(): raise RuntimeError('Snapshot export incomplete')
            target=output/'snapshots'/folder.name
            if target.exists(): raise RuntimeError('Snapshot already exists; refusing overwrite')
            shutil.copytree(folder,target)
        for name in ['checkpoint.pt','config.json','chunk_complete.json','source_manifest.json','failures.json']:
            shutil.copy2(root/name,output/name)
        if (download/'advanced_training.log').exists(): shutil.copy2(download/'advanced_training.log',output/'latest_training.log')
        state.update(completed_steps=done['step'],status='complete' if done['training_complete'] else 'ready',gpu=done['gpu'],updated_utc=done['timestamp_utc'])
        write_state(file,state)
    if state['status']=='ready':
        folder=temp/('submit_'+str(state['completed_steps']));folder.mkdir(exist_ok=True)
        resume=output/'checkpoint.pt' if state['completed_steps'] else None
        fingerprint=build(repo,folder,owner,state['kernel'].split('/')[1],state['settings'],resume,state.get('source_fingerprint'))
        result=api.kernels_push(str(folder),timeout=7200,acc='NvidiaTeslaT4')
        if result is None or result.error: raise RuntimeError('Kaggle rejected the chunk submission')
        state.update(status='running',source_fingerprint=fingerprint,kaggle_version=result.versionNumber,
                     submitted_utc=datetime.now(timezone.utc).isoformat(),url=result.url)
        write_state(file,state)
        print('Submitted private GPU chunk:',state['kernel'],'starting at step',state['completed_steps'])
    print('State:',state['status'],'completed steps:',state['completed_steps'])


if __name__=='__main__': main()
