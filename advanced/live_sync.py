"""Collect private dataset versions while the Kaggle GPU session stays active."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from datetime import datetime,timezone


def check_remote(api,repo,state,write_state):
    remote=str(api.kernels_status(state['kernel']).status).split('.')[-1].upper()
    if remote in ['ERROR','CANCELLED','CANCELED']:
        state.update(status='interrupted',remote_status=remote);write_state(repo/'advanced/run_state.json',state)
        raise RuntimeError('Continuous GPU session was interrupted; latest published checkpoint preserved')
    if remote=='COMPLETE' and state['completed_steps']<state['settings']['total_steps']:
        now=datetime.now(timezone.utc)
        first=state.setdefault('remote_complete_observed_utc',now.isoformat())
        if (now-datetime.fromisoformat(first)).total_seconds()>180:
            state.update(status='interrupted',remote_status=remote)
            write_state(repo/'advanced/run_state.json',state)
            raise RuntimeError('GPU session ended before final publication; preserve checkpoint and inspect kernel outputs')
        write_state(repo/'advanced/run_state.json',state)
        print('Session ended; waiting for the last dataset version to finish processing.')


def safe_extract(archive,target):
    target=Path(target).resolve();target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (target/item.filename).resolve().is_relative_to(target): raise RuntimeError('Unsafe publication path')
        z.extractall(target)


def collect_live(api,repo,state,temp,write_state):
    output=repo/'advanced_runs'/state['run_id'];output.mkdir(parents=True,exist_ok=True)
    dataset=state['live_dataset']
    try: metadata=json.loads(api.dataset_status(dataset,format='json'))
    except Exception as error:
        if getattr(getattr(error,'response',None),'status_code',None)==404:
            check_remote(api,repo,state,write_state)
            print('Waiting for the first live snapshot.');return
        raise
    if metadata['status']!='ready':
        check_remote(api,repo,state,write_state)
        print('Snapshot dataset processing; retrying shortly.');return
    for version in range(state.get('dataset_version',0)+1,metadata['current_version_number']+1):
        folder=temp/f'live_version_{version}';folder.mkdir(exist_ok=True)
        api.dataset_download_files(dataset+'/'+str(version),path=str(folder),quiet=True,force=True)
        archive=folder/(dataset.split('/')[1]+'.zip');unpacked=folder/'dataset'
        safe_extract(archive,unpacked)
        manifest=json.loads((unpacked/'publication.json').read_text());bundle=unpacked/'snapshot.bin'
        if hashlib.sha256(bundle.read_bytes()).hexdigest()!=manifest['bundle_sha256']: raise RuntimeError('Live snapshot checksum mismatch')
        if manifest['source_fingerprint']!=state['source_fingerprint']: raise RuntimeError('Live training source mismatch')
        root=folder/'snapshot';safe_extract(bundle,root)
        step=manifest['step']
        if step>state['completed_steps']:
            snapshot=root/'snapshots'/f'step_{step:06d}'
            if not (snapshot/'complete.json').exists(): raise RuntimeError('Incomplete live snapshot')
            target=output/'snapshots'/snapshot.name
            if not target.exists(): shutil.copytree(snapshot,target)
            for name in ['checkpoint.pt','config.json','source_manifest.json','chunk_complete.json','failures.json']:
                shutil.copy2(root/name,output/name)
            state['completed_steps']=step
            if manifest['training_complete']: state['status']='complete'
            print('LIVE_SNAPSHOT_COLLECTED step='+str(step))
        state['dataset_version']=version
        write_state(repo/'advanced/run_state.json',state)
    if state['status']!='complete':
        check_remote(api,repo,state,write_state)
