"""Live publication integrity and preservation of every intermediate snapshot."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('live_sync_tested',ROOT/'advanced/live_sync.py')
live=importlib.util.module_from_spec(spec);spec.loader.exec_module(live)


class LiveSyncTests(unittest.TestCase):
    def test_archive_paths_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('../escape.txt','bad')
            with self.assertRaises(RuntimeError): live.safe_extract(archive,root/'output')
            self.assertFalse((root/'escape.txt').exists())

    def test_each_dataset_version_is_collected_while_training_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'advanced').mkdir();temp=root/'tmp';temp.mkdir()
            state={'run_id':'study','live_dataset':'owner/dataset','source_fingerprint':'known',
                   'completed_steps':0,'status':'running','kernel':'owner/kernel','settings':{'total_steps':3000}}
            versions=[]
            for step in [1000,2000]:
                stream=io.BytesIO()
                with zipfile.ZipFile(stream,'w') as z:
                    z.writestr(f'snapshots/step_{step:06d}/complete.json','{}')
                    for name in ['config.json','source_manifest.json','chunk_complete.json','failures.json']: z.writestr(name,'{}')
                    z.writestr('checkpoint.pt',str(step))
                data=stream.getvalue();outer=io.BytesIO()
                with zipfile.ZipFile(outer,'w') as z:
                    z.writestr('snapshot.bin',data)
                    z.writestr('publication.json',json.dumps(dict(step=step,bundle_sha256=hashlib.sha256(data).hexdigest(),
                        source_fingerprint='known',training_complete=False)))
                versions.append(outer.getvalue())
            class API:
                def dataset_status(self,*a,**k): return '{"status":"ready","current_version_number":2}'
                def dataset_download_files(self,dataset,path,**kwargs):
                    number=int(dataset.split('/')[-1]);(Path(path)/'dataset.zip').write_bytes(versions[number-1])
                def kernels_status(self,*a): return types.SimpleNamespace(status='RUNNING')
            def save(path,value): path.write_text(json.dumps(value))
            live.collect_live(API(),root,state,temp,save)
            self.assertEqual(state['completed_steps'],2000);self.assertEqual(state['dataset_version'],2)
            self.assertEqual(state['status'],'running')
            self.assertEqual((root/'advanced_runs/study/checkpoint.pt').read_text(),'2000')
            self.assertEqual(len(list((root/'advanced_runs/study/snapshots').glob('step_*'))),2)


if __name__=='__main__': unittest.main()
