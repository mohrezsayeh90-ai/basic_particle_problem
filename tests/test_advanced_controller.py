"""Exercise scheduled collection/continuation without any real API mutation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('particle_advanced_controller',ROOT/'advanced/controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def fixture(self,root):
        (root/'advanced').mkdir();(root/'advanced_runs').mkdir()
        (root/'advanced/run_state.json').write_text(json.dumps({'status':'running','run_id':'test',
            'kernel':'testowner/testkernel','completed_steps':1000,'source_fingerprint':'source',
            'settings':{'total_steps':3000,'chunk_steps':1000}}))
        result=root/'remote/advanced_output';result.mkdir(parents=True)
        (result/'snapshots/step_002000').mkdir(parents=True)
        (result/'snapshots/step_002000/complete.json').write_text('{"step":2000,"export_errors":0}')
        (result/'checkpoint.pt').write_bytes(b'fake trusted checkpoint for controller transport test')
        for name,value in [('config.json',{}),('failures.json',[]),('source_manifest.json',{'fingerprint':'source'}),
            ('chunk_complete.json',{'start_step':1000,'step':2000,'reached_chunk_target':True,
             'training_complete':False,'execution_failures':0,'gpu':'fake-test-gpu','timestamp_utc':'test-time'})]:
            (result/name).write_text(json.dumps(value))
        class FakeAPI:
            config_values={'username':'testowner'}
            def authenticate(self): pass
            def kernels_status(self,kernel): return types.SimpleNamespace(status='COMPLETE')
            def kernels_output(self,kernel,path,**kwargs): shutil.copytree(root/'remote',path,dirs_exist_ok=True)
            def kernels_push(self,*args,**kwargs):
                self.pushes+=1
                return types.SimpleNamespace(error=None,versionNumber=3,url='https://example.invalid/test')
            pushes=0
        return FakeAPI()

    def invoke(self,root,api):
        fake=types.ModuleType('kaggle.api.kaggle_api_extended');fake.KaggleApi=lambda:api
        with patch.dict(sys.modules,{'kaggle':types.ModuleType('kaggle'),'kaggle.api':types.ModuleType('kaggle.api'),
                                     'kaggle.api.kaggle_api_extended':fake}), \
             patch.object(controller,'__file__',str(root/'advanced/controller.py')), \
             patch.object(controller,'build',return_value='source') as builder, \
             patch.object(sys,'argv',['controller.py']), \
             patch.dict(os.environ,{'KAGGLE_API_TOKEN':'test-placeholder-not-a-real-secret','RUNNER_TEMP':str(root/'tmp')}):
            controller.main()
            return builder

    def test_collect_then_continue_preserves_checkpoint_and_step(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);api=self.fixture(root);builder=self.invoke(root,api)
            state=json.loads((root/'advanced/run_state.json').read_text())
            self.assertEqual(state['status'],'running');self.assertEqual(state['completed_steps'],2000)
            self.assertEqual(api.pushes,1)
            self.assertTrue((root/'advanced_runs/test/snapshots/step_002000/complete.json').exists())
            self.assertEqual(builder.call_args.args[5],root/'advanced_runs/test/checkpoint.pt')

    def test_stale_outputs_never_start_a_duplicate_training_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);api=self.fixture(root)
            p=root/'remote/advanced_output/chunk_complete.json';v=json.loads(p.read_text());v['step']=1000;p.write_text(json.dumps(v))
            self.invoke(root,api)
            self.assertEqual(api.pushes,0)
            self.assertEqual(json.loads((root/'advanced/run_state.json').read_text())['completed_steps'],1000)

    def test_failed_chunk_is_recorded_and_cannot_continue(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);api=self.fixture(root)
            p=root/'remote/advanced_output/chunk_complete.json';v=json.loads(p.read_text());v['execution_failures']=1;p.write_text(json.dumps(v))
            with self.assertRaises(RuntimeError): self.invoke(root,api)
            self.assertEqual(api.pushes,0)
            self.assertEqual(json.loads((root/'advanced/run_state.json').read_text())['status'],'failed')


if __name__=='__main__': unittest.main()
