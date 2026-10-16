"""Regression for Kaggle's 1 MB notebook-source limit and numerical-code guards."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('checkpoint_builder',ROOT/'advanced/build_notebook.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class CheckpointTransportTests(unittest.TestCase):
    def test_megabyte_checkpoint_is_referenced_not_embedded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'requirements.txt').write_text('numpy\n')
            checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(bytes(range(256))*10000)
            cfg=json.loads((ROOT/'advanced/settings.json').read_text())
            builder.build(root,root/'notebook','owner','study',cfg,checkpoint,
                          resume_url='https://example.invalid/immutable/checkpoint.pt')
            source=(root/'notebook/advanced.ipynb').read_text()
            self.assertLess(len(source.encode()),950000)
            self.assertIn('https://example.invalid/immutable/checkpoint.pt',source)
            self.assertIn('hashlib.sha256(data)',source)
            self.assertEqual(json.loads((root/'notebook/kernel-metadata.json').read_text())['is_private'],True)

    def test_only_numerical_code_changes_invalidate_training_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'requirements.txt').write_text('numpy\n');(root/'advanced').mkdir()
            (root/'run_parametric_pinn.py').write_text('training=1\n')
            helper=root/'advanced/controller.py';helper.write_text('transport=1\n')
            _,first,_=builder.source_payload(root)
            helper.write_text('transport=2\n');_,second,_=builder.source_payload(root)
            self.assertEqual(first,second)
            (root/'run_parametric_pinn.py').write_text('training=2\n');_,third,_=builder.source_payload(root)
            self.assertNotEqual(first,third)


if __name__=='__main__': unittest.main()
