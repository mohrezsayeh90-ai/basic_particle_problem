"""Evidence-integrity checks for the data-to-LaTeX bridge (standard library)."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("proposal_generator",ROOT/"proposal"/"generate_results.py")
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class ProposalEvidenceTests(unittest.TestCase):
    def snapshot(self, destination):
        source=ROOT/"results"/"final_study"
        for name in ["metrics.json","config.json","manifest.json","stages.json",
                     "data/mesh_sensitivity.csv","data/seed_summary.csv","data/trajectory_summary.json"]:
            target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
        (destination/"figures").mkdir()
        for name in generator.REQUIRED_FIGURES:
            (destination/"figures"/f"{name}.pdf").symlink_to(source/"figures"/f"{name}.pdf")

    def test_missing_3d_figure_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.snapshot(root/"inputs")
            (root/"inputs/figures/19_3d_benchmark.pdf").unlink()
            with self.assertRaisesRegex(FileNotFoundError,"19_3d_benchmark"):
                generator.generate(root/"inputs",root/"generated")

    def test_wrong_endpoint_agreement_is_rejected_against_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.snapshot(root/"inputs")
            path=root/"inputs/metrics.json"
            metrics=json.loads(path.read_text())
            metrics["trajectory"]["baseline_comparison"][-1]["outcome_agreement"]=True
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,"outcome flags disagree"):
                generator.generate(root/"inputs",root/"generated")


if __name__=="__main__":
    unittest.main()
