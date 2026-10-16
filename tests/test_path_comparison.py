"""Independent analytical examples for saved-trajectory error metrics."""
import unittest
import numpy as np
from particle_pinn.path_comparison import SavedPath,compare_pair,geometry_errors,interpolate


def path(times,positions,outcome='exited',index=0):
    return SavedPath(index,'test','baseline',0.,.025,1000.,outcome,np.array(times,dtype=float),np.array(positions,dtype=float))


class PathComparisonTests(unittest.TestCase):
    def test_identical_paths_have_zero_errors(self):
        a=path([0,1,2],[[0,0],[1,.3],[2,0]])
        r,_=compare_pair(a,a,width=.3)
        self.assertAlmostEqual(r['time_rms'],0.)
        self.assertAlmostEqual(r['geometric_rms'],0.,places=12)
        self.assertEqual(r['common_time_coverage'],1.)

    def test_same_geometry_different_speed_has_nonzero_motion_error(self):
        a=path([0,2],[[0,0],[1,0]])
        b=path([0,1],[[0,0],[1,0]],index=1)
        r,_=compare_pair(a,b,width=1.)
        self.assertAlmostEqual(r['time_rms'],.5/np.sqrt(3),places=6)
        self.assertAlmostEqual(r['geometric_rms'],0.,places=12)
        self.assertEqual(r['common_time_coverage'],.5)
        self.assertAlmostEqual(r['final_endpoint_distance'],0.)

    def test_early_contact_cannot_be_hidden_by_common_time_agreement(self):
        a=path([0,1],[[0,0],[1,0]],outcome='stuck')
        b=path([0,2],[[0,0],[2,0]],index=1)
        r,_=compare_pair(a,b,width=1.)
        self.assertAlmostEqual(r['time_rms'],0.)
        self.assertFalse(r['outcome_agreement'])
        self.assertEqual(r['common_time_coverage'],.5)
        self.assertGreater(r['geometric_rms'],.28)
        self.assertAlmostEqual(r['sampled_hausdorff'],1.)

    def test_parallel_line_geometry_has_known_distance(self):
        r=geometry_errors(np.array([[0,0],[2,0]]),np.array([[0,.1],[2,.1]]))
        self.assertAlmostEqual(r['geometric_rms'],.1,places=12)
        self.assertAlmostEqual(r['sampled_hausdorff'],.1,places=12)

    def test_post_termination_extrapolation_is_rejected(self):
        a=path([0,1],[[0,0],[1,0]])
        with self.assertRaisesRegex(ValueError,'extrapolation'):
            interpolate(a,np.array([0.,2.]))


if __name__=='__main__':
    unittest.main()
