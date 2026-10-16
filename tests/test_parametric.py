"""Physics, parameter coverage and checkpoint properties for the advanced model."""
import unittest
import numpy as np
import torch
from torch import nn
from particle_pinn.parametric import BASE,Case,ParametricFlow,anchors,sample_cases,map_points,coordinates,pde_residual,drag_acceleration
from particle_pinn.advanced_exports import trace
from particle_pinn.geometry import Channel

torch.set_num_threads(1)


class ParametricTests(unittest.TestCase):
    def test_sampling_covers_decades_without_invalid_cases(self):
        values,rejected=sample_cases(np.random.default_rng(3),3000)
        self.assertGreater(rejected,0)
        self.assertTrue(all(Case(*v).valid() for v in values))
        ratios=values/BASE
        self.assertTrue(np.all(ratios>=.1));self.assertTrue(np.all(ratios<=10))
        self.assertTrue(np.all(ratios.min(0)<.13));self.assertTrue(np.all(ratios.max(0)>8))
        self.assertEqual(len(np.unique(values[:,2])),len(values))

    def test_analytic_coordinate_roundtrip(self):
        for case in anchors().values():
            q=torch.linspace(.01,.99,101,dtype=torch.float64)
            eta=torch.linspace(-.9,.9,101,dtype=torch.float64)
            p=torch.tensor(case.array()[None,:3],dtype=torch.float64).expand(len(q),-1)
            x=map_points(q,eta,p);a,b,t=coordinates(x,p)
            torch.testing.assert_close(a,q,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(b,eta,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close((t*t).sum(1),torch.ones_like(q))

    def test_hard_wall_cap_velocities_and_pressure_gauge(self):
        model=ParametricFlow(16,2)
        with torch.no_grad(): model.net[-1].weight.normal_(0,.2)
        for case in anchors().values():
            q=torch.linspace(0,1,40);p=torch.tensor(case.array()[None,:3],dtype=torch.float32).expand(len(q),-1)
            for side in [-1,1]:
                result=model(map_points(q,torch.full_like(q,side),p),p)
                torch.testing.assert_close(result[:,:2],torch.zeros_like(result[:,:2]),atol=1e-5,rtol=0)
            eta=torch.linspace(-.95,.95,40)
            for end in [0,1]:
                xy=map_points(torch.full_like(eta,end),eta,p);_,_,t=coordinates(xy,p)
                expected=(1-eta.square())[:,None]*t
                torch.testing.assert_close(model(xy,p)[:,:2],expected,atol=1e-5,rtol=1e-5)
            r=case.radius
            gauge=model(torch.tensor([[4*r,0]],dtype=torch.float32),p[:1])[0,2]
            self.assertLess(abs(float(gauge.detach())),1e-3)

    def test_physical_pde_derivatives_and_parameter_scaling(self):
        class Poiseuille(nn.Module):
            def forward(self,xy,p):
                h=(p[:,1]-p[:,0])/2
                u=1-(xy[:,1]/h).square();v=0*xy[:,0]**3
                pressure=-2*xy[:,0]/h.square()
                return torch.stack((u,v,pressure),dim=1)
        p=torch.tensor([[.02,.05,.1],[.2,.5,1],[2,5,10]],dtype=torch.float64)
        x=torch.tensor([[.02,.003],[.2,.03],[2,.3]],dtype=torch.float64,requires_grad=True)
        torch.testing.assert_close(pde_residual(Poiseuille(),x,p),torch.zeros((3,3),dtype=torch.float64),atol=1e-12,rtol=0)

    def test_spline_cap_roundoff_never_selects_the_wrong_bend(self):
        model=ParametricFlow(16,2)
        for case in anchors().values():
            channel=Channel('serpentine',case.rin,case.rout)
            xy,_,_=channel.grid(80,25)
            output=model.evaluate(xy,case)
            self.assertLess(float(np.linalg.norm(output[:,:2],axis=1).max()),1.01)

    def test_network_backward_is_finite_over_three_scales(self):
        model=ParametricFlow(16,2)
        p=torch.tensor(np.stack([v.array()[:3] for v in anchors().values()]),dtype=torch.float32)
        xy=map_points(torch.tensor([.17,.39,.77]),torch.tensor([-.3,.2,.6]),p).detach().requires_grad_(True)
        loss=pde_residual(model,xy,p).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))
        self.assertGreater(sum(float(v.grad.abs().sum()) for v in model.parameters() if v.grad is not None),0)

    def test_velocity_has_no_fixed_shear_jump_at_bend_interface(self):
        model=ParametricFlow(16,2).double()
        with torch.no_grad(): model.net[-1].weight.normal_(0,.1)
        for case in anchors().values():
            r=case.radius
            xy=torch.tensor([[2*r+.2*case.h,1e-8*r],[2*r+.2*case.h,-1e-8*r]],dtype=torch.float64,requires_grad=True)
            p=torch.tensor(case.array()[None,:3],dtype=torch.float64).expand(2,-1)
            uv=model(xy,p)[:,:2]
            for j in range(2):
                derivative=torch.autograd.grad(uv[:,j].sum(),xy,retain_graph=True)[0]*r
                torch.testing.assert_close(derivative[0],derivative[1],atol=1e-4,rtol=1e-4)

    def test_drag_has_stokes_limit_opposes_slip_and_is_odd(self):
        a=drag_acceleration([1e-10,0],[0,0],.025,1,1000)
        np.testing.assert_allclose(a,[18e-10/(1000*.025**2),0],rtol=1e-7)
        b=drag_acceleration([1.,-.4],[0,0],.025,10,1000)
        np.testing.assert_allclose(drag_acceleration([-1.,.4],[0,0],.025,10,1000),-b)
        np.testing.assert_array_equal(drag_acceleration([1.,2.],[1.,2.],.025,1,1000),[0,0])
        self.assertGreater(float(b[0]),0);self.assertLess(float(b[1]),0)

    def test_advanced_trace_in_constant_flow_exits_without_fabrication(self):
        c=Channel('straight');case=Case(diameter=.01)
        def field(x): return np.tile([1.,0.,0.],(len(x),1))
        path=trace(c,case,field,0,times=20)
        self.assertEqual(path['outcome'],'exited')
        np.testing.assert_allclose(path['velocity'],np.tile([1.,0.],(20,1)),atol=1e-8)
        self.assertLess(abs(path['xy'][-1,0]-2),1e-6)


if __name__=='__main__': unittest.main()
