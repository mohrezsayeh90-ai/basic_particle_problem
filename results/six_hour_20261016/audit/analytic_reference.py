"""Independent exact two-semicircle coordinates for the audit FEM mesh."""
import numpy as np
class AnalyticChannel:
 def __init__(self,rin,rout):
  self.rin=rin;self.rout=rout;self.radius=(rin+rout)/2;self.half_width=(rout-rin)/2;self.length=2*np.pi*self.radius
 def frame(self,s):
  q=np.asarray(s)/self.length;second=q>.5;a=np.where(second,2*np.pi*q-np.pi,2*np.pi*q)
  si=np.sin(a);co=np.cos(a);si=np.where((q==0)|(q==.5)|(q==1),0.,si)
  R=self.radius;c=np.stack((np.where(second,3*R-R*co,R-R*co),np.where(second,-R*si,R*si)),axis=-1)
  t=np.stack((si,np.where(second,-co,co)),axis=-1);n=np.stack((-t[...,1],t[...,0]),axis=-1)
  return c,t,n
 def map(self,s,eta):
  c,t,n=self.frame(s);return c+self.half_width*np.asarray(eta)[...,None]*n
 def coordinates(self,points):
  x=np.atleast_2d(points);R=self.radius;second=(x[:,0]>3*R)|((x[:,0]>=R)&(x[:,1]<0))
  dx=x[:,0]-np.where(second,3*R,R);dy=np.maximum(np.where(second,-x[:,1],x[:,1]),0.)
  a=np.pi-np.arctan2(dy,dx);q=np.where(second,a+np.pi,a)/(2*np.pi)
  r=np.hypot(dx,dy);eta=np.where(second,(R-r)/self.half_width,(r-R)/self.half_width)
  return q*self.length,eta
 def grid(self,ns,nn):
  s,e=np.meshgrid(np.linspace(0,self.length,ns),np.linspace(-1,1,nn),indexing='ij');return self.map(s.ravel(),e.ravel()),s,e
 def sample(self,n,rng,margin=.03):
  return self.map(rng.uniform(margin,1-margin,n)*self.length,rng.uniform(-1+margin,1-margin,n))
