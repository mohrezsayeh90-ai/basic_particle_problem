"""Audit saved paths: synchronous motion error and timing-independent geometry.

No extrapolation, terminal-position padding, re-training or re-integration.
All distances are nondimensional lengths; percentage metrics divide by width.
"""
from dataclasses import dataclass
import csv
import numpy as np


@dataclass
class SavedPath:
    track: int
    source: str
    case: str
    eta: float
    diameter: float
    density_ratio: float
    outcome: str
    t: np.ndarray
    xy: np.ndarray


def load_paths(file):
    groups = {}
    with open(file, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(int(row["track"]), []).append(row)
    tracks = []
    for index, rows in groups.items():
        first = rows[0]
        for row in rows:
            for key in ("source", "case", "eta", "diameter", "density_ratio", "outcome"):
                if row[key] != first[key]:
                    raise ValueError(f"Metadata changes within track {index}: {key}")
        t = np.array([float(r["t"]) for r in rows])
        xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        if not np.isfinite(t).all() or not np.isfinite(xy).all():
            raise ValueError(f"Non-finite samples in track {index}")
        if len(t)>1 and np.any(np.diff(t)<=0):
            raise ValueError(f"Non-increasing sample times in track {index}")
        tracks.append(SavedPath(index,first["source"],first["case"],float(first["eta"]),
                               float(first["diameter"]),float(first["density_ratio"]),first["outcome"],t,xy))
    return tracks


def pair_paths(tracks, source, reference):
    def key(t):
        return (t.case, round(t.eta,12), round(t.diameter,12), round(t.density_ratio,8))
    refs = {}
    for track in tracks:
        if track.source == reference:
            if key(track) in refs:
                raise ValueError("Ambiguous reference pairing")
            refs[key(track)] = track
    pairs = []
    for track in tracks:
        if track.source == source:
            if key(track) not in refs:
                raise ValueError(f"Missing reference for track {track.track}")
            pairs.append((track,refs[key(track)]))
    if not pairs:
        raise ValueError(f"No paths for {source} versus {reference}")
    for a,b in pairs:
        if not np.allclose(a.xy[0],b.xy[0],rtol=0,atol=1e-8):
            raise ValueError(f"Initial positions differ: {a.track}, {b.track}")
    return pairs


def interpolate(path, times):
    if times[0] < path.t[0]-1e-12 or times[-1] > path.t[-1]+1e-12:
        raise ValueError("Trajectory extrapolation is prohibited")
    return np.column_stack([np.interp(times,path.t,path.xy[:,j]) for j in range(2)])


def arc_samples(xy, count=501):
    """Uniform-arclength samples of the saved piecewise-linear curve."""
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    unique = np.r_[True,np.diff(lengths)>1e-14]
    lengths,xy = lengths[unique],xy[unique]
    if lengths[-1]<=1e-14:
        return np.repeat(xy[:1],count,axis=0),0.
    q = np.linspace(0,lengths[-1],count)
    return np.column_stack([np.interp(q,lengths,xy[:,j]) for j in range(2)]),float(lengths[-1])


def point_polyline_distance(points, polyline):
    """Exact point-to-segment minimum for each supplied sample point."""
    starts, vectors = polyline[:-1],np.diff(polyline,axis=0)
    squared = np.sum(vectors*vectors,axis=1)
    valid = squared>1e-28
    starts,vectors,squared = starts[valid],vectors[valid],squared[valid]
    if len(starts)==0:
        return np.linalg.norm(points-polyline[0],axis=1)
    result=[]
    for first in range(0,len(points),256):
        delta=points[first:first+256,None,:]-starts[None,:,:]
        alpha=np.clip(np.sum(delta*vectors[None,:,:],axis=2)/squared,0.,1.)
        offset=delta-alpha[:,:,None]*vectors[None,:,:]
        result.extend(np.sqrt(np.min(np.sum(offset*offset,axis=2),axis=1)))
    return np.array(result)


def geometry_errors(a,b,count=501):
    pa,la = arc_samples(a,count)
    pb,lb = arc_samples(b,count)
    da,db = point_polyline_distance(pa,b),point_polyline_distance(pb,a)
    q=np.linspace(0,1,count)
    rms=float(np.sqrt((np.trapezoid(da**2,q)+np.trapezoid(db**2,q))/2))
    return dict(geometric_rms=rms,sampled_hausdorff=float(max(da.max(),db.max())),
                predicted_path_length=la,reference_path_length=lb)


def compare_pair(pred,ref,width,channel=None,count=1001):
    if width<=0 or count<3:
        raise ValueError("Positive width and at least three comparison samples are required")
    row=dict(predicted_track=pred.track,reference_track=ref.track,source=pred.source,
             reference=ref.source,case=pred.case,eta=pred.eta,diameter=pred.diameter,
             density_ratio=pred.density_ratio,predicted_outcome=pred.outcome,reference_outcome=ref.outcome,
             outcome_agreement=pred.outcome==ref.outcome,
             terminal_events_complete=pred.outcome in ("exited","stuck","backflow_exit") and ref.outcome in ("exited","stuck","backflow_exit"))
    if len(pred.t)<2 or len(ref.t)<2 or "integration_failed" in (pred.outcome,ref.outcome):
        return dict(row,status="unavailable_failed_or_short_path"),None
    start=max(pred.t[0],ref.t[0]); end=min(pred.t[-1],ref.t[-1])
    if end<=start:
        return dict(row,status="unavailable_no_common_time"),None
    times=np.linspace(start,end,count)
    p,r=interpolate(pred,times),interpolate(ref,times)
    delta=p-r; distance=np.linalg.norm(delta,axis=1)
    duration=end-start
    rms=float(np.sqrt(np.trapezoid(distance**2,times)/duration))
    # Reproduce the previous 150-sample arithmetic RMS exactly for auditing.
    old_times=np.linspace(start,end,150)
    old_delta=interpolate(pred,old_times)-interpolate(ref,old_times)
    old_rms=float(np.sqrt(np.mean(np.sum(old_delta**2,axis=1))))
    fine_times=np.linspace(start,end,2*count-1)
    fine_delta=interpolate(pred,fine_times)-interpolate(ref,fine_times)
    fine_rms=float(np.sqrt(np.trapezoid(np.sum(fine_delta**2,axis=1),fine_times)/duration))
    row.update(status="compared",common_start=float(start),common_end=float(end),
               common_time_coverage=float(duration/(max(pred.t[-1],ref.t[-1])-start)),
               reference_time_coverage=float(duration/(ref.t[-1]-start)),
               predicted_end_time=float(pred.t[-1]),reference_end_time=float(ref.t[-1]),
               termination_time_difference_percent=float(100*(pred.t[-1]-ref.t[-1])/(ref.t[-1]-start)),
               time_rms=rms,time_rms_width_percent=100*rms/width,
               max_time_error=float(distance.max()),max_time_error_width_percent=float(100*distance.max()/width),
               max_error_time=float(times[np.argmax(distance)]),
               final_endpoint_distance=float(np.linalg.norm(pred.xy[-1]-ref.xy[-1])),
               final_endpoint_width_percent=float(100*np.linalg.norm(pred.xy[-1]-ref.xy[-1])/width),
               legacy_sample_rms_150=old_rms,metric_quadrature_change_width_percent=100*abs(fine_rms-rms)/width)
    geometry=geometry_errors(pred.xy,ref.xy)
    row.update(geometry)
    row["geometric_rms_width_percent"]=100*geometry["geometric_rms"]/width
    row["sampled_hausdorff_width_percent"]=100*geometry["sampled_hausdorff"]/width
    if channel is not None:
        s,_=channel.coordinates(r)
        _,tangent,normal=channel.frame(np.clip(s,0,channel.length))
        along=np.sum(delta*tangent,axis=1)
        across=np.sum(delta*normal,axis=1)
        row["tangential_rms_width_percent"]=100*np.sqrt(np.trapezoid(along**2,times)/duration)/width
        row["normal_rms_width_percent"]=100*np.sqrt(np.trapezoid(across**2,times)/duration)/width
        assert np.isclose(row["tangential_rms_width_percent"]**2+row["normal_rms_width_percent"]**2,
                          row["time_rms_width_percent"]**2,rtol=1e-9)
    else:
        along=across=np.full(count,np.nan)
    series=dict(t=times,predicted=p,reference=r,distance=distance,tangential=along,normal=across)
    return row,series
