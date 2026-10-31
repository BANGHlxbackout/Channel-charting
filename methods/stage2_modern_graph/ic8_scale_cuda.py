"""P2A-R IC8 scalar PDHG with one explicit disabled-aux Ball-GLR control.

The accepted P2A solver stays read-only. Graph storage and exact projection are
reused; the IC8 update is retained and tested against that implementation.
"""
import time
import numpy as np
import torch

from methods.stage2_modern_graph.ic_cuda import CudaICGraph
from methods.stage2_modern_graph.ic_solver import project_observation


def solve_ic8_scale(x,graph,*,alpha,eta,aux_mode='enabled',max_iter=30000,tolerance=2e-6,callback=None):
    if aux_mode not in ('enabled','disabled'):raise ValueError('invalid aux_mode')
    disabled=aux_mode=='disabled'
    if not disabled and not 0<alpha<1:raise ValueError('IC8 alpha must lie in (0,1)')
    coefficient=0. if disabled else float(alpha);beta=1. if disabled else 1-coefficient
    x_cpu=np.asarray(x,dtype=np.float64);device=graph.device
    x=torch.as_tensor(x_cpu,device=device);norm=lambda a:torch.linalg.vector_norm(a,dim=0)
    radius=eta*norm(x-x.mean(0));base=.95/np.sqrt(2*graph.max_degree+1)
    scale=max(1.,float(radius.amin().item())/beta);tau=base*scale;sigma=base/scale
    lo=x.amin(0)-.5*(x.amax(0)-x.amin(0));hi=x.amax(0)+.5*(x.amax(0)-x.amin(0))
    u=x.clone();ub=u.clone();p=torch.zeros((graph.m,2),dtype=torch.float64,device=device)
    pb=p.clone();q=p.clone();box_bad=torch.zeros((),dtype=torch.bool,device=device)
    history=[];replays=0

    def apply(a,b):return graph.sqrtw*torch.sparse.mm(graph.d,a)-b
    def adjoint(a):return torch.sparse.mm(graph.dt,graph.sqrtw*a),-a
    def prox_dual(a):return a/(1+sigma/(2*beta))
    def prox_p(a):return torch.zeros_like(a) if disabled else a.sign()*torch.clamp(a.abs()-tau*coefficient,min=0.)
    def project(a,exact):
        diff=a-x;value=x+diff*torch.clamp(radius/norm(diff).clamp_min(1e-30),max=1.)
        bad=torch.any((value<lo)|(value>hi));box_bad.logical_or_(bad)
        if exact and bool(bad.item()):
            raw=a.cpu().numpy()
            value=torch.as_tensor(np.column_stack([project_observation(raw[:,i:i+1],x_cpu[:,i:i+1],float(radius[i].item()),
                (float(lo[i].item()),float(hi[i].item()))).ravel() for i in range(2)]),device=device)
        return value
    def step(exact=False):
        q.copy_(prox_dual(q+sigma*apply(ub,pb)));gu,gp=adjoint(q)
        un=project(u-tau*gu,exact);pn=prox_p(p-tau*gp)
        ub.copy_(2*un-u);pb.copy_(2*pn-p);u.copy_(un);p.copy_(pn)

    initial=[a.clone() for a in (u,p,q,ub,pb)];capture_error=None;replay=None
    try:
        stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        torch.cuda.current_stream(device).wait_stream(stream)
        for a,b in zip((u,p,q,ub,pb),initial):a.copy_(b)
        replay=torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):
            for _ in range(50):step()
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():raise
        capture_error=str(error);replay=None
    for a,b in zip((u,p,q,ub,pb),initial):a.copy_(b)
    del initial;box_bad.zero_();torch.cuda.synchronize(device);started=time.perf_counter();converged=False
    for offset in range(0,max_iter,50):
        count=min(50,max_iter-offset);box_bad.zero_()
        if replay is not None and count==50:
            saved=[a.clone() for a in (u,p,q,ub,pb)];replay.replay()
            if bool(box_bad.item()):
                for a,b in zip((u,p,q,ub,pb),saved):a.copy_(b)
                for _ in range(count):step(True)
                replays+=1
            del saved
        else:
            for _ in range(count):step(True)
        gu,gp=adjoint(q);res=apply(u,p)
        ru=norm(u-project(u-tau*gu,True))/tau;rp=norm(p-prox_p(p-tau*gp))/tau
        rq=norm(q-prox_dual(q+sigma*res))/sigma
        residual=torch.maximum(torch.maximum(ru/(1+norm(u)+norm(gu)),rp/(1+norm(p)+norm(gp))),rq/(1+norm(q)+norm(res)))
        feasible=torch.clamp(norm(u-x)-radius,min=0.)/norm(x).clamp_min(1e-30)
        feasible=torch.maximum(feasible,torch.maximum(torch.clamp(lo-u.amin(0),min=0.),torch.clamp(u.amax(0)-hi,min=0.)))
        l1=coefficient*p.abs().sum(0);quadratic=beta*norm(res)**2
        values=lambda a:a.detach().cpu().reshape(-1).tolist()
        row=dict(iteration=offset+count,objective=values(l1+quadratic),l1_component=values(l1),quadratic_component=values(quadratic),
            feasibility=values(feasible),prox_residual=values(residual),elapsed_s=time.perf_counter()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(device))
        if not all(np.isfinite(row[k]).all() for k in ('objective','feasibility','prox_residual')):raise FloatingPointError('nonfinite IC8/GLR state')
        history.append(row)
        if callback:callback(row)
        converged=bool(torch.all(residual<tolerance).item() and torch.all(feasible<1e-9).item())
        if converged:break
    torch.cuda.synchronize(device)
    return dict(U=u.cpu().numpy(),P=p.cpu().numpy(),dual=q.cpu().numpy(),iterations=history[-1]['iteration'],history=history,
        objective=history[-1]['objective'],l1_component=history[-1]['l1_component'],quadratic_component=history[-1]['quadratic_component'],
        feasibility=history[-1]['feasibility'],stationarity=history[-1]['prox_residual'],converged=converged,
        aux_mode=aux_mode,alpha=None if disabled else coefficient,beta=beta,tau=float(tau),sigma=float(sigma),
        step_scale=scale,step_product_times_norm_bound=float(tau*sigma*(2*graph.max_degree+1)),
        radius=values(radius),box=[values(lo),values(hi)],cuda_graph_used=replay is not None,cuda_graph_error=capture_error,
        guarded_chunk_replays=replays,solve_time_s=time.perf_counter()-started)
