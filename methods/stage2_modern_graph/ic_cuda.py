"""Float64 CUDA execution of the P1 PDHG iterates, including both scalar axes.

CUDA graphs batch 50 unchanged steps. Scalar box violations are detected on
device; that chunk is restored and replayed with the original exact projection.
"""
import time

import numpy as np
import torch
from scipy import sparse

from methods.stage2_modern_graph.ic_solver import project_observation


def csr_tensor(matrix,device):
    matrix=sparse.csr_matrix(matrix,dtype=np.float64)
    matrix.sort_indices()
    return torch.sparse_csr_tensor(torch.as_tensor(matrix.indptr.astype(np.int64),device=device),
        torch.as_tensor(matrix.indices.astype(np.int64),device=device),
        torch.as_tensor(matrix.data,device=device),size=matrix.shape,dtype=torch.float64,device=device,check_invariants=True)


class CudaICGraph:
    def __init__(self,d,weights,device='cuda:0'):
        self.device=torch.device(device)
        d=sparse.csr_matrix(d,dtype=np.float64)
        self.d=csr_tensor(d,self.device);self.dt=csr_tensor(d.T,self.device)
        self.l=csr_tensor(d.T@sparse.diags(weights)@d,self.device)
        self.w=torch.as_tensor(weights[:,None],dtype=torch.float64,device=self.device)
        self.sqrtw=self.w.sqrt()
        self.max_degree=float(np.asarray(abs(d).T@weights).max())
        self.max_unweighted_degree=float(np.asarray(abs(d).sum(axis=0)).max())
        self.n,self.m=d.shape[1],d.shape[0]


def solve_ic_cuda(x,graph,*,alpha,eta,kind,group,max_iter=30000,tolerance=2e-6,
                  use_cuda_graph=True,balance_steps=True,callback=None):
    device=graph.device
    x_cpu=np.asarray(x,dtype=np.float64)
    x=torch.as_tensor(x_cpu,device=device)
    if x.ndim==1:x=x[:,None];x_cpu=x_cpu[:,None]
    beta=1-alpha
    bound=2*graph.max_degree+1 if kind=='IC8' else (2*graph.max_degree)**2+2*graph.max_unweighted_degree
    base_step=.95/np.sqrt(bound)
    norm=lambda v: torch.linalg.vector_norm(v) if group else torch.linalg.vector_norm(v,dim=0)
    radius=eta*norm(x-x.mean(dim=0))
    step_scale=max(1.,float(radius.amin().item())/beta) if balance_steps else 1.
    tau=base_step*step_scale;sigma=base_step/step_scale
    lo=x.amin(dim=0)-.5*(x.amax(dim=0)-x.amin(dim=0))
    hi=x.amax(dim=0)+.5*(x.amax(dim=0)-x.amin(dim=0))
    u=x.clone();p=torch.zeros((graph.m,x.shape[1]),dtype=torch.float64,device=device)
    q=torch.zeros((graph.m if kind=='IC8' else graph.n,x.shape[1]),dtype=torch.float64,device=device)
    ub=u.clone();pb=p.clone();box_bad=torch.zeros((),dtype=torch.bool,device=device)
    aux_bad=torch.zeros((),dtype=torch.bool,device=device)
    batch=50;history=[];fallback_chunks=0;graph_error=None

    def apply(a,b):
        if kind=='IC8':return graph.sqrtw*torch.sparse.mm(graph.d,a)-b
        return torch.sparse.mm(graph.l,a)-torch.sparse.mm(graph.dt,b)

    def adjoint(a):
        if kind=='IC8':return torch.sparse.mm(graph.dt,graph.sqrtw*a),-a
        return torch.sparse.mm(graph.l,a),-torch.sparse.mm(graph.d,a)

    def prox_dual(a):
        return a/(1+sigma/(2*beta)) if kind=='IC8' else a*torch.clamp(beta/norm(a).clamp_min(1e-30),max=1.)

    def prox_p(a):
        if group:return a*torch.clamp(1-tau*alpha/torch.linalg.vector_norm(a,dim=1,keepdim=True).clamp_min(1e-30),min=0.)
        return a.sign()*torch.clamp(a.abs()-tau*alpha,min=0.)

    def project(a,exact):
        diff=a-x
        value=x+diff*torch.clamp(radius/norm(diff).clamp_min(1e-30),max=1.)
        if not group:
            bad=torch.any((value<lo)|(value>hi));box_bad.logical_or_(bad)
            if exact and bool(bad.item()):
                raw=a.detach().cpu().numpy()
                projected=np.column_stack([project_observation(raw[:,i:i+1],x_cpu[:,i:i+1],float(radius[i].item()),
                    (float(lo[i].item()),float(hi[i].item()))).ravel() for i in range(x.shape[1])])
                value=torch.as_tensor(projected,device=device)
        return value

    def step(exact=False,zero_aux=False):
        if zero_aux:
            q.copy_(prox_dual(q+sigma*torch.sparse.mm(graph.l,ub)))
            # A sufficient certificate for prox(P+tau Dq)=0 at P=0:
            # max_e ||(Dq)_e|| <= 2 max_i ||q_i|| <= alpha.
            largest=torch.linalg.vector_norm(q,dim=1).amax() if group else q.abs().amax()
            aux_bad.logical_or_(2*largest>alpha)
            un=project(u-tau*torch.sparse.mm(graph.l,q),False)
            ub.copy_(2*un-u);u.copy_(un)
            return
        q.copy_(prox_dual(q+sigma*apply(ub,pb)))
        gu,gp=adjoint(q)
        un=project(u-tau*gu,exact);pn=prox_p(p-tau*gp)
        ub.copy_(2*un-u);pb.copy_(2*pn-p);u.copy_(un);p.copy_(pn)

    replay=None;zero_replay=None;zero_aux_active=kind=='IC9';zero_aux_steps=0
    if use_cuda_graph:
        initial=[a.clone() for a in (u,p,q,ub,pb)]
        try:
            stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):step()
            torch.cuda.current_stream(device).wait_stream(stream)
            for a,b in zip((u,p,q,ub,pb),initial):a.copy_(b)
            replay=torch.cuda.CUDAGraph()
            with torch.cuda.graph(replay):
                for _ in range(batch):step()
            for a,b in zip((u,p,q,ub,pb),initial):a.copy_(b)
            if kind=='IC9':
                zero_replay=torch.cuda.CUDAGraph()
                with torch.cuda.graph(zero_replay):
                    for _ in range(batch):step(zero_aux=True)
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():raise
            replay=None;zero_replay=None;graph_error=str(error)
        for a,b in zip((u,p,q,ub,pb),initial):a.copy_(b)
        del initial
        box_bad.zero_();aux_bad.zero_()
    torch.cuda.synchronize(device);started=time.perf_counter()
    converged=False
    for offset in range(0,max_iter,batch):
        count=min(batch,max_iter-offset);box_bad.zero_();aux_bad.zero_()
        if replay is not None and count==batch:
            fast=zero_aux_active and zero_replay is not None
            saved=None if group and not fast else [a.clone() for a in (u,p,q,ub,pb)]
            (zero_replay if fast else replay).replay()
            bad=(fast and bool(aux_bad.item())) or (not group and bool(box_bad.item()))
            if bad:
                for a,b in zip((u,p,q,ub,pb),saved):a.copy_(b)
                for _ in range(batch):step(True)
                fallback_chunks+=1
                zero_aux_active=False
            elif fast:zero_aux_steps+=batch
            del saved
        else:
            for _ in range(count):step(True)
        gu,gp=adjoint(q);res=apply(u,p)
        ru=norm(u-project(u-tau*gu,True))/tau
        rp=norm(p-prox_p(p-tau*gp))/tau
        rq=norm(q-prox_dual(q+sigma*res))/sigma
        residual=torch.maximum(torch.maximum(ru/(1+norm(u)+norm(gu)),rp/(1+norm(p)+norm(gp))),rq/(1+norm(q)+norm(res)))
        feasibility=torch.clamp(norm(u-x)-radius,min=0.)/norm(x).clamp_min(1e-30)
        if not group:
            feasibility=torch.maximum(feasibility,torch.maximum(torch.clamp(lo-u.amin(dim=0),min=0.),torch.clamp(u.amax(dim=0)-hi,min=0.)))
        sparsity=torch.linalg.vector_norm(p,dim=1).sum() if group else p.abs().sum(dim=0)
        objective=alpha*sparsity+beta*(norm(res)**2 if kind=='IC8' else norm(res))
        def values(a):return a.detach().cpu().reshape(-1).tolist()
        row=dict(iteration=offset+count,objective=values(objective),feasibility=values(feasibility),prox_residual=values(residual),
                 elapsed_s=time.perf_counter()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(device))
        if not all(np.isfinite(v).all() for v in (row['objective'],row['feasibility'],row['prox_residual'])):
            raise FloatingPointError('nonfinite IC objective/residual')
        history.append(row)
        if callback:callback(row)
        converged=bool(torch.all(residual<tolerance).item() and torch.all(feasibility<1e-9).item())
        if converged:break
    torch.cuda.synchronize(device)
    return dict(U=u.cpu().numpy(),P=p.cpu().numpy(),dual=q.cpu().numpy(),history=history,iterations=history[-1]['iteration'],
        objective=history[-1]['objective'],feasibility=history[-1]['feasibility'],stationarity=history[-1]['prox_residual'],
        converged=converged,tau=float(tau),sigma=float(sigma),radius=values(radius),
        step_scale=float(step_scale),step_product_times_norm_bound=float(tau*sigma*bound),
        box=None if group else [values(lo),values(hi)],cuda_graph_used=replay is not None,cuda_graph_error=graph_error,
        guarded_chunk_replays=fallback_chunks,certified_zero_aux_steps=zero_aux_steps,
        solve_time_s=time.perf_counter()-started)
