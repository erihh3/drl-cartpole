# Passo sintético do IQL (tensores aleatórios, sem datasets, sem o script de treino): conta as operações com tensores e mede o tempo na CPU.
import copy, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
torch.set_num_threads(1); torch.manual_seed(0)
class Conta(TorchDispatchMode):
    n = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        Conta.n += 1; return func(*args, **(kwargs or {}))
def mlp(o=2): return nn.Sequential(nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))
def suavizar(a, b, tau=0.005):
    with torch.no_grad():
        for p, q in zip(a.parameters(), b.parameters()): q.mul_(1 - tau).add_(tau * p)
def monta(foreach):
    q1, q2, v, pi = mlp(), mlp(), mlp(1), mlp(); a1, a2 = copy.deepcopy(q1), copy.deepcopy(q2)
    op = dict(q=torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 3e-4, foreach=foreach),
              v=torch.optim.Adam(v.parameters(), 3e-4, foreach=foreach), pi=torch.optim.Adam(pi.parameters(), 3e-4, foreach=foreach))
    return q1, q2, v, pi, a1, a2, op
def passo(q1, q2, v, pi, a1, a2, op):
    s, s2 = torch.randn(256, 4), torch.randn(256, 4); a = torch.randint(2, (256,)); r = torch.ones(256); d = torch.zeros(256)
    with torch.no_grad(): qa = torch.min(a1(s), a2(s)).gather(1, a[:, None]).squeeze(1)
    u = qa - v(s).squeeze(1); pv = (torch.abs(0.7 - (u < 0).float()) * u ** 2).mean()
    op["v"].zero_grad(); pv.backward(); op["v"].step()
    with torch.no_grad(): y = r + 0.99 * (1 - d) * v(s2).squeeze(1)
    pq = (F.mse_loss(q1(s).gather(1, a[:, None]).squeeze(1), y) + F.mse_loss(q2(s).gather(1, a[:, None]).squeeze(1), y)) / 2
    op["q"].zero_grad(); pq.backward(); op["q"].step(); suavizar(q1, a1); suavizar(q2, a2)
    with torch.no_grad(): w = torch.exp(qa - v(s).squeeze(1)).clamp(max=100.0)
    pp = (w * F.cross_entropy(pi(s), a, reduction="none")).mean()
    op["pi"].zero_grad(); pp.backward(); op["pi"].step()
    return pq.item(), pv.item(), pp.item(), w.mean().item()
for foreach in (False, True):
    m = monta(foreach); passo(*m)                # o primeiro passo cria o estado do Adam
    Conta.n = 0
    with Conta(): passo(*m)
    n = Conta.n
    t = time.perf_counter()
    for _ in range(300): passo(*m)
    ms = (time.perf_counter() - t) / 300 * 1e3
    print(f"foreach={foreach}: {n} operações por passo, {ms:.1f} ms por passo em 1 thread de CPU, {ms * 1e3 / n:.0f} µs por operação")
