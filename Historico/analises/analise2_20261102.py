import sys, numpy as np, torch
sys.path.insert(0, "Código/IQL")
import cartpole_comum as C, iql_cartpole as I
SP = sys.argv[1]
N = torch.load("experimentos/run_20261102/figuras_iql_cartpole/modelos.pt", weights_only=False)
Q = torch.load("experimentos/run_20261102/figuras_cql_cartpole/modelos_expert.pt", weights_only=False)
pega = lambda ds, rot: next(r for r in N if r["dataset"] == ds and r["rotulo"] == rot)
prev = lambda t: 1 + 0.99 * 100 * (1 - np.exp(-t / 20_000))   # Polyak 0.005, γ 0.99, r = 1, no terminals

print("== Q on expert vs closed form Q(t) = 1 + 0.99*100*(1-exp(-t/20000)) (no penalty term in the formula)")
ts = [3750, 7500, 11250, 15000]
print(f"{'formula':16s}", " ".join(f"{prev(t):6.1f}" for t in ts))
def serie(L, chave): return [min(L, key=lambda l: abs(l["passo"] - t))[chave] for t in ts]
for al in sorted(Q):
    if al > 0: print(f"CQL α={al:<10g}", " ".join(f"{x:6.1f}" for x in serie(Q[al]["log"], "q_dados")), f"  other action final {Q[al]['log'][-1]['q_outra']:7.1f}")
print(f"{'IQL τ=0.7':16s}", " ".join(f"{x:6.1f}" for x in serie(pega('expert','IQL (τ=0.7)')['log'], 'q_dados')))
print("steps for Q to reach 95 / 99 per the formula:", int(-20000*np.log(1-(95-1)/99)), int(-20000*np.log(1-(99-1)/99)))

print("\n== CQL on misto and ruim around the drop")
for ds in ("misto", "ruim"):
    print(ds); print(" passo retorno q_dados q_outra conc penal perda_td")
    for l in pega(ds, "CQL (α=0.5)")["log"]:
        if "retorno" in l:
            print(f"{l['passo']:6d} {l['retorno']:6.0f} {l['q_dados']:7.1f} {l['q_outra']:7.1f} {l['concordancia']:5.2f} {l['penalidade']:5.2f} {l['perda_td']:7.3f}")

print("\n== redraw fig 5 from the real .pt into scratch (perturbed values come from the saved keys)")
ds = ["expert", "medio", "ruim", "misto"]; dados = {d: C.carregar(d) for d in ds}
orig = C.avaliar
def proibido(*a, **k): raise AssertionError("no evaluation expected here")
C.avaliar = proibido
num = I.gerar_figuras(N, dados, dict(datasets=ds, analise="ruim", alpha=.5, tau=.7, beta=1.0), f"{SP}/fig_iql_real")
print("redrawn; episodes", num["episodios"], "hp_BC", num["hp_BC"])
