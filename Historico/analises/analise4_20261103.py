"""60.000 passos no expert e estudo de dados do BC: só lê os .pt de experimentos/ (não treina, não avalia, não usa datasets).
Rodar da raiz do projeto:  python Historico/analises/analise4_20261103.py"""
import json, numpy as np, torch
E = "experimentos"
lp = lambda a: torch.load(a, weights_only=False)
novo, velho, s0 = lp(f"{E}/iql_60k_expert/figuras_iql_cartpole/modelos.pt"), lp(f"{E}/iql_60k/figuras_iql_cartpole/modelos.pt"), lp(f"{E}/iql_s0/figuras_iql_cartpole/modelos.pt")
bc = lp(f"{E}/bc_dados/figuras_bc_cartpole/modelos_dados.pt")
pega = lambda res, ds, rot: next(r for r in res if r["dataset"] == ds and r["rotulo"] == rot)
def achatar(x): return [t for v in x.values() for t in achatar(v)] if isinstance(x, dict) else [x]
F = lambda t: 1 + 0.99 * 100 * (1 - np.exp(-t / 20_000))

print("=== 1. os modelos do ruim repetem os da rodada iql_60k?")
iguais = [(r["rotulo"], all(torch.equal(a, b) for a, b in zip(achatar(r["pesos"]), achatar(pega(velho, "ruim", r["rotulo"])["pesos"]))), r["retornos"] == pega(velho, "ruim", r["rotulo"])["retornos"]) for r in novo if r["dataset"] == "ruim"]
print(iguais)

print("\n=== 2. os primeiros 15.000 passos no expert repetem a rodada de 15.000?")
for rot in ("CQL (α=0.5)", "IQL (τ=0.7)", "BC"):
    a = {l["passo"]: l for l in pega(novo, "expert", rot)["log"]}; b = {l["passo"]: l for l in pega(s0, "expert", rot)["log"]}
    com = sorted(set(a) & set(b)); chave = "q_dados" if "q_dados" in b[com[0]] else "concordancia"
    print(rot, "passos em comum:", len(com), "iguais em", chave, ":", sum(a[p][chave] == b[p][chave] for p in com))

print("\n=== 3. Q no expert contra a fórmula (seed 0)")
P = (3750, 7500, 15000, 22500, 30000, 37500, 45000, 52500, 60000)
print("passo     ", " ".join(f"{p:8d}" for p in P))
print("fórmula   ", " ".join(f"{F(p):8.1f}" for p in P))
for rot in ("CQL (α=0.5)", "IQL (τ=0.7)"):
    L = {l["passo"]: l for l in pega(novo, "expert", rot)["log"]}
    disp = [p for p in P if p in L]
    print(f"{rot:12s}", "q_dados", " ".join(f"{p}:{L[p]['q_dados']:.1f}" for p in disp))
    print(f"{rot:12s}", "q_outra", " ".join(f"{p}:{L[p]['q_outra']:.1f}" for p in disp))
    print(f"{rot:12s}", "concord", " ".join(f"{p}:{L[p]['concordancia']:.3f}" for p in disp))
    print(f"{rot:12s}", "retorno", " ".join(f"{l['passo']}:{l['retorno']:.0f}" for l in pega(novo, "expert", rot)["log"] if "retorno" in l))
    fim = pega(novo, "expert", rot)["log"][-1]; print(f"{rot:12s} final: q_dados {fim['q_dados']:.2f}  fórmula {F(60000):.2f}  diferença {fim['q_dados'] - F(60000):+.2f}")
n = json.load(open(f"{E}/iql_60k_expert/figuras_iql_cartpole/numeros.json", encoding="utf-8"))
print("retornos:", {k: v for k, v in n.items() if k.startswith("retorno") and "expert" in str(v)})

print("\n=== 4. estudo de dados do BC, por seed (parada antecipada + regularização)")
nomes = sorted({r["nome"] for r in bc}, key=lambda x: [r["nome"] for r in bc].index(x))
for nome in nomes:
    g = sorted((r for r in bc if r["nome"] == nome), key=lambda r: r["seed"]); v = [r["retorno"] for r in g]
    print(f"{nome:24s} eps de treino {g[0]['n_eps']:5d}  retornos " + " ".join(f"{x:6.1f}" for x in v) + f"  média {np.mean(v):6.1f}  mín {min(v):6.1f}  máx {max(v):6.1f}  passo guardado {[r['passo_guardado'] for r in g]}  concord esp {[round(r['val']['acc_esp'], 3) for r in g]}")
print("\nchaves:", list(bc[0].keys()))
