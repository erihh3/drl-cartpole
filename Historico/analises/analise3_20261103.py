"""Cinco seeds, 60.000 passos e varredura: só lê os .pt de experimentos/ (não treina, não avalia, não usa datasets).
Rodar da raiz do projeto:  python Historico/analises/analise3_20261103.py"""
import csv, numpy as np, torch
E = "experimentos"; SEEDS = range(5); DS = ["expert", "medio", "ruim", "misto"]
CH = {"BC": "BC", "CQL": "CQL (α=0.5)", "IQL": "IQL (τ=0.7)"}
def lp(a): return torch.load(a, weights_only=False)
iql = {s: lp(f"{E}/iql_s{s}/figuras_iql_cartpole/modelos.pt") for s in SEEDS}
cql = {s: lp(f"{E}/cql_s{s}/figuras_cql_cartpole/modelos_expert.pt") for s in SEEDS}
longo = lp(f"{E}/iql_60k/figuras_iql_cartpole/modelos.pt"); velho = lp(f"{E}/run_20261102/figuras_iql_cartpole/modelos.pt")
def pega(res, ds, rot): return next(r for r in res if r["dataset"] == ds and r["rotulo"] == rot)
def achatar(x): return [t for v in x.values() for t in achatar(v)] if isinstance(x, dict) else [x]
def desc(v): v = np.array(v, float); return f"{v.mean():7.1f} ± {v.std(ddof=1):6.1f}  [{v.min():6.1f}, {v.max():6.1f}]  " + " ".join(f"{x:6.1f}" for x in v)

print("=== 1. seed 0 em duas máquinas: run_20261102 (uma máquina) contra iql_s0 (outra)")
iguais = dif = pesos_iguais = 0; difs = []; max_dif = 0.0
for r in velho:
    n = pega(iql[0], r["dataset"], r["rotulo"])
    for chave in ("retornos", "retornos_perturbados"):
        if r[chave] == n[chave]: iguais += 1
        else: dif += 1; difs.append((r["dataset"], r["rotulo"], chave, sum(a != b for a, b in zip(r[chave], n[chave])), float(np.abs(np.array(r[chave]) - np.array(n[chave])).max())))
    A, B = achatar(r["pesos"]), achatar(n["pesos"]); pesos_iguais += all(torch.equal(a, b) for a, b in zip(A, B)); max_dif = max(max_dif, max(float((a - b).abs().max()) for a, b in zip(A, B)))
print(f"listas de retornos idênticas: {iguais}; diferentes: {dif}"); [print("  ", d) for d in difs]
print(f"modelos com pesos idênticos: {pesos_iguais} de {len(velho)}; maior diferença em um peso: {max_dif:.3g}")

print("\n=== 2. retorno por método e dataset (média ± dp [mín, máx]  s0..s4)")
for ds in DS:
    for m, rot in CH.items():
        print(f"{ds:7s}{m:4s} ret {desc([pega(iql[s], ds, rot)['retorno'] for s in SEEDS])}")
print()
for ds in DS:
    for m, rot in CH.items():
        print(f"{ds:7s}{m:4s} per {desc([pega(iql[s], ds, rot)['retorno_perturbado'] for s in SEEDS])}")

print("\n=== 3. IQL contra CQL, seed a seed (pareado)")
for ds in ("medio", "ruim", "misto"):
    d = np.array([pega(iql[s], ds, CH["IQL"])["retorno"] - pega(iql[s], ds, CH["CQL"])["retorno"] for s in SEEDS])
    print(f"{ds:7s} dif {d.mean():6.1f} ± {d.std(ddof=1):5.1f}; IQL maior em {(d > 0).sum()}/5")

print("\n=== 4. tau no ruim")
for tau in (0.5, 0.7, 0.9):
    rot = f"IQL (τ={tau})"
    print(f"tau {tau}: ret {desc([pega(iql[s], 'ruim', rot)['retorno'] for s in SEEDS])}")
for a, b in ((0.5, 0.7), (0.7, 0.9), (0.5, 0.9)):
    d = np.array([pega(iql[s], "ruim", f"IQL (τ={a})")["retorno"] - pega(iql[s], "ruim", f"IQL (τ={b})")["retorno"] for s in SEEDS])
    print(f"tau {a} - tau {b}: {d.mean():6.1f} ± {d.std(ddof=1):5.1f}; positivo em {(d > 0).sum()}/5")
print("60k, seed 0:", {t: round(pega(longo, "ruim", f"IQL (τ={t})")["retorno"], 1) for t in (0.5, 0.7, 0.9)})
def serie(r, c): return [(l["passo"], l[c]) for l in r["log"] if c in l]
for t in (0.5, 0.7, 0.9):
    r = pega(longo, "ruim", f"IQL (τ={t})"); print(f"  60k tau {t} retorno:", " ".join(f"{p//1000}k:{v:.0f}" for p, v in serie(r, "retorno")))
    print(f"           V:      ", " ".join(f"{p//1000}k:{v:.0f}" for p, v in serie(r, "v")[9::10]))

print("\n=== 5. DQN (alfa = 0) e CQL no expert, por seed")
for s in SEEDS:
    d = cql[s][0.0]; c = cql[s][0.1]
    fin = lambda r, k: [l[k] for l in r["log"] if k in l][-1]
    print(f"s{s}: DQN ret {d['retorno']:5.2f} Q dados {fin(d,'q_dados'):11.1f} Q outra {fin(d,'q_outra'):11.1f} conc {fin(d,'concordancia'):.3f} | CQL 0,1 ret {c['retorno']:.0f} Q dados {fin(c,'q_dados'):6.1f} Q outra {fin(c,'q_outra'):6.1f} conc {fin(c,'concordancia'):.3f}")
    print("     retornos alfa>0:", [cql[s][a]["retorno"] for a in (0.1, 0.5, 2.0, 5.0)])

print("\n=== 6. curvas de retorno no misto e no medio (10 avaliações por modelo, n=10 episódios)")
for ds in ("misto", "medio", "ruim"):
    for m in ("CQL", "IQL"):
        for s in SEEDS:
            r = pega(iql[s], ds, CH[m]); c = serie(r, "retorno"); v = [x for _, x in c]; pi = int(np.argmax(v))
            print(f"{ds:6s}{m:4s}s{s}: " + " ".join(f"{x:4.0f}" for x in v) + f"  | pico {v[pi]:.0f} em {c[pi][0]}, final(50ep) {r['retorno']:.0f}")
print("\nmisto 60k seed 0:")
for m in ("CQL", "IQL", "BC"):
    c = serie(pega(longo, "misto", CH[m]), "retorno"); print(f"{m:4s}", " ".join(f"{p//1000}k:{v:.0f}" for p, v in c))
print("ruim 60k seed 0:")
for m in ("CQL", "IQL", "BC"):
    c = serie(pega(longo, "ruim", CH[m]), "retorno"); print(f"{m:4s}", " ".join(f"{p//1000}k:{v:.0f}" for p, v in c))

print("\n=== 7. Q e V ao longo dos 60.000 passos (seed 0, amostra de monitoramento do dataset)")
for ds in ("ruim", "misto"):
    for rot in (CH["CQL"], CH["IQL"]):
        r = pega(longo, ds, rot)
        for k in ("q_dados", "q_outra", "v"):
            c = serie(r, k)
            if c: print(f"{ds:6s}{rot:13s}{k:8s}", " ".join(f"{p//1000}k:{v:.1f}" for p, v in c[9::10]))
print("15k, mesma seed:")
for ds in ("ruim", "misto"):
    for rot in (CH["CQL"], CH["IQL"]):
        r = pega(iql[0], ds, rot)
        for k in ("q_dados", "q_outra", "v"):
            c = serie(r, k)
            if c: print(f"{ds:6s}{rot:13s}{k:8s} final {c[-1][1]:.1f}")

print("\n=== 8. varredura, por k e seed")
with open(f"{E}/varredura_especialista/varredura.csv", encoding="utf-8") as f: V = list(csv.DictReader(f))
for m in ("BC", "CQL", "IQL"):
    print(m)
    for k in sorted({int(l["k"]) for l in V}):
        ls = sorted((l for l in V if l["metodo"] == m and int(l["k"]) == k), key=lambda l: int(l["seed"]))
        print(f"  k={k:2d} {float(ls[0]['frac_transicoes'])*100:5.1f}% ret " + " ".join(f"{float(l['retorno']):6.1f}" for l in ls) + "  per " + " ".join(f"{float(l['retorno_perturbado']):6.1f}" for l in ls) + f"  media {np.mean([float(l['retorno']) for l in ls]):6.1f} origem {ls[0]['origem']}")

print("\n=== 9. o DQN vira uma ação fixa? (contagem de ações na avaliação final)")
for s in SEEDS:
    d = cql[s][0.0]; print(f"s{s}: ret {d['retorno']:.2f}  contagem de ações [esquerda, direita] = {d['acoes']}")

print("\n=== 10. a queda do CQL, verificada por seed (curva de treino, n = 10 episódios por ponto)")
for ds in ("misto", "ruim"):
    C_ = {s: serie(pega(iql[s], ds, CH["CQL"]), "retorno") for s in SEEDS}
    passos = [p for p, _ in C_[0]]
    print(ds, "passos:", passos)
    print("  em 500 nas 5 seeds, nos passos:", [p for j, p in enumerate(passos) if all(C_[s][j][1] == 500 for s in SEEDS)])
    print("  média por passo:", " ".join(f"{np.mean([C_[s][j][1] for s in SEEDS]):.0f}" for j in range(len(passos))))
    print("  último ponto abaixo do primeiro em", sum(C_[s][-1][1] < C_[s][0][1] for s in SEEDS), "de 5;  último abaixo do pico em", sum(C_[s][-1][1] < max(v for _, v in C_[s]) for s in SEEDS), "de 5")
print("BC no misto, média por passo:", " ".join(f"{np.mean([serie(pega(iql[s], 'misto', 'BC'), 'retorno')[j][1] for s in SEEDS]):.0f}" for j in range(10)))

print("\n=== 11. IQL no medio com perturbação: por que 490,2 em toda seed?")
for s in SEEDS:
    r = pega(iql[s], "medio", CH["IQL"])["retornos_perturbados"]; print(f"s{s}: abaixo de 500: {[(k, x) for k, x in enumerate(r) if x < 500]}")

print("\n=== 12. o teste real na GPU (2.000 passos, 5 episódios): CPU contra CUDA, mesmas seeds")
cpu = lp(f"{E}/teste_cpu_8/figuras_iql_cartpole/modelos.pt"); gpu = lp(f"{E}/teste_cuda_8/figuras_iql_cartpole/modelos.pt"); gpu4 = lp(f"{E}/teste_cuda_4/figuras_iql_cartpole/modelos.pt")
print("dispositivo:", {r["dispositivo"] for r in cpu}, {r["dispositivo"] for r in gpu}, {r["dispositivo"] for r in gpu4})
print("pesos salvos em:", {t.device.type for r in gpu for t in achatar(r["pesos"])})
pi = iguais = 0
for r in cpu:
    n = pega(gpu, r["dataset"], r["rotulo"]); m = pega(gpu4, r["dataset"], r["rotulo"])
    print(f"  {r['dataset']:7s}{r['rotulo']:14s} cpu {r['retorno']:6.1f}  cuda8 {n['retorno']:6.1f}  cuda4 {m['retorno']:6.1f}")
    pi += all(torch.equal(a, b) for a, b in zip(achatar(n["pesos"]), achatar(m["pesos"])))
print(f"pesos iguais entre cuda 8 e cuda 4: {pi} de {len(cpu)}")
print("diferença máxima de peso entre cpu e cuda:", max(float((a - b).abs().max()) for r in cpu for a, b in zip(achatar(r["pesos"]), achatar(pega(gpu, r["dataset"], r["rotulo"])["pesos"]))))

print("\n=== 13. Q(s, a) nos estados do especialista dentro do misto (seed 0), contra a fórmula do capítulo 6")
import sys
sys.path.insert(0, "Código/IQL")
import cartpole_comum as C, iql_cartpole as I
exp = C.carregar("expert"); S_ = torch.as_tensor(exp["s"]); A_ = torch.as_tensor(exp["a"])
form = lambda t: 1 + 0.99 * 100 * (1 - np.exp(-t / 20_000))
print(f"fórmula: 15.000 passos = {form(15_000):.1f}; 60.000 passos = {form(60_000):.1f}")
for nome, res, t in (("15k", iql[0], 15_000), ("60k", longo, 60_000)):
    for rot, algo in ((CH["CQL"], "CQL"), (CH["IQL"], "IQL")):
        ag = I.Agente.montar(algo, pega(res, "misto", rot)["pesos"])
        with torch.no_grad(): Q = ag.q(S_)
        qd = Q.gather(1, A_[:, None]).mean().item(); qo = Q.gather(1, (1 - A_)[:, None]).mean().item()
        print(f"{nome} {algo:4s} misto: Q ação do especialista {qd:6.1f}; Q outra ação {qo:6.1f}; concorda com o especialista {(Q.argmax(1) == A_).float().mean().item():.3f}")

print("\n=== 14. média das 5 seeds ao longo do treino: retorno e Q do CQL (e IQL) nos dados")
for ds in ("misto", "ruim"):
    for m in ("CQL", "IQL"):
        L = {s: [l for l in pega(iql[s], ds, CH[m])["log"] if "retorno" in l] for s in SEEDS}
        print(f"{ds:6s}{m:4s} passo   " + " ".join(f"{l['passo']:6d}" for l in L[0]))
        for k in ("retorno", "q_dados", "q_outra", "concordancia", "perda_td", "penalidade"):
            if k not in L[0][0]: continue
            print(f"{'':10s} {k:12s} " + " ".join(f"{np.mean([L[s][j][k] for s in SEEDS]):6.2f}" for j in range(10)))

print("\n=== 15. viés de Q nos dados do ruim (numeros.json), 15.000 passos nas 5 seeds e 60.000 na seed 0")
import json
N = {s: json.load(open(f"{E}/iql_s{s}/figuras_iql_cartpole/numeros.json", encoding="utf-8")) for s in SEEDS}; N60 = json.load(open(f"{E}/iql_60k/figuras_iql_cartpole/numeros.json", encoding="utf-8"))
for k in ("vies_Q_dados_CQL", "vies_Q_dados_IQL", "igual_especialista_BC", "igual_especialista_CQL", "igual_especialista_IQL", "peso_mediano_acao_certa", "peso_mediano_acao_errada"):
    print(f"{k:26s}", " ".join(f"{N[s][k]:8.3f}" for s in SEEDS), f"  média {np.mean([N[s][k] for s in SEEDS]):8.3f} dp {np.std([N[s][k] for s in SEEDS], ddof=1):7.3f}  60k {N60[k]:8.3f}")
for t in ("0.5", "0.7", "0.9"):
    print(f"V tau {t}:", " ".join(f"{N[s]['V_por_tau'][t]:6.1f}" for s in SEEDS), f"  60k {N60['V_por_tau'][t]:6.1f}")
NC = {s: json.load(open(f"{E}/cql_s{s}/figuras_cql_cartpole/numeros.json", encoding="utf-8")) for s in SEEDS}
for k in ("vies_DQN", "vies_CQL", "concordancia_DQN", "concordancia_CQL"):
    print(f"{k:20s}", " ".join(f"{NC[s][k]:12.3f}" for s in SEEDS), f"  média {np.mean([NC[s][k] for s in SEEDS]):.3f} dp {np.std([NC[s][k] for s in SEEDS], ddof=1):.3f}")
r0 = [cql[s][0.0]["retorno"] for s in SEEDS]; print("DQN retorno:", r0, f"média {np.mean(r0):.2f} dp {np.std(r0, ddof=1):.2f}")
