# Read-only: inspects saved results; trains nothing and evaluates no trained model in the env.
import sys, json, numpy as np, torch, gymnasium as gym
sys.path.insert(0, "Código/IQL")
import cartpole_comum as C, iql_cartpole as I
N = torch.load("experimentos/run_20261102/figuras_iql_cartpole/modelos.pt", weights_only=False)
O = torch.load("figuras_iql_cartpole/modelos.pt", weights_only=False)
ch = lambda r: (r["dataset"], r["rotulo"]); O_ = {ch(r): r for r in O}
boot = lambda x, rng=np.random.default_rng(0): np.percentile(rng.choice(x, (5000, len(x))).mean(1), [2.5, 97.5])

print("== 1. new run vs original: weights and first 30 episodes")
for r in N:
    o = O_[ch(r)]
    same = all(torch.equal(r["pesos"][k][n], o["pesos"][k][n]) for k in r["pesos"] for n in r["pesos"][k]) if set(r["pesos"]) == set(o["pesos"]) and r["algo"] != "BC" else None
    print(f"{r['dataset']:7s}{r['rotulo']:14s} weights_same={str(same):5s} first30_same={r['retornos'][:30]==o['retornos']!s:5s} old={o['retorno']:6.1f} new30={np.mean(r['retornos'][:30]):6.1f} new50={r['retorno']:6.1f}")

print("\n== 2. evaluation uncertainty (50 episodes) and paired perturbation effect")
print(f"{'':21s}{'std ret':>9s}{'95% CI':>15s}{'pert':>8s}{'95% CI':>15s}{'paired diff CI':>18s}  fell_early(<100) std/pert")
for r in N:
    a, b = np.array(r["retornos"]), np.array(r["retornos_perturbados"]); ca, cb, cd = boot(a), boot(b), boot(b - a)
    print(f"{r['dataset']:7s}{r['rotulo']:14s}{a.mean():9.1f}  [{ca[0]:5.0f},{ca[1]:5.0f}]{b.mean():8.1f}  [{cb[0]:5.0f},{cb[1]:5.0f}]    [{cd[0]:6.1f},{cd[1]:6.1f}]   {(a<100).sum():2d}/{(b<100).sum():2d}")

print("\n== 3. learning curves: return at the 10 checkpoints (10 episodes each)")
for r in N:
    L = [l for l in r["log"] if "retorno" in l]; v = np.array([l["retorno"] for l in L])
    print(f"{r['dataset']:7s}{r['rotulo']:14s}", " ".join(f"{x:4.0f}" for x in v), f"| last5 mean={v[-5:].mean():5.0f} std={v[-5:].std():4.0f} first_500_at={next((l['passo'] for l in L if l['retorno']>=500), None)}")

print("\n== 4. value drift: V / Q(dados) at 25%, 50%, 75%, 100% of training")
for r in N:
    if r["algo"] == "BC": continue
    L = r["log"]; idx = [len(L)//4 - 1, len(L)//2 - 1, 3*len(L)//4 - 1, len(L) - 1]
    print(f"{r['dataset']:7s}{r['rotulo']:14s} Q_dados:", " ".join(f"{L[i]['q_dados']:7.1f}" for i in idx), " Q_outra:", " ".join(f"{L[i]['q_outra']:7.1f}" for i in idx), (" V: " + " ".join(f"{L[i]['v']:6.1f}" for i in idx)) if r["algo"] == "IQL" else "")

print("\n== 5. why does BC on misto lose 2/3 under perturbation? agreement with the expert heuristic")
Dm = C.carregar("misto"); Ne = 25_000
def handoff(k, n=50):
    env = gym.make(C.ENV_ID); out = []
    for e in range(n):
        obs, _ = env.reset(seed=10_000 + e); g = np.random.default_rng(10_000 + e); fim = False
        for _ in range(k):
            if fim: break
            obs, *_x = env.step(int(g.integers(2))); fim = _x[1] or _x[2]
        out.append(obs)
    return np.array(out, np.float32)
print("misto transitions: expert", Ne, "ruim", len(Dm["a"]) - Ne, f"-> expert share {Ne/len(Dm['a']):.1%}; episodes 50 of {len(np.unique(Dm['ep']))} = {50/len(np.unique(Dm['ep'])):.1%}")
sub = np.random.default_rng(0).choice(len(Dm["a"]), 30000, replace=False); sc = Dm["s"].std(0)
def viz_expert(X):  # share of the 50 nearest dataset states that came from the expert
    d = ((X[:, None] / sc - Dm["s"][sub][None] / sc) ** 2).sum(-1); nn = np.argsort(d, 1)[:, :50]
    return (sub[nn] < Ne).mean(1)
for k in (0, 1, 2, 3, 5):
    H = handoff(k); ae = np.array([C.acao_especialista(o) for o in H]); ve = viz_expert(H)
    linha = f"k={k}: median share of expert among 50 NN={np.median(ve):4.0%} |"
    for rot in ("BC", "CQL (α=0.5)", "IQL (τ=0.7)"):
        r = next(x for x in N if x["dataset"] == "misto" and x["rotulo"] == rot); A = I.Agente.montar(r["algo"], r["pesos"])
        with torch.no_grad(): a = A.politica(torch.as_tensor(H)).argmax(1).numpy()
        linha += f" {rot[:3]} agrees {np.mean(a == ae):4.0%}"
    print(linha)

print("\n== 6. BC 3-seed spread, from the tracked bc numeros.json")
b = json.load(open("figuras_bc_cartpole/numeros.json", encoding="utf-8"))["variantes"]
for v, d in b.items(): print(f"{v:18s} ret={d['retorno'][0]:6.1f} ± {d['retorno'][1]:5.1f}  seeds={[round(x) for x in d['seeds']]}")

print("\n== 7. wind feasibility: expert heuristic only (no trained model). force = ±10 + w, exact via force_mag")
def episodio(seed, vento):
    env = gym.make(C.ENV_ID); obs, _ = env.reset(seed=seed); g = np.random.default_rng(seed); u = env.unwrapped; T = 0; fim = False; caiu_x = False
    while not fim:
        a = C.acao_especialista(obs); w = vento(g, T)
        u.force_mag = 10.0 + w if a == 1 else 10.0 - w
        obs, r, term, trunc, _ = env.step(a); T += 1; fim = term or trunc
    return T, abs(obs[0]) > 2.4
for nome, f in [("constant w=%g" % w, (lambda w: (lambda g, t: w))(w)) for w in (0, 0.5, 1, 2, 3, 4, 5, 7)] + \
               [("gust σ=%g (per step)" % s, (lambda s: (lambda g, t: g.normal(0, s)))(s)) for s in (2, 5, 10, 15, 20)]:
    R = [episodio(10_000 + e, f) for e in range(50)]; T = np.array([x[0] for x in R])
    print(f"{nome:24s} mean={T.mean():6.1f} min={T.min():4d}  survived={np.mean(T==500):4.0%}  left_track={np.mean([x[1] for x in R]):4.0%}")
