"""
IQL (Implicit Q-Learning) offline no CartPole, comparado com o CQL e o BC.

Treina, com as mesmas redes, sementes e número de passos:
  - BC   (referência: só imita o dataset)
  - CQL  (perda TD + alpha * [logsumexp_a Q(s,a) - Q(s, a_dados)])
  - IQL  (V por expectil, Q por TD com V(s'), política por regressão ponderada pela vantagem)
  - varredura de tau do IQL (no dataset de análise)

Os datasets são a mesma heurística com uma fração das ações trocadas:
  expert (0%), medio (30%), ruim (60%) e misto (expert + ruim).
Com 30% de troca a ação certa ainda é a mais frequente em cada estado, então o argmax do BC recupera
o especialista. Com 60% a ação ERRADA é a mais frequente: o BC copia o erro, e só um método que olha
a recompensa (CQL, IQL) consegue sair disso. Por isso o dataset de análise padrão é o "ruim".

Figuras em ./figuras_iql_cartpole/:
  1_acoes_sem_dados.png  Q da outra ação e calibração (Q × retorno real)
  2_mecanismo_iql.png    V como expectil, vantagem e pesos da AWR
  3_politicas.png        concordância com o dataset e mapas de política (θ × ω)
  4_tau.png              varredura de tau, comparada com CQL e BC
  5_resultado.png        retorno por dataset e ângulo do poste num episódio

Uso:
  pip install "minari[create]" gymnasium torch matplotlib
  python iql_cartpole.py                                  # 4 datasets, análise no "ruim"
  python iql_cartpole.py --datasets ruim misto            # mais rápido
  python iql_cartpole.py --so-graficos
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch, torch.nn.functional as F

import cartpole_comum as C

torch.set_num_threads(1)
COR = {"BC": "#8A93A3", "CQL": "#1F7A6D", "IQL": "#B07A00", "dados": "#C9CED6"}


class Agente:
    """Redes de um método. politica(s) -> logits/valores para o argmax; q(s) -> Q(s,·) (None no BC);
    v(s) -> V(s) (só no IQL)."""

    def __init__(self, algo, redes):
        self.algo, self.redes = algo, redes
        self.net = self.redes["q"] if algo == "CQL" else self.redes["pi"]   # usado por C.avaliar/desenhar_mapa

    def reset(self, obs):
        pass

    def agir(self, obs, primeiro=False):
        with torch.no_grad():
            return int(self.net(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax())

    def politica(self, s):
        return self.net(s)

    def q(self, s):
        if self.algo == "CQL":
            return self.redes["q"](s)
        if self.algo == "IQL":
            return torch.min(self.redes["q1"](s), self.redes["q2"](s))
        return None

    def v(self, s):
        return self.redes["v"](s).squeeze(1) if self.algo == "IQL" else None

    def pesos(self):
        return {k: {n: p.cpu() for n, p in r.state_dict().items()} for k, r in self.redes.items()}

    @staticmethod
    def montar(algo, pesos):
        redes = {k: C.mlp(saida=1 if k == "v" else C.N_ACOES) for k in pesos}
        for k, r in redes.items():
            r.load_state_dict(pesos[k]); r.eval()
        return Agente(algo, redes)


def suavizar(rede, alvo, tau=0.005):
    with torch.no_grad():
        for p, pa in zip(rede.parameters(), alvo.parameters()):
            pa.mul_(1 - tau).add_(tau * p)


def rotulo(algo, hp):
    if algo == "CQL":
        return f"CQL (α={hp['alpha']:g})"
    if algo == "IQL":
        return f"IQL (τ={hp['tau']:g})"
    return "BC"


# =============================================================================== treino
def treinar(args):
    nome_ds, algo, hp, passos, seed, n_aval = args
    D = C.carregar(nome_ds)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    T = {k: torch.as_tensor(D[k]) for k in ("s", "a", "r", "s2", "d")}
    mon = np.random.default_rng(123).choice(len(D["a"]), 2000, replace=False)
    s_mon, a_mon = T["s"][mon], T["a"][mon]
    mlp = C.mlp

    if algo == "BC":
        redes = {"pi": mlp()}
        opt = {"pi": torch.optim.Adam(redes["pi"].parameters(), 1e-3)}
    elif algo == "CQL":
        redes = {"q": mlp()}; alvo = {"q": mlp()}; alvo["q"].load_state_dict(redes["q"].state_dict())
        opt = {"q": torch.optim.Adam(redes["q"].parameters(), 3e-4)}
    else:  # IQL
        redes = {"q1": mlp(), "q2": mlp(), "v": mlp(saida=1), "pi": mlp()}
        alvo = {k: mlp() for k in ("q1", "q2")}
        for k in alvo:
            alvo[k].load_state_dict(redes[k].state_dict())
        opt = {"q": torch.optim.Adam(list(redes["q1"].parameters()) + list(redes["q2"].parameters()), 3e-4),
               "v": torch.optim.Adam(redes["v"].parameters(), 3e-4),
               "pi": torch.optim.Adam(redes["pi"].parameters(), 3e-4)}
    agente = Agente(algo, redes)

    log, acum = [], {}
    log_cada, aval_cada = max(1, passos // 40), max(1, passos // 10)
    t0 = time.time()
    for t in range(1, passos + 1):
        i = torch.as_tensor(rng.integers(len(D["a"]), size=256))
        s, a, r, s2, d = (T[k][i] for k in ("s", "a", "r", "s2", "d"))
        m = {}
        if algo == "BC":
            perda = F.cross_entropy(redes["pi"](s), a)
            opt["pi"].zero_grad(); perda.backward(); opt["pi"].step()
            m["perda_bc"] = perda.item()

        elif algo == "CQL":
            q, qa = redes["q"], alvo["q"]
            with torch.no_grad():
                a2 = q(s2).argmax(1, keepdim=True)                                       # Double DQN
                y = r + C.GAMMA * (1 - d) * qa(s2).gather(1, a2).squeeze(1)
            q_s = q(s); q_sa = q_s.gather(1, a[:, None]).squeeze(1)
            perda_td = F.mse_loss(q_sa, y)
            penal = (torch.logsumexp(q_s, 1) - q_sa).mean()
            perda = perda_td + hp["alpha"] * penal
            opt["q"].zero_grad(); perda.backward(); opt["q"].step(); suavizar(q, qa)
            m.update(perda_td=perda_td.item(), penalidade=penal.item())

        else:  # IQL
            q1, q2, v, pi = redes["q1"], redes["q2"], redes["v"], redes["pi"]
            # 1) V(s): expectil tau de Q_alvo(s, a) com a vindo do DATASET
            with torch.no_grad():
                q_alvo = torch.min(alvo["q1"](s), alvo["q2"](s)).gather(1, a[:, None]).squeeze(1)
            u = q_alvo - v(s).squeeze(1)
            perda_v = (torch.abs(hp["tau"] - (u < 0).float()) * u ** 2).mean()
            opt["v"].zero_grad(); perda_v.backward(); opt["v"].step()
            # 2) Q(s, a) <- r + gamma V(s'): nenhum max sobre ações
            with torch.no_grad():
                y = r + C.GAMMA * (1 - d) * v(s2).squeeze(1)
            perda_q = (F.mse_loss(q1(s).gather(1, a[:, None]).squeeze(1), y) +
                       F.mse_loss(q2(s).gather(1, a[:, None]).squeeze(1), y)) / 2
            opt["q"].zero_grad(); perda_q.backward(); opt["q"].step()
            suavizar(q1, alvo["q1"]); suavizar(q2, alvo["q2"])
            # 3) política: BC ponderado por exp(beta * vantagem), peso limitado a 100
            with torch.no_grad():
                w = torch.exp(hp["beta"] * (q_alvo - v(s).squeeze(1))).clamp(max=100.0)
            perda_pi = (w * F.cross_entropy(pi(s), a, reduction="none")).mean()
            opt["pi"].zero_grad(); perda_pi.backward(); opt["pi"].step()
            m.update(perda_td=perda_q.item(), perda_v=perda_v.item(), perda_pi=perda_pi.item(),
                     peso_awr=w.mean().item())

        for k, val in m.items():
            acum.setdefault(k, []).append(val)
        if t % log_cada == 0 or t % aval_cada == 0:
            linha = {"passo": t, **{k: float(np.mean(v_)) for k, v_ in acum.items()}}
            acum = {}
            with torch.no_grad():
                linha["concordancia"] = (agente.politica(s_mon).argmax(1) == a_mon).float().mean().item()
                qm = agente.q(s_mon)
                if qm is not None:
                    linha["q_dados"] = qm.gather(1, a_mon[:, None]).mean().item()
                    linha["q_outra"] = qm.gather(1, (1 - a_mon)[:, None]).mean().item()
                if algo == "IQL":
                    linha["v"] = agente.v(s_mon).mean().item()
            if t % aval_cada == 0:
                linha["retorno"] = C.avaliar(agente, n=10, seed=50_000)[0]
            log.append(linha)

    ret, rets, acoes = C.avaliar(agente, n=n_aval)
    rot = rotulo(algo, hp)
    print(f"  {nome_ds:7s} {rot:14s} retorno={ret:6.1f}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(dataset=nome_ds, algo=algo, hp=hp, rotulo=rot, log=log, retorno=ret, retornos=rets.tolist(),
                acoes=acoes.tolist(), pesos=agente.pesos())


# =============================================================================== gráficos
def gerar_figuras(res, dados, cfg, pasta, mostrar=False):
    plt = C.estilo_graficos(mostrar)
    os.makedirs(pasta, exist_ok=True)
    ds0 = cfg["analise"]; D = dados[ds0]
    pega = lambda ds, rot: next(r for r in res if r["dataset"] == ds and r["rotulo"] == rot)
    rot_cql, rot_iql = rotulo("CQL", {"alpha": cfg["alpha"]}), rotulo("IQL", {"tau": cfg["tau"]})
    bc, cql, iql = pega(ds0, "BC"), pega(ds0, rot_cql), pega(ds0, rot_iql)
    A_bc, A_cql, A_iql = (Agente.montar(r["algo"], r["pesos"]) for r in (bc, cql, iql))
    numeros = {"dataset_analise": ds0, "beta": cfg["beta"]}

    idx = np.random.default_rng(7).choice(len(D["a"]), 5000, replace=False)
    S, A, G = torch.as_tensor(D["s"][idx]), torch.as_tensor(D["a"][idx]), D["G"][idx]
    A_esp = torch.as_tensor([C.acao_especialista(o) for o in D["s"][idx]])      # a ação "certa" em cada estado
    with torch.no_grad():
        QC, QI, VI = A_cql.q(S), A_iql.q(S), A_iql.v(S)
        PI_I, PI_B = F.softmax(A_iql.politica(S), 1), F.softmax(A_bc.politica(S), 1)
    numeros["frac_dataset_igual_especialista"] = float((A == A_esp).float().mean())

    def curva(ax, chave, nome, r, **kw):
        L = [l for l in r["log"] if chave in l]
        ax.plot([l["passo"] for l in L], [l[chave] for l in L], label=nome, **kw)

    def salvar(fig, nome, titulo):
        fig.suptitle(titulo, fontsize=13); fig.tight_layout(); fig.savefig(f"{pasta}/{nome}", dpi=120)

    # 1. ações sem dados / calibração
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for nome, r, c in [(rot_cql, cql, COR["CQL"]), (rot_iql, iql, COR["IQL"])]:
        curva(ax[0], "q_outra", f"{nome}: a outra ação", r, color=c, lw=2.2)
        curva(ax[0], "q_dados", f"{nome}: ação do dataset", r, color=c, lw=1.6, ls="--")
    curva(ax[0], "v", f"{rot_iql}: V(s)", iql, color="#14213D", lw=1.4, ls="-.")
    ax[0].axhline(float(G.mean()), color="k", lw=1, ls=":", label="retorno real médio (G)")
    ax[0].set(title="Q ao longo do treino", xlabel="passo de treino", ylabel="valor médio"); ax[0].legend(fontsize=9)
    sel = np.random.default_rng(0).choice(len(G), 1500, replace=False)
    for nome, Q, c in [(rot_cql, QC, COR["CQL"]), (rot_iql, QI, COR["IQL"])]:
        qd = Q.gather(1, A[:, None]).squeeze(1).numpy()
        vies = float(np.mean(qd - G)); numeros[f"vies_Q_dados_{nome[:3]}"] = vies
        ax[1].scatter(G[sel], qd[sel], s=7, alpha=0.35, color=c, label=f"{nome}: viés médio {vies:+.1f}")
    lim = [0, max(1, G.max())]
    ax[1].plot(lim, lim, "k--", lw=1, label="Q = retorno real")
    ax[1].set(title="Q(s, a_dados) previsto × retorno real G", xlabel="retorno descontado observado G",
              ylabel="Q(s, a do dataset)")
    ax[1].legend(fontsize=9, markerscale=3)
    salvar(fig, "1_acoes_sem_dados.png", f"1. Valores (dataset {ds0}): o CQL rebaixa a outra ação; "
                                         "o IQL só usa Q nas ações do dataset")

    # 2. mecanismo do IQL: um estado em que o dataset fez a ação errada e o IQL corrige
    errou = (A != A_esp).numpy() & (PI_I.argmax(1) == A_esp).numpy()
    cand = np.where(errou)[0]
    k = int(cand[np.argmax(PI_I[cand].max(1).values.numpy())]) if len(cand) else 0
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    qv, vv, a_d, a_e = QI[k].numpy(), float(VI[k]), int(A[k]), int(A_esp[k])
    b = ax[0].bar(C.NOMES_ACOES, qv, color=[COR["IQL"] if i == a_d else COR["dados"] for i in range(2)], width=0.55)
    ax[0].bar_label(b, fmt="%.2f", padding=2)
    ax[0].axhline(vv, color="k", ls="--", lw=1.2)
    ax[0].text(0.5, vv, f"V(s) = {vv:.2f}", ha="center", va="bottom", fontsize=9)
    lo = min(qv.min(), vv); hi = max(qv.max(), vv); pad = max(0.5, (hi - lo) * 0.6)
    ax[0].set_ylim(lo - pad, hi + pad)
    ax[0].set(title=f"Q do IQL num estado\nA(ação do dataset) = Q − V = {qv[a_d] - vv:+.2f}", ylabel="Q(s, a)")
    ax[0].set_xlabel(f"dataset: {C.NOMES_ACOES[a_d]} · especialista: {C.NOMES_ACOES[a_e]}", fontsize=9)
    x, w_ = np.arange(2), 0.38
    ax[1].bar(x - w_ / 2, PI_B[k].numpy(), w_, color=COR["BC"], label="BC (imitação pura)")
    ax[1].bar(x + w_ / 2, PI_I[k].numpy(), w_, color=COR["IQL"], label=rot_iql)
    ax[1].set_xticks(x, C.NOMES_ACOES); ax[1].set_ylim(0, 1)
    ax[1].set(title="Política no mesmo estado: π_IQL ∝ π_dados · exp(β·A)", ylabel="probabilidade")
    ax[1].legend(fontsize=9)
    with torch.no_grad():
        adv = (QI.gather(1, A[:, None]).squeeze(1) - VI).numpy()
    pesos = np.minimum(np.exp(cfg["beta"] * adv), 100)
    certo = (A == A_esp).numpy()
    bins = np.geomspace(max(pesos.min(), 1e-4), 100, 50)
    ax[2].hist(pesos[certo], bins=bins, color=COR["CQL"], alpha=0.7, label="dataset = especialista")
    ax[2].hist(pesos[~certo], bins=bins, color="#B04A16", alpha=0.7, label="dataset ≠ especialista")
    ax[2].axvline(1, color="k", ls="--", lw=1); ax[2].set_xscale("log")
    wc, we = float(np.median(pesos[certo])), float(np.median(pesos[~certo])) if (~certo).any() else float("nan")
    numeros.update(peso_mediano_acao_certa=wc, peso_mediano_acao_errada=we)
    ax[2].set(title=f"Pesos da AWR exp(β·A), β={cfg['beta']:g}\nmedianas: ação certa {wc:.2f} · errada {we:.2f}",
              xlabel="peso (escala log)", ylabel="nº de transições")
    ax[2].legend(fontsize=9)
    salvar(fig, "2_mecanismo_iql.png", "2. O mecanismo do IQL: a vantagem separa as ações boas das ruins do dataset "
                                      "e a AWR pesa cada uma por exp(β·A)")

    # 3. políticas
    fig, ax = plt.subplots(1, 4, figsize=(22, 5), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    for nome, r, c in [("BC", bc, COR["BC"]), (rot_cql, cql, COR["CQL"]), (rot_iql, iql, COR["IQL"])]:
        curva(ax[0], "concordancia", nome, r, color=c, lw=2)
    fe = numeros["frac_dataset_igual_especialista"]
    ax[0].axhline(fe, color="k", ls=":", lw=1, label=f"especialista concorda com o dataset: {fe:.0%}")
    ax[0].set(title="Concordância com a ação do dataset", xlabel="passo de treino", ylabel="fração")
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=9)
    for j, (nome, ag) in enumerate([("BC", A_bc), (rot_cql, A_cql), (rot_iql, A_iql)]):
        C.desenhar_mapa(ax[1 + j], ag.net, nome, D)
        with torch.no_grad():
            numeros[f"igual_especialista_{nome[:3]}"] = float((ag.politica(S).argmax(1) == A_esp).float().mean())
    salvar(fig, "3_politicas.png", f"3. As políticas no dataset {ds0}: quem copia o dataset e quem se aproxima "
                                   "do especialista (linha tracejada)")

    # 4. tau
    taus = sorted(r["hp"]["tau"] for r in res if r["dataset"] == ds0 and r["algo"] == "IQL")
    R_tau = [pega(ds0, rotulo("IQL", {"tau": t})) for t in taus]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    xs = np.arange(len(taus))
    b = ax[0].bar(xs, [r["retorno"] for r in R_tau], color=COR["IQL"], width=0.6)
    ax[0].bar_label(b, labels=[f"{r['retorno']:.0f}" for r in R_tau], padding=2)
    ax[0].axhline(cql["retorno"], color=COR["CQL"], lw=2, ls="--", label=f"{rot_cql}: {cql['retorno']:.0f}")
    ax[0].axhline(bc["retorno"], color=COR["BC"], lw=2, ls=":", label=f"BC: {bc['retorno']:.0f}")
    ax[0].set_xticks(xs, [f"τ = {t:g}" for t in taus]); ax[0].set_ylim(0, 540)
    ax[0].set(title="Retorno final do IQL por τ (máximo 500)", ylabel="retorno médio"); ax[0].legend(fontsize=9)
    vs, qds = [], []
    for r in R_tau:
        ag = Agente.montar("IQL", r["pesos"])
        with torch.no_grad():
            vs.append(ag.v(S).mean().item()); qds.append(ag.q(S).gather(1, A[:, None]).mean().item())
    ax[1].plot(xs, vs, "o-", color=COR["IQL"], lw=2, label="V(s) médio (expectil)")
    ax[1].plot(xs, qds, "s--", color="#14213D", lw=1.5, label="Q(s, a_dados) médio")
    ax[1].axhline(float(G.mean()), color="k", ls=":", lw=1, label="retorno real médio do dataset")
    ax[1].set_xticks(xs, [f"τ = {t:g}" for t in taus])
    ax[1].set(title="τ maior empurra V para o “quase máximo”", ylabel="valor médio"); ax[1].legend(fontsize=9)
    numeros["V_por_tau"] = dict(zip([f"{t:g}" for t in taus], vs))
    numeros["retorno_por_tau"] = {f"{r['hp']['tau']:g}": r["retorno"] for r in R_tau}
    for nome, r, c in [("BC", bc, COR["BC"]), (rot_cql, cql, COR["CQL"]), (rot_iql, iql, COR["IQL"])]:
        curva(ax[2], "retorno", nome, r, color=c, lw=2, marker="o", ms=3)
    ax[2].set(title="Retorno durante o treino (10 episódios)", xlabel="passo de treino", ylabel="retorno")
    ax[2].set_ylim(0, 540); ax[2].legend(fontsize=9)
    salvar(fig, "4_tau.png", f"4. O ajuste do IQL: τ (expectil) no dataset {ds0}, comparado com CQL e BC")

    # 5. resultado
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    dss = cfg["datasets"]; xs, w_ = np.arange(len(dss)), 0.26
    ret_dados = {ds: float(dados[ds]["ret_ep"][dados[ds]["ini"] == np.arange(len(dados[ds]["a"]))].mean()) for ds in dss}
    for j, (rot, c) in enumerate([("BC", COR["BC"]), (rot_cql, COR["CQL"]), (rot_iql, COR["IQL"])]):
        vals = [pega(ds, rot)["retorno"] for ds in dss]
        b = ax[0].bar(xs + (j - 1) * w_, vals, w_, color=c, label=rot)
        ax[0].bar_label(b, labels=[f"{v:.0f}" for v in vals], padding=2, fontsize=9)
        numeros[f"retorno_{rot}"] = dict(zip(dss, vals))
    ax[0].scatter(xs, [ret_dados[ds] for ds in dss], marker="_", s=900, color="k", lw=2, zorder=5,
                  label="retorno médio do dataset")
    numeros["retorno_dataset"] = ret_dados
    ax[0].set_xticks(xs, dss); ax[0].set_ylim(0, 660)
    ax[0].set(title=f"Retorno final por dataset ({cfg['episodios']} episódios, máximo 500)", ylabel="retorno médio")
    ax[0].legend(fontsize=9, loc="upper center", ncol=4)
    for nome, ag, c in [("BC", A_bc, COR["BC"]), (rot_cql, A_cql, COR["CQL"]), (rot_iql, A_iql, COR["IQL"])]:
        S_ep = C.rodar_episodio(ag, 10_000)
        ax[1].plot(S_ep[:, 2], color=c, lw=2, label=f"{nome}: {len(S_ep) - 1} passos")
    for y in (0.2095, -0.2095):
        ax[1].axhline(y, color="#999", ls=":", lw=1)
    ax[1].set(title=f"Ângulo do poste num episódio (dataset {ds0}, seed 10000)", xlabel="passo",
              ylabel="ângulo θ (rad)")
    ax[1].legend(fontsize=9)
    salvar(fig, "5_resultado.png", "5. O resultado no ambiente")

    with open(f"{pasta}/numeros.json", "w", encoding="utf-8") as fh:
        json.dump(numeros, fh, indent=2, ensure_ascii=False)
    if mostrar:
        plt.show()
    plt.close("all")
    return numeros


def main():
    C.utf8_no_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["expert", "medio", "ruim", "misto"],
                   choices=["expert", "medio", "ruim", "misto"])
    p.add_argument("--analise", default="ruim", help="dataset das figuras 1–4 e da varredura de τ")
    p.add_argument("--passos", type=int, default=15_000)
    p.add_argument("--alpha", type=float, default=0.5, help="alpha do CQL")
    p.add_argument("--tau", type=float, default=0.7, help="tau (expectil) principal do IQL")
    p.add_argument("--beta", type=float, default=1.0,
                   help="temperatura da AWR (no CartPole Q ~ 100, as vantagens são grandes: β menor que no MiniGrid)")
    p.add_argument("--taus-varredura", type=float, nargs="+", default=[0.5, 0.9])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodios", type=int, default=30, help="episódios na avaliação final")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--pasta", default="figuras_iql_cartpole")
    p.add_argument("--mostrar", action="store_true")
    p.add_argument("--so-graficos", action="store_true")
    a = p.parse_args()
    if a.analise not in a.datasets:
        a.datasets = [a.analise] + a.datasets

    C.garantir_datasets()
    dados = {ds: C.carregar(ds) for ds in a.datasets}
    for ds, D in dados.items():
        print(C.resumo_dataset(ds, D))
    os.makedirs(a.pasta, exist_ok=True)
    arq = os.path.join(a.pasta, "modelos.pt")
    cfg = dict(datasets=a.datasets, analise=a.analise, alpha=a.alpha, tau=a.tau, beta=a.beta, episodios=a.episodios)
    if a.so_graficos:
        res = torch.load(arq, weights_only=False)
    else:
        jobs = []
        for ds in a.datasets:
            taus = [a.tau] + ([t for t in a.taus_varredura if t != a.tau] if ds == a.analise else [])
            jobs += [(ds, "IQL", {"tau": t, "beta": a.beta}, a.passos, a.seed, a.episodios) for t in taus]
            jobs += [(ds, "CQL", {"alpha": a.alpha}, a.passos, a.seed, a.episodios),
                     (ds, "BC", {}, a.passos, a.seed, a.episodios)]
        print(f"treinando {len(jobs)} modelos ({a.passos} passos cada, {a.workers} processos)...")
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(treinar, jobs))
        torch.save(res, arq)
    print("gerando gráficos...")
    numeros = gerar_figuras(res, dados, cfg, a.pasta, a.mostrar)
    print(json.dumps(numeros, indent=2, ensure_ascii=False))
    print(f"figuras salvas em {os.path.abspath(a.pasta)}")


if __name__ == "__main__":
    main()
