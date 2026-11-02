"""
CQL (Conservative Q-Learning) offline no CartPole, com datasets gerados pelo Minari.

Treina, no mesmo dataset e com as mesmas sementes:
  - DQN offline  (CQL com alpha = 0: só a perda TD)
  - CQL          (alpha > 0: perda TD + alpha * [logsumexp_a Q(s,a) - Q(s, a_dados)])
  - varredura de alpha

No dataset expert o especialista é determinístico: em cada estado só uma das 2 ações aparece, e o
Q da outra nunca é corrigido por dado algum. É o caso clássico de erro de extrapolação.

Figuras em ./figuras_cql_cartpole/:
  1_superestimacao.png  Q da ação do dataset × da outra ação; Q previsto × retorno real
  2_mecanismo.png       Q(s,·) num estado e decomposição da perda
  3_gap.png             gap entre as duas ações e concordância com o dataset
  4_alfa.png            varredura do alpha
  5_resultado.png       ângulo do poste num episódio e mapas de política (θ × ω)

Uso:
  pip install "minari[create]" gymnasium torch matplotlib
  python cql_cartpole.py                      # dataset expert
  python cql_cartpole.py --dataset ruim       # também: medio, misto
  python cql_cartpole.py --so-graficos
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch, torch.nn.functional as F

import cartpole_comum as C

torch.set_num_threads(1)
COR = {"DQN": "#B04A16", "CQL": "#1F7A6D", "dados": "#8A93A3"}


def treinar(args):
    """alpha = 0 -> DQN offline (Double DQN); alpha > 0 -> CQL(H) discreto."""
    nome_ds, alpha, passos, seed, n_aval = args
    D = C.carregar(nome_ds)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    T = {k: torch.as_tensor(D[k]) for k in ("s", "a", "r", "s2", "d")}
    mon = np.random.default_rng(123).choice(len(D["a"]), 2000, replace=False)
    s_mon, a_mon = T["s"][mon], T["a"][mon]

    q, q_alvo = C.mlp(), C.mlp(); q_alvo.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), 3e-4)
    log, acum = [], {"td": [], "pen": []}
    log_cada, aval_cada = max(1, passos // 40), max(1, passos // 10)
    t0 = time.time()
    for t in range(1, passos + 1):
        i = torch.as_tensor(rng.integers(len(D["a"]), size=256))
        s, a, r, s2, d = (T[k][i] for k in ("s", "a", "r", "s2", "d"))
        with torch.no_grad():
            a2 = q(s2).argmax(1, keepdim=True)                                       # Double DQN
            y = r + C.GAMMA * (1 - d) * q_alvo(s2).gather(1, a2).squeeze(1)
        q_s = q(s); q_sa = q_s.gather(1, a[:, None]).squeeze(1)
        perda_td = F.mse_loss(q_sa, y)
        penal = (torch.logsumexp(q_s, 1) - q_sa).mean()                              # termo conservador
        perda = perda_td + alpha * penal
        opt.zero_grad(); perda.backward(); opt.step()
        with torch.no_grad():
            for p, pa in zip(q.parameters(), q_alvo.parameters()):
                pa.mul_(0.995).add_(0.005 * p)
        acum["td"].append(perda_td.item()); acum["pen"].append(penal.item())
        if t % log_cada == 0 or t % aval_cada == 0:
            with torch.no_grad():
                qm = q(s_mon)
                linha = dict(passo=t, perda_td=float(np.mean(acum["td"])), penalidade=float(np.mean(acum["pen"])),
                             q_dados=qm.gather(1, a_mon[:, None]).mean().item(),
                             q_outra=qm.gather(1, (1 - a_mon)[:, None]).mean().item(),
                             gap=(qm[:, 0] - qm[:, 1]).abs().median().item(),
                             concordancia=(qm.argmax(1) == a_mon).float().mean().item())
            acum = {"td": [], "pen": []}
            if t % aval_cada == 0:
                linha["retorno"] = C.avaliar(C.Agente(q), n=10, seed=50_000)[0]
            log.append(linha)
    ret, rets, acoes = C.avaliar(C.Agente(q), n=n_aval)
    print(f"  alpha={alpha:<5} retorno={ret:6.1f}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(alpha=alpha, log=log, retorno=ret, retornos=rets.tolist(), acoes=acoes.tolist(),
                pesos={k: v.cpu() for k, v in q.state_dict().items()})


def montar(pesos):
    q = C.mlp(); q.load_state_dict(pesos); return q.eval()


def gerar_figuras(res, D, nome_ds, pasta, mostrar=False):
    plt = C.estilo_graficos(mostrar)
    os.makedirs(pasta, exist_ok=True)
    alfas = sorted(res)
    a_cql = min([a for a in alfas if a > 0], key=lambda a: (-res[a]["retorno"], a))
    dqn, cql = res[0.0], res[a_cql]
    q_dqn, q_cql = montar(dqn["pesos"]), montar(cql["pesos"])
    rot_cql = f"CQL (α={a_cql:g})"
    numeros = {"dataset": nome_ds, "alpha_cql": a_cql}
    idx = np.random.default_rng(7).choice(len(D["a"]), 5000, replace=False)
    S, A, G = torch.as_tensor(D["s"][idx]), torch.as_tensor(D["a"][idx]), D["G"][idx]
    with torch.no_grad():
        QD, QC = q_dqn(S), q_cql(S)

    def curva(ax, chave, nome, r, **kw):
        L = [l for l in r["log"] if chave in l]
        ax.plot([l["passo"] for l in L], [l[chave] for l in L], label=nome, **kw)

    def salvar(fig, nome, titulo):
        fig.suptitle(titulo, fontsize=13); fig.tight_layout(); fig.savefig(f"{pasta}/{nome}", dpi=120)

    # 1. superestimação
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for nome, r, c in [("DQN offline", dqn, COR["DQN"]), (rot_cql, cql, COR["CQL"])]:
        curva(ax[0], "q_outra", f"{nome}: a outra ação", r, color=c, lw=2.2)
        curva(ax[0], "q_dados", f"{nome}: ação do dataset", r, color=c, lw=1.6, ls="--")
    ax[0].axhline(float(G.mean()), color="k", lw=1, ls=":", label="retorno real médio (G)")
    ax[0].set_yscale("symlog", linthresh=10)
    ax[0].set(title="Q ao longo do treino (escala log)", xlabel="passo de treino", ylabel="Q médio"); ax[0].legend(fontsize=9)
    sel = np.random.default_rng(0).choice(len(G), 1500, replace=False)
    for nome, Q, c in [("DQN offline", QD, COR["DQN"]), (rot_cql, QC, COR["CQL"])]:
        qmax = Q.max(1).values.numpy(); vies = float(np.mean(qmax - G)); numeros[f"vies_{nome[:3]}"] = vies
        ax[1].scatter(G[sel], qmax[sel], s=7, alpha=0.35, color=c, label=f"{nome}: viés médio {vies:+.1f}")
    lim = [0, max(G.max(), 1)]
    ax[1].plot(lim, lim, "k--", lw=1, label="Q = retorno real")
    ax[1].set_yscale("symlog", linthresh=10)
    ax[1].set(title="max Q(s,·) previsto × retorno real G (escala log)", xlabel="retorno descontado observado G", ylabel="max Q(s,·)")
    ax[1].legend(fontsize=9, markerscale=3)
    fd, fc = dqn["log"][-1], cql["log"][-1]
    salvar(fig, "1_superestimacao.png", f"1. Valores de Q — DQN: dataset {fd['q_dados']:.0f}, outra ação {fd['q_outra']:.0f}"
                                         f" · CQL: {fc['q_dados']:.0f} e {fc['q_outra']:.0f}")

    # 2. mecanismo
    cand = np.where((QD.argmax(1) != A).numpy() & (QC.argmax(1) == A).numpy())[0]
    k = int(cand[0]) if len(cand) else 0
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for j, (nome, Q, c) in enumerate([("DQN offline", QD, COR["DQN"]), (rot_cql, QC, COR["CQL"])]):
        qv = Q[k].numpy(); lse = float(torch.logsumexp(Q[k], 0)); a_d = int(A[k])
        b = ax[j].bar(C.NOMES_ACOES, qv, color=[c if i == a_d else "#C9CED6" for i in range(2)], width=0.55)
        ax[j].bar_label(b, fmt="%.1f", padding=2)
        ax[j].axhline(lse, color="k", ls="--", lw=1.2)
        ax[j].text(0.5, lse * 1.04, f"logsumexp = {lse:.1f}", ha="center", va="bottom", fontsize=9)
        ax[j].set_ylim(0, max(lse, qv.max()) * 1.2)
        ax[j].set(title=f"{nome}\npenalidade = {lse - qv[a_d]:.3f}", ylabel="Q(s, a)")
        ax[j].set_xlabel(f"barra colorida = ação do dataset ({C.NOMES_ACOES[a_d]})", fontsize=9)
        numeros[f"estado_Q_{nome[:3]}"] = qv.round(2).tolist()
    curva(ax[2], "perda_td", f"{rot_cql}: perda TD", cql, color=COR["CQL"], lw=2)
    curva(ax[2], "penalidade", f"{rot_cql}: penalidade", cql, color=COR["CQL"], lw=2, ls="--")
    curva(ax[2], "perda_td", "DQN: perda TD", dqn, color=COR["DQN"], lw=2)
    curva(ax[2], "penalidade", "DQN: penalidade (medida, não otimizada)", dqn, color=COR["DQN"], lw=2, ls="--")
    ax[2].set_yscale("log"); ax[2].set(title="Decomposição da perda", xlabel="passo de treino"); ax[2].legend(fontsize=9)
    salvar(fig, "2_mecanismo.png", "2. O mecanismo: a penalidade empurra para baixo o Q da ação que não está no dataset")

    # 3. gap e concordância
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = np.geomspace(1e-3, 200, 60)
    for nome, Q, c in [("DQN offline", QD, COR["DQN"]), (rot_cql, QC, COR["CQL"])]:
        gap = (Q[:, 0] - Q[:, 1]).abs().numpy().clip(1e-3); conc = (Q.argmax(1) == A).float().mean().item()
        ax[0].hist(gap, bins=bins, alpha=0.6, color=c, label=f"{nome}: mediana {np.median(gap):.3g} · concorda {conc:.0%}")
        numeros[f"gap_{nome[:3]}"] = float(np.median(gap)); numeros[f"concordancia_{nome[:3]}"] = conc
    ax[0].set_xscale("log"); ax[0].set(title="|Q(esquerda) − Q(direita)|", xlabel="gap (escala log)", ylabel="nº de estados")
    ax[0].legend(fontsize=9)
    for nome, r, c in [("DQN offline", dqn, COR["DQN"]), (rot_cql, cql, COR["CQL"])]:
        curva(ax[1], "concordancia", nome, r, color=c, lw=2)
    ax[1].set_ylim(0, 1); ax[1].set(title="Concordância com a ação do dataset", xlabel="passo de treino", ylabel="fração")
    ax[1].legend(fontsize=9)
    salvar(fig, "3_gap.png", "3. O efeito: o CQL separa a ação do dataset da outra")

    # 4. alfa
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    xs = np.arange(len(alfas)); rot = [f"{a:g}" for a in alfas]
    rets = [res[a]["retorno"] for a in alfas]
    b = ax[0].bar(xs, rets, color=[COR["DQN"] if a == 0 else COR["CQL"] for a in alfas])
    ax[0].bar_label(b, labels=[f"{x:.0f}" for x in rets], padding=2)
    ax[0].set_xticks(xs, rot); ax[0].set_ylim(0, 540)
    ax[0].set(title="Retorno final (máximo 500)", xlabel="α (α = 0 é o DQN offline)", ylabel="retorno médio")
    fim = {a: res[a]["log"][-1] for a in alfas}
    ax[1].plot(xs, [fim[a]["q_dados"] for a in alfas], "o-", color="#14213D", label="Q da ação do dataset")
    ax[1].plot(xs, [fim[a]["q_outra"] for a in alfas], "s--", color=COR["DQN"], label="Q da outra ação")
    ax[1].axhline(float(G.mean()), color="k", ls=":", lw=1, label="retorno real médio")
    ax[1].set_yscale("symlog", linthresh=10)
    ax[1].set_xticks(xs, rot); ax[1].set(title="Q ao final do treino (escala log)", xlabel="α", ylabel="Q médio"); ax[1].legend(fontsize=9)
    cmap = plt.get_cmap("viridis")
    for j, a in enumerate(alfas):
        curva(ax[2], "retorno", f"α = {a:g}", res[a], color=COR["DQN"] if a == 0 else cmap(j / len(alfas)), lw=2,
              marker="o", ms=3)
    ax[2].set_ylim(0, 540); ax[2].set(title="Retorno durante o treino (10 episódios)", xlabel="passo de treino")
    ax[2].legend(fontsize=9)
    numeros["retorno_por_alfa"] = {f"{a:g}": res[a]["retorno"] for a in alfas}
    salvar(fig, "4_alfa.png", "4. O ajuste: α controla o quanto o Q da ação fora dos dados é empurrado para baixo")

    # 5. resultado
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    for nome, q, c in [("DQN offline", q_dqn, COR["DQN"]), (rot_cql, q_cql, COR["CQL"])]:
        S_ep = C.rodar_episodio(C.Agente(q), 10_000)
        ax[0].plot(S_ep[:, 2], color=c, lw=2, label=f"{nome}: {len(S_ep) - 1} passos")
    ax[0].axhline(0.2095, color="#999", ls=":", lw=1); ax[0].axhline(-0.2095, color="#999", ls=":", lw=1, label="limite (queda)")
    ax[0].set(title="Ângulo do poste num episódio (seed 10000)", xlabel="passo", ylabel="ângulo θ (rad)")
    ax[0].legend(fontsize=9)
    C.desenhar_mapa(ax[1], q_dqn, "DQN offline: ação escolhida", D)
    C.desenhar_mapa(ax[2], q_cql, f"{rot_cql}: ação escolhida", D)
    salvar(fig, "5_resultado.png", f"5. O resultado no ambiente (dataset {nome_ds})")

    with open(f"{pasta}/numeros.json", "w", encoding="utf-8") as fh:
        json.dump(numeros, fh, indent=2, ensure_ascii=False)
    if mostrar:
        plt.show()
    plt.close("all")
    return numeros


def main():
    C.utf8_no_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="expert", choices=["expert", "medio", "ruim", "misto"])
    p.add_argument("--passos", type=int, default=15_000)
    p.add_argument("--alfas", type=float, nargs="+", default=[0, 0.1, 0.5, 2, 5])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodios", type=int, default=50, help="episódios na avaliação final")
    p.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    p.add_argument("--pasta", default="figuras_cql_cartpole")
    p.add_argument("--mostrar", action="store_true")
    p.add_argument("--so-graficos", action="store_true")
    a = p.parse_args()
    if 0 not in a.alfas:
        a.alfas = [0.0] + a.alfas

    C.garantir_datasets()
    D = C.carregar(a.dataset); print(C.resumo_dataset(a.dataset, D))
    os.makedirs(a.pasta, exist_ok=True)
    arq = os.path.join(a.pasta, f"modelos_{a.dataset}.pt")
    if a.so_graficos:
        res = torch.load(arq, weights_only=False)
    else:
        jobs = [(a.dataset, float(al), a.passos, a.seed, a.episodios) for al in a.alfas]
        print(f"treinando {len(jobs)} modelos ({a.passos} passos cada)...")
        with ProcessPoolExecutor(a.workers) as ex:
            res = {r["alpha"]: r for r in ex.map(treinar, jobs)}
        torch.save(res, arq)
    print("gerando gráficos...")
    numeros = gerar_figuras(res, D, a.dataset, a.pasta, a.mostrar)
    print(json.dumps(numeros, indent=2, ensure_ascii=False))
    print(f"figuras salvas em {os.path.abspath(a.pasta)}")


if __name__ == "__main__":
    main()
