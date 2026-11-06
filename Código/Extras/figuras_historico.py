"""
Refaz as figuras do Historico/HISTORIA.md só a partir dos CSVs de Historico/dados/: não treina, não avalia, não carrega modelos.

Uso (da raiz do projeto):
  python Código/Extras/coletar_historico.py     # atualiza os CSVs
  python Código/Extras/figuras_historico.py     # escreve em Historico/figuras/
"""
import ast, csv, os, shutil

import numpy as np

from comum_extras import C, COR, MARCA, METODOS, ROTULO, TINTA, TINTA2, br, eixo_limpo

DADOS, SAIDA = "Historico/dados", "Historico/figuras"
DS = ["expert", "medio", "ruim", "misto"]
CHAVE = {"BC": "BC", "CQL": "CQL (α=0.5)", "IQL": "IQL (τ=0.7)"}
TONS_TAU = {"0.5": "#D8AE45", "0.7": COR["IQL"], "0.9": "#5C3D00"}     # τ é ordinal: um só matiz, do claro ao escuro
FORMA_TAU = {"0.5": "o", "0.7": "D", "0.9": "s"}


def ler(nome):
    with open(os.path.join(DADOS, nome), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def valores(resumo, dataset, rotulo, avaliacao="padrao"):
    return ast.literal_eval(next(r for r in resumo if r["script"] == "iql" and r["dataset"] == dataset and r["rotulo"] == rotulo and r["avaliacao"] == avaliacao)["valores"])


def curva(curvas, marco, dataset, rotulo, seed):
    ls = [l for l in curvas if l["marco"] == marco and l["script"] == "iql" and l["dataset"] == dataset and l["rotulo"] == rotulo and int(l["seed"]) == seed and l["retorno"]]
    return np.array([int(l["passo"]) for l in ls]), np.array([float(l["retorno"]) for l in ls])


def cabecalho(fig, titulo, sub, H=None, L=None, y=0.985, y_sub=0.895):
    fig.suptitle(titulo, x=0.01, y=y, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, y_sub, sub, fontsize=9, color=TINTA2)
    if H:
        fig.legend(H, L, loc="upper right", ncol=len(L), frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.995))


def milhar(x):
    return f"{int(x):,}".replace(",", ".")


def fig_seeds(resumo, medias_dataset, plt):
    fig, ax = plt.subplots(figsize=(9.6, 5.0)); desloc = {"BC": -0.26, "CQL": 0.0, "IQL": 0.26}
    for i, ds in enumerate(DS):
        ax.plot([i - 0.42, i + 0.42], [medias_dataset[ds]] * 2, color=TINTA2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for m in METODOS:
            v = np.array(valores(resumo, ds, CHAVE[m])); x = i + desloc[m]
            ax.scatter(x + np.linspace(-0.055, 0.055, len(v)), v, s=38, color=COR[m], marker=MARCA[m], edgecolor="white", linewidth=0.8, zorder=3, label=ROTULO[m] if i == 0 else None)
            ax.plot([x - 0.1, x + 0.1], [v.mean()] * 2, color=TINTA, lw=1.8, solid_capstyle="round", zorder=4)
            ax.annotate(br(v.mean()), (x, v.max()), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color=TINTA)
    ax.plot([], [], color=TINTA2, lw=1.2, ls=(0, (4, 3)), label="média do dataset")
    ax.set_xticks(range(4), [f"treinado no {d}" for d in DS]); ax.set(ylabel="retorno (máx. 500)", ylim=(0, 545)); eixo_limpo(ax)
    H, L = ax.get_legend_handles_labels()
    cabecalho(fig, "Cinco seeds: o que se repete", "Cada ponto é uma seed; o traço e o número são a média. Seeds mudam a inicialização e os lotes; dados e episódios de avaliação são fixos.", H, L)
    fig.tight_layout(rect=(0, 0, 0.995, 0.885)); fig.savefig(os.path.join(SAIDA, "1_cinco_seeds.png"), dpi=150); plt.close(fig)


def fig_queda(curvas, plt):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, ds, titulo in ((axs[0], "misto", "misto, 5 seeds"), (axs[1], "ruim", "ruim, 5 seeds")):
        for m in ("CQL", "IQL"):
            ys = []
            for s in range(5):
                x, y = curva(curvas, "4_seeds", ds, CHAVE[m], s); ys.append(y)
                ax.plot(x, y, color=COR[m], lw=0.9, alpha=0.35, zorder=2)
            ax.plot(x, np.mean(ys, 0), "-" + MARCA[m], color=COR[m], lw=2, ms=6, mec="white", mew=1, zorder=3, label=ROTULO[m])
        ax.set_title(titulo, loc="left", fontsize=11); ax.set_xticks([0, 5000, 10000, 15000], ["0", "5.000", "10.000", "15.000"]); ax.set_xlabel("passos de treino"); eixo_limpo(ax)
    ax = axs[2]
    for m in ("CQL", "IQL"):
        x, y = curva(curvas, "5_longo_60k", "misto", CHAVE[m], 0)
        ax.plot(x, y, "-" + MARCA[m], color=COR[m], lw=2, ms=6, mec="white", mew=1, zorder=3)
    ax.axvline(15000, color=TINTA2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.annotate("15.000 passos", (15000, 30), xytext=(5, 0), textcoords="offset points", fontsize=9, color=TINTA2, va="center")
    ax.set_title("misto, seed 0, até 60.000 passos", loc="left", fontsize=11); ax.set_xticks([0, 20000, 40000, 60000], ["0", "20.000", "40.000", "60.000"]); ax.set_xlabel("passos de treino"); eixo_limpo(ax)
    axs[0].set(ylabel="retorno (máx. 500)", ylim=(0, 530))
    H, L = axs[0].get_legend_handles_labels()
    cabecalho(fig, "O CQL chega alto e depois cai; o IQL não", "Retorno durante o treino, com 10 episódios por ponto. Linha forte: média; linhas finas: cada seed.", H, L, y_sub=0.885)
    fig.tight_layout(rect=(0, 0, 0.995, 0.875)); fig.savefig(os.path.join(SAIDA, "2_queda_do_cql.png"), dpi=150); plt.close(fig)


def fig_tau(resumo, curvas, plt):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True, gridspec_kw=dict(width_ratios=[0.8, 1.35]))
    for j, t in enumerate(TONS_TAU):
        v = np.array(valores(resumo, "ruim", f"IQL (τ={t})"))
        a.scatter(j + np.linspace(-0.09, 0.09, 5), v, s=42, color=TONS_TAU[t], marker=FORMA_TAU[t], edgecolor="white", linewidth=0.8, zorder=3)
        a.plot([j - 0.2, j + 0.2], [v.mean()] * 2, color=TINTA, lw=1.8, solid_capstyle="round", zorder=4)
        a.annotate(br(v.mean()), (j, v.max()), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color=TINTA)
        x, y = curva(curvas, "5_longo_60k", "ruim", f"IQL (τ={t})", 0)
        b.plot(x, y, "-" + FORMA_TAU[t], color=TONS_TAU[t], lw=2, ms=6, mec="white", mew=1, zorder=3, label=f"τ = {t.replace('.', ',')}")
    a.set_xticks(range(3), [f"τ = {t.replace('.', ',')}" for t in TONS_TAU]); a.set_title("15.000 passos, 5 seeds", loc="left", fontsize=11); a.set_ylabel("retorno (máx. 500)"); a.set_xlim(-0.5, 2.5); eixo_limpo(a)
    b.set_title("até 60.000 passos, seed 0", loc="left", fontsize=11); b.set_xticks([0, 20000, 40000, 60000], ["0", "20.000", "40.000", "60.000"]); b.set_xlabel("passos de treino"); b.set_ylim(0, 330); eixo_limpo(b)
    H, L = b.get_legend_handles_labels()
    cabecalho(fig, "Com τ = 0,9, o retorno cai com mais treino", "IQL no ruim. Esquerda: cada ponto é uma seed, com 50 episódios. Direita: 10 episódios por ponto.", H, L, y_sub=0.885)
    fig.tight_layout(rect=(0, 0, 0.995, 0.875)); fig.savefig(os.path.join(SAIDA, "3_tau.png"), dpi=150); plt.close(fig)


def fig_formula(curvas, plt):
    def serie(rotulo, coluna):
        ls = [l for l in curvas if l["marco"] == "9_longo_expert" and l["dataset"] == "expert" and l["rotulo"] == rotulo and l[coluna]]
        return [int(l["passo"]) for l in ls], [float(l[coluna]) for l in ls]
    fig, ax = plt.subplots(figsize=(9.6, 5.0)); t = np.linspace(0, 60000, 200)
    ax.axhline(100, color=TINTA2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.annotate("máximo possível: 100", (0, 100), xytext=(4, 5), textcoords="offset points", fontsize=9, color=TINTA2)
    for m, marca in (("CQL", MARCA["CQL"]), ("IQL", MARCA["IQL"])):
        x, y = serie(CHAVE[m], "q_dados")
        ax.plot(x, y, "-" + marca, color=COR[m], lw=2, ms=7, mec="white", mew=1, markevery=(3 if m == "CQL" else 5, 5), zorder=3, label=f"{ROTULO[m]}: ação dos dados")
    ax.plot(t, 1 + 99 * (1 - np.exp(-t / 20000)), color=TINTA, lw=1.4, ls=(0, (4, 3)), zorder=4, label="fórmula, sem penalidade")
    x, y = serie(CHAVE["CQL"], "q_outra")
    ax.plot(x, y, color=COR["CQL"], lw=2, ls=(0, (1, 1.5)), zorder=3, label=f"{ROTULO['CQL']}: a outra ação")
    ax.axhline(0, color="#C9CED6", lw=1, zorder=1)
    ax.set_xticks([0, 15000, 30000, 45000, 60000], ["0", "15.000", "30.000", "45.000", "60.000"]); ax.set(xlabel="passos de treino", ylabel="Q", xlim=(0, 61000)); eixo_limpo(ax)
    H, L = ax.get_legend_handles_labels()
    fig.suptitle("No expert, Q da ação dos dados segue a fórmula; o CQL só derruba a outra ação", x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.905, "Seed 0, 60.000 passos. Nada foi ajustado à curva: ela só usa recompensa 1, γ = 0,99 e Polyak 0,005.", fontsize=9, color=TINTA2)
    fig.legend(H, L, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.005, 0.885))
    fig.tight_layout(rect=(0, 0, 0.995, 0.80)); fig.savefig(os.path.join(SAIDA, "5_q_e_formula.png"), dpi=150); plt.close(fig)


def fig_parada(sens, plt):
    """Treinar só até N passos recria o modelo daquele passo: dá para medir cada parada com 50 episódios."""
    N = (3000, 6000, 9000, 12000, 15000)
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    for ax, ds in zip(axs, ("ruim", "misto")):
        for m in ("CQL", "IQL"):
            por = [[float(l["retorno"]) for l in sens if l["bloco"] == f"parada_n{n}" and l["dataset"] == ds and l["rotulo"] == CHAVE[m]] for n in N]
            for n, v in zip(N, por):
                ax.scatter([n] * len(v), v, s=18, color=COR[m], alpha=0.35, lw=0, zorder=2)
            ax.plot(N, [np.mean(v) for v in por], "-" + MARCA[m], color=COR[m], lw=2, ms=6, mec="white", mew=1, zorder=3, label=ROTULO[m])
        ax.set_title(f"{ds}, 5 seeds", loc="left", fontsize=11); ax.set_xticks(N, ["3.000", "6.000", "9.000", "12.000", "15.000"]); ax.set_xlabel("passos de treino")
    axs[0].set(ylabel="retorno (máx. 500)", ylim=(0, 530))
    H, L = axs[0].get_legend_handles_labels()
    cabecalho(fig, "Os picos do CQL são reais", "Cada ponto é um modelo treinado até ali e avaliado em 50 episódios. Linha: média das seeds; pontos: cada seed. Em GPU.", H, L)
    fig.tight_layout(rect=(0, 0, 0.995, 0.875)); fig.savefig(os.path.join(SAIDA, "7_parada.png"), dpi=150); plt.close(fig)


def fig_alfa(sens, sens_c, resumo, curvas, plt):
    """Em cima, o retorno final por α; embaixo, o treino com o α original e o maior α testado."""
    ALFAS = ("0", "0.1", "0.5", "2", "5"); TONS = {"0.5": "#8CC7BD", "5": COR["CQL"]}            # α é ordinal: um só matiz
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 7.2), sharey=True)
    for j, ds in enumerate(("medio", "misto", "ruim")):
        ax = axs[0, j]; por = []
        for i, a in enumerate(ALFAS):
            rot = "DQN (α=0)" if a == "0" else f"CQL (α={a})"
            v = [float(l["retorno"]) for l in sens if l["bloco"] == f"alfa_{ds}" and l["rotulo"] == rot]; por.append(v)
            ax.scatter([i] * len(v), v, s=18, color=COR["CQL"], alpha=0.35, lw=0, zorder=2)
        ax.plot(range(5), [np.mean(v) for v in por], "-o", color=COR["CQL"], lw=2, ms=6, mec="white", mew=1, zorder=3, label="CQL, por α")
        iql = np.mean(valores(resumo, "medio", CHAVE["IQL"])) if ds == "medio" else np.mean([float(l["retorno"]) for l in sens if l["bloco"] == "parada_n15000" and l["dataset"] == ds and l["rotulo"] == CHAVE["IQL"]])
        ax.axhline(iql, color=COR["IQL"], lw=2, zorder=1, label=f"IQL (τ = 0,7, β = 1)")
        ax.set_title(f"{ds}: retorno final", loc="left", fontsize=11); ax.set_xticks(range(5), ["0", "0,1", "0,5", "2", "5"]); ax.set_xlabel("α")
        ax = axs[1, j]
        for a in ("0.5", "5"):
            ls = [l for l in sens_c if l["bloco"] == f"alfa_{ds}" and l["rotulo"] == f"CQL (α={a})"]; x = sorted({int(l["passo"]) for l in ls})
            ax.plot(x, [np.mean([float(l["retorno"]) for l in ls if int(l["passo"]) == p]) for p in x], "-o", color=TONS[a], lw=2, ms=5, mec="white", mew=1, zorder=3, label=f"CQL (α = {a.replace('.', ',')})")
        if ds == "medio":
            x, ys = None, []
            for s in range(5):
                x, y = curva(curvas, "4_seeds", "medio", CHAVE["IQL"], s); ys.append(y)
            my = np.mean(ys, 0)
        else:
            ls = [l for l in sens_c if l["bloco"] == "parada_n15000" and l["dataset"] == ds and l["rotulo"] == CHAVE["IQL"]]; x = sorted({int(l["passo"]) for l in ls})
            my = [np.mean([float(l["retorno"]) for l in ls if int(l["passo"]) == p]) for p in x]
        ax.plot(x, my, "-D", color=COR["IQL"], lw=2, ms=5, mec="white", mew=1, zorder=2, label="IQL (τ = 0,7)")
        ax.set_title(f"{ds}: durante o treino", loc="left", fontsize=11); ax.set_xticks([0, 5000, 10000, 15000], ["0", "5.000", "10.000", "15.000"]); ax.set_xlabel("passos de treino")
    for i in range(2):
        axs[i, 0].set(ylabel="retorno (máx. 500)", ylim=(0, 530)); axs[i, 2].legend(fontsize=9, loc="lower right" if i == 0 else "upper right")
    cabecalho(fig, "Com α maior, o CQL alcança o IQL no medio e no misto, chega perto no ruim e para de cair", "Cinco seeds, em GPU; no medio, a referência do IQL vem da rodada em CPU. Durante o treino, 10 episódios por ponto.", y=0.99, y_sub=0.945)
    fig.tight_layout(rect=(0, 0, 0.995, 0.935)); fig.savefig(os.path.join(SAIDA, "8_sensibilidade_alfa.png"), dpi=150); plt.close(fig)


def fig_alfa_50(sens, resumo, plt):
    """α de 0 a 50: de um lado o Q-learning sem penalidade, do outro o CQL que vira BC."""
    ALFAS = ("0", "0.1", "0.5", "2", "5", "10", "20", "50")
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    for ax, ds in zip(axs, ("medio", "misto", "ruim")):
        por = []
        for i, a in enumerate(ALFAS):
            rot = "DQN (α=0)" if a == "0" else f"CQL (α={a})"; bl = f"alfa_alto_{ds}" if float(a) > 5 else f"alfa_{ds}"
            v = [float(l["retorno"]) for l in sens if l["bloco"] == bl and l["rotulo"] == rot]; por.append(v)
            ax.scatter([i] * len(v), v, s=18, color=COR["CQL"], alpha=0.35, lw=0, zorder=2)
        ax.plot(range(8), [np.mean(v) for v in por], "-o", color=COR["CQL"], lw=2, ms=6, mec="white", mew=1, zorder=3, label="CQL, por α")
        for m, cor in (("IQL", COR["IQL"]), ("BC", COR["BC"])):
            ref = np.mean(valores(resumo, "medio", CHAVE[m])) if ds == "medio" else np.mean([float(l["retorno"]) for l in sens if l["bloco"] == "parada_n15000" and l["dataset"] == ds and l["rotulo"] == CHAVE[m]])
            ax.axhline(ref, color=cor, lw=2, zorder=1, label=ROTULO[m])
        ax.set_title(f"{ds}, 15.000 passos", loc="left", fontsize=11); ax.set_xticks(range(8), [a.replace(".", ",") for a in ALFAS]); ax.set_xlabel("α (fora de escala)")
    axs[0].set(ylabel="retorno (máx. 500)", ylim=(0, 530))
    H, L = axs[0].get_legend_handles_labels()
    cabecalho(fig, "O α vai do Q-learning ao BC", "Cinco seeds, em GPU; no medio, as referências vêm da rodada em CPU. Com α = 50 no ruim, o CQL faz o mesmo retorno do BC.", H, L)
    fig.tight_layout(rect=(0, 0, 0.995, 0.865)); fig.savefig(os.path.join(SAIDA, "9_alfa_ate_50.png"), dpi=150); plt.close(fig)


def fig_longo(sens_c, plt):
    """O α forte atrasa a queda, não a evita."""
    TONS = {"0.5": "#A9D6CE", "2": "#5FAA9C", "5": COR["CQL"]}            # α é ordinal: um só matiz
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    for ax, ds in zip(axs, ("misto", "ruim")):
        for a in TONS:
            ls = [l for l in sens_c if l["bloco"] == f"longo_{ds}" and l["rotulo"] == f"CQL (α={a})"]; x = sorted({int(l["passo"]) for l in ls})
            ax.plot(x, [np.mean([float(l["retorno"]) for l in ls if int(l["passo"]) == p]) for p in x], "-o", color=TONS[a], lw=2, ms=5, mec="white", mew=1, zorder=3, label=f"CQL (α = {a.replace('.', ',')})")
        ls = [l for l in sens_c if l["bloco"] == "iql_longo" and l["dataset"] == ds and l["rotulo"] == CHAVE["IQL"]]; x = sorted({int(l["passo"]) for l in ls})
        ax.plot(x, [np.mean([float(l["retorno"]) for l in ls if int(l["passo"]) == p]) for p in x], "-D", color=COR["IQL"], lw=2, ms=5, mec="white", mew=1, zorder=4, label=ROTULO["IQL"])
        ax.axvline(15000, color=TINTA2, lw=1, zorder=1); ax.annotate("15.000 passos", (15000, 30), xytext=(5, 0), textcoords="offset points", fontsize=9, color=TINTA2, va="center")
        ax.set_title(f"{ds}, 5 seeds", loc="left", fontsize=11); ax.set_xticks([0, 20000, 40000, 60000], ["0", "20.000", "40.000", "60.000"]); ax.set_xlabel("passos de treino")
    axs[0].set(ylabel="retorno (máx. 500)", ylim=(0, 530))
    H, L = axs[0].get_legend_handles_labels()
    cabecalho(fig, "Um α mais forte só atrasa a queda do CQL", "Média de cinco seeds, em GPU, com 10 episódios por ponto.", H, L)
    fig.tight_layout(rect=(0, 0, 0.995, 0.865)); fig.savefig(os.path.join(SAIDA, "10_longo_alfa.png"), dpi=150); plt.close(fig)


def main():
    os.makedirs(SAIDA, exist_ok=True); plt = C.estilo_graficos(False)
    resumo, curvas = ler("resumo_seeds.csv"), ler("curvas.csv")
    medias = {l["dataset"]: float(l["valor"]) for l in ler("evolucao.csv") if l["marco"] == "4_seeds" and l["metodo"] == "dados" and l["metrica"] == "retorno" and l["seed"] == "0"}
    fig_seeds(resumo, medias, plt); fig_queda(curvas, plt); fig_tau(resumo, curvas, plt); fig_formula(curvas, plt)
    if os.path.exists(os.path.join(DADOS, "sensibilidade.csv")):
        sens = ler("sensibilidade.csv"); fig_parada(sens, plt); sens_c = ler("sensibilidade_curvas.csv"); fig_alfa(sens, sens_c, resumo, curvas, plt)
        if any(l["bloco"] == "iql_longo" for l in sens):
            fig_alfa_50(sens, resumo, plt); fig_longo(sens_c, plt)
    for origem, nome in (("experimentos/varredura_especialista/1_quanto_especialista.png", "4_quanto_especialista.png"),
                         ("experimentos/bc_dados/figuras_bc_cartpole/6_dados.png", "6_dados_bc.png")):
        if os.path.exists(origem):
            shutil.copyfile(origem, os.path.join(SAIDA, nome))
    print("figuras em", SAIDA, sorted(os.listdir(SAIDA)))


if __name__ == "__main__":
    main()
