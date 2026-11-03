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


def main():
    os.makedirs(SAIDA, exist_ok=True); plt = C.estilo_graficos(False)
    resumo, curvas = ler("resumo_seeds.csv"), ler("curvas.csv")
    medias = {l["dataset"]: float(l["valor"]) for l in ler("evolucao.csv") if l["marco"] == "4_seeds" and l["metodo"] == "dados" and l["metrica"] == "retorno" and l["seed"] == "0"}
    fig_seeds(resumo, medias, plt); fig_queda(curvas, plt); fig_tau(resumo, curvas, plt)
    origem = "experimentos/varredura_especialista/1_quanto_especialista.png"
    if os.path.exists(origem):
        shutil.copyfile(origem, os.path.join(SAIDA, "4_quanto_especialista.png"))
    print("figuras em", SAIDA, sorted(os.listdir(SAIDA)))


if __name__ == "__main__":
    main()
