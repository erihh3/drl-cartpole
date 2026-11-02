"""
Peças usadas só pelos scripts desta pasta (varredura_especialista.py, teste_vento.py, coletar_historico.py).

Os scripts principais (Código/BC, Código/CQL, Código/IQL) são apenas importados: nada aqui os altera.
"""
import csv, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.normpath(os.path.join(AQUI, "..", "IQL"))]
import torch
import cartpole_comum as C
import iql_cartpole as IQL

COR = IQL.COR                                                          # as mesmas cores das outras figuras
METODOS = ("BC", "CQL", "IQL")
MARCA = {"BC": "s", "CQL": "o", "IQL": "D"}                            # a forma também identifica o método
ROTULO = {"BC": "BC regularizado", "CQL": "CQL (α = 0,5)", "IQL": "IQL (τ = 0,7)"}
CHAVE = {"BC": "BC", "CQL": IQL.rotulo("CQL", {"alpha": 0.5}), "IQL": IQL.rotulo("IQL", {"tau": 0.7})}
HP = {"BC": IQL.BC_HP, "CQL": {"alpha": 0.5}, "IQL": {"tau": 0.7, "beta": 1.0}}
TINTA, TINTA2, GRADE = "#14213D", "#52514E", "#E6E6E3"
# pasta em experimentos/ -> seed. Duas pastas da mesma seed: vale a primeira que existir (iql_s0 é da mesma máquina das outras seeds).
RUNS = {"iql_s0": 0, "run_20261102": 0, "iql_s1": 1, "iql_s2": 2, "iql_s3": 3, "iql_s4": 4}


def ler_runs(raiz="experimentos", runs=None):
    """Os modelos.pt do iql_cartpole.py por seed. Run que ainda não existe é só pulado."""
    achados, vistas, todas = [], set(), set()
    for run, seed in (runs or RUNS).items():
        todas.add(seed); arq = os.path.join(raiz, run, "figuras_iql_cartpole", "modelos.pt")
        if seed not in vistas and os.path.exists(arq):
            achados.append((run, seed, arq, torch.load(arq, weights_only=False))); vistas.add(seed)
    for seed in sorted(todas - vistas):
        print(f"  (sem modelos.pt da seed {seed} em {raiz}/: seed pulada)")
    return achados


def pega(res, dataset, metodo):
    return next((r for r in res if r["dataset"] == dataset and r["rotulo"] == CHAVE[metodo]), None)


def salvar_csv(arq, linhas):
    os.makedirs(os.path.dirname(arq) or ".", exist_ok=True)
    colunas = list(dict.fromkeys(k for l in linhas for k in l))
    with open(arq, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, colunas); w.writeheader(); w.writerows(linhas)


def br(x, casas=0):
    return f"{x:.{casas}f}".replace(".", ",")


def eixo_limpo(ax):
    ax.grid(axis="y", color=GRADE, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#C9CED6")


def rotular_fim(ax, x, ys, nomes, folga=26):
    """Rótulo direto no fim de cada linha, afastando os que ficariam um em cima do outro."""
    pos, ordem = {}, sorted(ys, key=ys.get)
    for j, m in enumerate(ordem):
        pos[m] = ys[m] if j == 0 else max(ys[m], pos[ordem[j - 1]] + folga)
    for m in ordem:
        ax.annotate(nomes[m], (x, pos[m]), xytext=(9, 0), textcoords="offset points", va="center", fontsize=10,
                    color=TINTA, annotation_clip=False)
