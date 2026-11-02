"""
E se o mundo mudar depois da coleta? Avalia as políticas já treinadas num CartPole com vento. Não treina nada e não
usa os datasets: só carrega os modelos.pt de cada seed.

  vento constante   força extra de w newtons no carrinho, sempre para a direita
  rajadas           força extra sorteada a cada passo, normal com média 0 e desvio σ

A ação continua sendo ±10 N, então a força aplicada é exatamente ±10 + w. Todo agente recebe as mesmas seeds de
episódio (10000 + k) e, nas rajadas, a mesma sequência de vento. Com vento 0 o resultado tem de ser idêntico ao
"retornos" salvo no .pt: o script confere isso e avisa se não for.

Nenhum destes métodos trata mudança de dinâmica, e a heurística que gerou os dados também sofre com o vento: ela é
a referência de cada nível, não um teto a ser batido por construção.

Uso (da raiz do projeto):
  python Código/Extras/teste_vento.py --workers 8
  python Código/Extras/teste_vento.py --so-graficos       # refaz as figuras a partir do vento.csv
"""
import argparse, csv, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import gymnasium as gym

from comum_extras import C, IQL, COR, MARCA, METODOS, ROTULO, TINTA, TINTA2, br, eixo_limpo, ler_runs, pega, rotular_fim, salvar_csv

CONSTANTE, RAJADA = [0, 2, 3, 4, 5], [5, 10, 15]
DATASETS = ["expert", "medio", "ruim", "misto"]


class Heuristica:
    def reset(self, obs):
        pass

    def agir(self, obs, primeiro=False):
        return C.acao_especialista(obs)


def episodios(agente, modo, nivel, n, seed=10_000):
    """n episódios com vento. Retorna os retornos e como cada um acabou (fim, poste, trilho)."""
    env = gym.make(C.ENV_ID); u = env.unwrapped; base = u.force_mag; rets, fins = [], []
    for k in range(n):
        obs, _ = env.reset(seed=seed + k); sorteio = np.random.default_rng(seed + k); agente.reset(obs)
        fim, R, prim, term = False, 0.0, True, False
        while not fim:
            a = agente.agir(obs, primeiro=prim); prim = False
            w = nivel if modo == "constante" else sorteio.normal(0, nivel)
            u.force_mag = base + w if a == 1 else base - w
            obs, r, term, trunc, _ = env.step(a); R += r; fim = term or trunc
        rets.append(R); fins.append(("trilho" if abs(obs[0]) > u.x_threshold else "poste") if term else "fim")
    env.close()
    return rets, fins


def avaliar_modelo(job):
    info, algo, pesos, n, niveis = job
    agente = Heuristica() if algo == "heuristica" else IQL.Agente.montar(algo, pesos)
    linhas, eps = [], []
    for modo, nivel in niveis:
        rets, fins = episodios(agente, modo, nivel, n)
        linhas.append(dict(**info, modo=modo, nivel=nivel, retorno=float(np.mean(rets)), n_episodios=n,
                           frac_ate_o_fim=fins.count("fim") / n, frac_poste_caiu=fins.count("poste") / n, frac_saiu_do_trilho=fins.count("trilho") / n))
        eps.append(dict(**info, modo=modo, nivel=nivel, retornos=rets))
    return linhas, eps


def figura(linhas, modo, titulo, xlabel, arq, datasets, mostrar=False):
    plt = C.estilo_graficos(mostrar)
    sel = [l for l in linhas if l["modo"] == modo or (modo == "rajada" and l["modo"] == "constante" and float(l["nivel"]) == 0)]
    niveis = sorted({float(l["nivel"]) for l in sel})
    ref = [next(float(l["retorno"]) for l in sel if l["metodo"] == "heuristica" and float(l["nivel"]) == v) for v in niveis]
    n_seeds = len({l["seed"] for l in sel if l["metodo"] != "heuristica"})
    fig, axs = plt.subplots(1, len(datasets), figsize=(3.05 * len(datasets) + 0.7, 3.9), sharey=True, squeeze=False); axs = axs[0]
    for ax, ds in zip(axs, datasets):
        ax.plot(niveis, ref, color=TINTA, lw=1.4, ls=(0, (4, 3)), label="heurística (referência)", zorder=2)
        fim = {}
        for m in METODOS:
            vs = [[float(l["retorno"]) for l in sel if l["dataset"] == ds and l["metodo"] == m and float(l["nivel"]) == v] for v in niveis]
            ok = [j for j, v in enumerate(vs) if v]
            if not ok:
                continue
            x = [niveis[j] for j in ok]; med = [float(np.mean(vs[j])) for j in ok]
            ax.fill_between(x, [min(vs[j]) for j in ok], [max(vs[j]) for j in ok], color=COR[m], alpha=0.12, lw=0)
            ax.plot(x, med, "-" + MARCA[m], color=COR[m], lw=2, ms=6, mec="white", mew=1, label=ROTULO[m], zorder=3)
            fim[m] = med[-1]
        ax.set_title(f"treinado no {ds}", loc="left", fontsize=11); ax.set_xticks(niveis, [br(v) for v in niveis]); ax.set_xlabel(xlabel)
        eixo_limpo(ax); ax.set_ylim(0, 530); ax.set_xlim(niveis[0] - 0.2, niveis[-1] + 0.2)
        if ds == datasets[-1] and fim:
            rotular_fim(ax, niveis[-1], fim, {m: m for m in fim}, folga=34)
    axs[0].set_ylabel("retorno (máx. 500)")
    H, L = axs[0].get_legend_handles_labels()
    fig.legend(H, L, loc="upper right", ncol=len(L), frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(titulo, x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.885, f"Linha: média de {n_seeds} seed{'s' if n_seeds > 1 else ''}. Faixa: da pior à melhor seed. Nenhum modelo foi retreinado.", fontsize=9, color=TINTA2)
    fig.tight_layout(rect=(0, 0, 0.985, 0.88)); fig.savefig(arq, dpi=150)
    if mostrar:
        plt.show()
    plt.close(fig)


def figuras(linhas, pasta, datasets, mostrar=False):
    figura(linhas, "constante", "Vento constante", "força do vento w (N)", os.path.join(pasta, "1_vento_constante.png"), datasets, mostrar)
    figura(linhas, "rajada", "Rajadas a cada passo", "desvio das rajadas σ (N)", os.path.join(pasta, "2_rajadas.png"), datasets, mostrar)


def main():
    C.utf8_no_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--constante", type=float, nargs="+", default=CONSTANTE, help="forças w do vento constante; inclua 0 para a conferência")
    p.add_argument("--rajada", type=float, nargs="+", default=RAJADA, help="desvios σ das rajadas")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--episodios", type=int, default=50)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--pasta", default="experimentos/vento")
    p.add_argument("--raiz", default="experimentos", help="onde estão os runs com modelos.pt")
    p.add_argument("--so-graficos", action="store_true")
    p.add_argument("--mostrar", action="store_true")
    a = p.parse_args()

    arq_csv = os.path.join(a.pasta, "vento.csv")
    if a.so_graficos:
        with open(arq_csv, encoding="utf-8") as f:
            return figuras(list(csv.DictReader(f)), a.pasta, a.datasets, a.mostrar)
    if os.path.exists(arq_csv):
        sys.exit(f"{arq_csv} já existe e não será sobrescrito: escolha outra --pasta (ou use --so-graficos).")
    os.makedirs(a.pasta, exist_ok=True)
    niveis = [("constante", w) for w in a.constante] + [("rajada", s) for s in a.rajada]
    runs = ler_runs(a.raiz)
    if not runs:
        sys.exit(f"nenhum modelos.pt encontrado em {a.raiz}/")
    jobs = [(dict(run="", seed="", dataset="", metodo="heuristica"), "heuristica", None, a.episodios, niveis)]
    salvos = {}
    for run, seed, _, res in runs:
        for ds in a.datasets:
            for m in METODOS:
                r = pega(res, ds, m)
                if r:
                    jobs.append((dict(run=run, seed=seed, dataset=ds, metodo=m), r["algo"], r["pesos"], a.episodios, niveis))
                    salvos[(run, ds, m)] = r["retornos"]
    print(f"{len(jobs)} agentes × {len(niveis)} níveis de vento × {a.episodios} episódios ({a.workers} processos)", flush=True)
    t0, linhas, eps = time.time(), [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (ls, es) in enumerate(ex.map(avaliar_modelo, jobs), 1):
            linhas += ls; eps += es
            print(f"[{n}/{len(jobs)}] {ls[0]['metodo']} {ls[0]['dataset']} seed {ls[0]['seed']} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    conferidos, diferentes = 0, []
    for e in eps:
        if e["metodo"] != "heuristica" and e["modo"] == "constante" and float(e["nivel"]) == 0:
            ref = salvos[(e["run"], e["dataset"], e["metodo"])][:a.episodios]; conferidos += 1
            if e["retornos"][:len(ref)] != ref:
                diferentes.append((e["run"], e["dataset"], e["metodo"]))
    if diferentes:
        print(f"ATENÇÃO: com vento 0, {len(diferentes)} de {conferidos} modelos não reproduzem os retornos salvos: {diferentes}")
    elif conferidos:
        print(f"conferência: com vento 0, os {conferidos} modelos reproduzem exatamente os retornos salvos")

    salvar_csv(arq_csv, linhas)
    with open(os.path.join(a.pasta, "vento_episodios.json"), "w", encoding="utf-8") as f:
        json.dump(eps, f, ensure_ascii=False)
    with open(os.path.join(a.pasta, "numeros.json"), "w", encoding="utf-8") as f:
        json.dump(dict(conferencia_vento_zero=dict(conferidos=conferidos, diferentes=diferentes), seeds=sorted(s for _, s, _, _ in runs),
                       episodios=a.episodios, constante=a.constante, rajada=a.rajada), f, indent=2, ensure_ascii=False)
    figuras(linhas, a.pasta, a.datasets, a.mostrar)
    print(f"resultados em {os.path.abspath(a.pasta)}")


if __name__ == "__main__":
    main()
