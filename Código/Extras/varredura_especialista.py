"""
Quanto especialista basta? O misto tem 2,4% de episódios do especialista, mas 43,8% das transições (cada episódio
do especialista dura 500 passos; os do ruim, cerca de 16). Esta varredura treina BC, CQL e IQL em todo o ruim mais
k episódios do especialista, para ver a partir de quanto cada método se recupera.

  k = 1, 2, 5, 10, 25   treinados aqui, com os mesmos hiperparâmetros do iql_cartpole.py
  k = 0 (ruim), 50 (misto)  lidos dos modelos.pt que já existem em experimentos/, se houver (nada é retreinado)

Quais k episódios: sorteio que depende só da seed e é aninhado (com a mesma seed, k = 5 contém k = 2). Portanto a
seed aqui muda também os dados, e k = 50 é exatamente o misto.

Cada modelo é salvo assim que termina. Rodar de novo com a mesma --pasta só treina o que falta: nada é sobrescrito.

Uso (da raiz do projeto):
  python Código/Extras/varredura_especialista.py --workers 8
  python Código/Extras/varredura_especialista.py --plano        # só lista o que seria treinado
  python Código/Extras/varredura_especialista.py --so-graficos  # refaz CSV e figura a partir do que já está salvo
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from comum_extras import C, IQL, COR, MARCA, METODOS, ROTULO, HP, TINTA2, br, eixo_limpo, ler_runs, pega, rotular_fim, salvar_csv

ORIGINAL = C.carregar                                                  # treinar_nivel troca C.carregar nos processos
_base = {}


def base(nome):
    if nome not in _base:
        _base[nome] = ORIGINAL(nome)
    return _base[nome]


def montar(k, seed):
    """Todo o ruim + k episódios do expert, na mesma ordem e com os mesmos campos do C.carregar("misto")."""
    a, b = base("expert"), base("ruim")
    escolhidos = np.random.default_rng(1_000 + seed).permutation(a["ep"].max() + 1)[:k]
    m = np.isin(a["ep"], escolhidos); n_a = int(m.sum())
    novo_ep = np.unique(a["ep"][m], return_inverse=True)[1] if n_a else np.zeros(0, a["ep"].dtype)
    inicio = np.r_[True, novo_ep[1:] != novo_ep[:-1]] if n_a else np.zeros(0, bool)
    A = {c: v[m] for c, v in a.items()}
    A["ep"] = novo_ep.astype(a["ep"].dtype)
    A["ini"] = (np.flatnonzero(inicio)[np.cumsum(inicio) - 1] if n_a else np.zeros(0, a["ini"].dtype)).astype(a["ini"].dtype)
    B = dict(b, ep=b["ep"] + k, ini=b["ini"] + n_a)
    return {c: np.concatenate([A[c], B[c]]) for c in a}


def descrever(D, k):
    n = len(D["a"]); n_ep = len(np.unique(D["ep"]))
    ret = float(D["ret_ep"][D["ini"] == np.arange(n)].mean())
    n_esp = int(k * 500)                                               # todo episódio do expert chega aos 500 passos
    return dict(ep_especialista=k, n_episodios_dataset=n_ep, n_transicoes=n, frac_episodios=k / n_ep,
                frac_transicoes=n_esp / n, retorno_dataset=ret)


def treinar_nivel(job):
    k, metodo, passos, seed, n_aval, dispositivo = job
    D = montar(k, seed)
    C.carregar = lambda nome: D                                        # IQL.treinar lê o dataset pelo nome
    r = IQL.treinar((f"k{k}", metodo, HP[metodo], passos, seed, n_aval), dispositivo)
    r.update(k=k, seed=seed, passos=passos, **descrever(D, k))
    return r


def arquivo(pasta, k, seed, metodo):
    return os.path.join(pasta, "modelos", f"k{k:02d}_s{seed}_{metodo}.pt")


# =============================================================================== saídas
def linha_de(r, k, seed, metodo, origem, info):
    a = np.array(r["retornos"]); p = np.array(r.get("retornos_perturbados", []))
    return dict(k=k, seed=seed, metodo=metodo, retorno=float(a.mean()), retorno_perturbado=float(p.mean()) if len(p) else "",
                n_episodios=len(a), frac_cairam_cedo=float((a < 100).mean()), origem=origem, **info)


def reunir(a):
    """Uma linha por (k, seed, método), do que já foi salvo aqui e dos extremos k = 0 e k = 50 dos runs existentes."""
    linhas, curvas = [], []
    def curva(r, k, seed, metodo):
        curvas.extend(dict(k=k, seed=seed, metodo=metodo, passo=l["passo"], retorno=l["retorno"]) for l in r["log"] if "retorno" in l)
    if os.path.isdir(os.path.join(a.pasta, "modelos")):
        for nome in sorted(os.listdir(os.path.join(a.pasta, "modelos"))):
            if nome.endswith(".pt"):
                r = torch.load(os.path.join(a.pasta, "modelos", nome), weights_only=False)
                info = {c: r[c] for c in ("ep_especialista", "n_episodios_dataset", "n_transicoes", "frac_episodios", "frac_transicoes", "retorno_dataset")}
                linhas.append(linha_de(r, r["k"], r["seed"], r["algo"], "varredura", info)); curva(r, r["k"], r["seed"], r["algo"])
    extremos = {0: "ruim", 50: "misto"}
    info = {k: descrever(montar(k, 0), k) for k in extremos}
    for run, seed, _, res in ler_runs(a.raiz):
        if seed in a.seeds:
            for k, ds in extremos.items():
                for m in METODOS:
                    r = pega(res, ds, m)
                    if r:
                        linhas.append(linha_de(r, k, seed, m, f"{run} ({ds})", info[k])); curva(r, k, seed, m)
    linhas.sort(key=lambda l: (l["k"], l["seed"], METODOS.index(l["metodo"])))
    return linhas, curvas


def figura(linhas, pasta, mostrar=False):
    plt = C.estilo_graficos(mostrar)
    ks = sorted({l["k"] for l in linhas}); xs = np.arange(len(ks))
    info = {l["k"]: l for l in linhas}
    n_seeds = len({l["seed"] for l in linhas})
    fig, ax = plt.subplots(figsize=(9.6, 5.0)); fim, numeros = {}, {}
    for m in METODOS:
        vals = [[l["retorno"] for l in linhas if l["k"] == k and l["metodo"] == m] for k in ks]
        ok = [j for j, v in enumerate(vals) if v]
        if not ok:
            continue
        med = [float(np.mean(vals[j])) for j in ok]
        ax.fill_between(xs[ok], [min(vals[j]) for j in ok], [max(vals[j]) for j in ok], color=COR[m], alpha=0.12, lw=0)
        ax.plot(xs[ok], med, "-" + MARCA[m], color=COR[m], lw=2, ms=7, mec="white", mew=1.2, label=ROTULO[m], solid_capstyle="round", zorder=3)
        fim[m] = med[-1]; numeros[m] = {str(ks[j]): v for j, v in zip(ok, med)}
    ax.plot(xs, [info[k]["retorno_dataset"] for k in ks], color=TINTA2, lw=1.2, ls=(0, (4, 3)), label="média do dataset", zorder=2)
    rotular_fim(ax, xs[-1], fim, {m: m for m in fim})
    ax.set_xticks(xs, [f"{k}\n{br(100 * info[k]['frac_transicoes'], 1)}%" for k in ks])
    ax.set(xlabel="episódios do especialista somados a todo o ruim (embaixo: % das transições)", ylabel="retorno (máx. 500)", ylim=(0, 530))
    ax.set_xlim(-0.25, len(ks) - 0.6)
    eixo_limpo(ax)
    fig.suptitle("Quanto especialista basta?", x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.905, f"Linha: média de {n_seeds} seed{'s' if n_seeds > 1 else ''}. Faixa: da pior à melhor seed.", fontsize=9, color=TINTA2)
    H, L = ax.get_legend_handles_labels()
    fig.legend(H, L, loc="upper right", ncol=len(L), frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0, 0.985, 0.90)); fig.savefig(os.path.join(pasta, "1_quanto_especialista.png"), dpi=150)
    if mostrar:
        plt.show()
    plt.close(fig)
    return numeros


def saidas(a):
    linhas, curvas = reunir(a)
    if not linhas:
        print("nada para resumir ainda"); return
    salvar_csv(os.path.join(a.pasta, "varredura.csv"), linhas)
    salvar_csv(os.path.join(a.pasta, "varredura_curvas.csv"), curvas)
    numeros = figura(linhas, a.pasta, a.mostrar)
    with open(os.path.join(a.pasta, "numeros.json"), "w", encoding="utf-8") as f:
        json.dump(dict(retorno_medio_por_k=numeros, seeds=sorted({l["seed"] for l in linhas})), f, indent=2, ensure_ascii=False)
    print(json.dumps(numeros, indent=2, ensure_ascii=False))
    print(f"resultados em {os.path.abspath(a.pasta)}")


def main():
    C.utf8_no_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--niveis", type=int, nargs="+", default=[1, 2, 5, 10, 25], help="episódios do especialista")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--metodos", nargs="+", default=list(METODOS), choices=METODOS)
    p.add_argument("--passos", type=int, default=15_000)
    p.add_argument("--episodios", type=int, default=50, help="episódios na avaliação final")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--device", default="cpu", help="onde treinar: cpu (padrão) ou cuda; a avaliação fica sempre na CPU")
    p.add_argument("--pasta", default="experimentos/varredura_especialista")
    p.add_argument("--raiz", default="experimentos", help="onde procurar os runs de k = 0 e k = 50")
    p.add_argument("--plano", action="store_true", help="só lista o que seria treinado")
    p.add_argument("--so-graficos", action="store_true")
    p.add_argument("--mostrar", action="store_true")
    a = p.parse_args()

    os.makedirs(a.pasta, exist_ok=True)
    if a.so_graficos:
        return saidas(a)
    jobs = [(k, m, a.passos, s, a.episodios, a.device) for s in a.seeds for k in a.niveis for m in a.metodos]
    feitos = [j for j in jobs if os.path.exists(arquivo(a.pasta, j[0], j[3], j[1]))]
    faltam = [j for j in jobs if j not in feitos]
    print(f"{len(jobs)} modelos no plano, {len(feitos)} já salvos em {a.pasta}, {len(faltam)} a treinar ({a.passos} passos, {a.workers} processos, {a.device})")
    if a.plano:
        for k in a.niveis:
            print(" ", C.resumo_dataset(f"k={k}", montar(k, a.seeds[0])), f"-> {br(100 * descrever(montar(k, a.seeds[0]), k)['frac_transicoes'], 1)}% das transições do especialista")
        return
    C.conferir_dispositivo(a.device)
    C.garantir_datasets()
    os.makedirs(os.path.join(a.pasta, "modelos"), exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=C.contexto_pool(a.device)) as ex:
        futuros = {ex.submit(treinar_nivel, j): j for j in faltam}
        for n, fut in enumerate(as_completed(futuros), 1):
            k, m, _, s, *_ = futuros[fut]; r = fut.result(); arq = arquivo(a.pasta, k, s, m)
            torch.save(r, arq + ".tmp"); os.replace(arq + ".tmp", arq)         # um modelo por arquivo: queda não perde o resto
            print(f"[{n}/{len(faltam)}] k={k} seed={s} {m}: retorno {r['retorno']:.1f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    saidas(a)


if __name__ == "__main__":
    main()
