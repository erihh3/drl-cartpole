"""
BC (Behavior Cloning) offline no CartPole, com as mesmas melhorias do bc_minigrid.py.

O BC trata o dataset como classificação: entrada = observação (x, v, θ, ω), rótulo = ação gravada,
perda = cross-entropy. Variantes que se somam:

  base          MLP, Adam, guarda a rede do ÚLTIMO passo
  + parada      parada antecipada: perda de validação a cada 500 passos, guarda a rede do MENOR
                valor e para quando ela não melhora por --paciencia passos
  + regulariz.  dropout 0,2 + weight decay (AdamW) + label smoothing 0,1
  + memória     empilha as últimas k observações e as k-1 ações anteriores (padrão k = 4)

Por que o dataset padrão é "medio 10%" (≈ 27 episódios da heurística com 30% das ações trocadas):
com o expert completo o BC base já faz 500 de 500 e não há o que melhorar. Com poucos dados e
rótulos ruidosos, treinar demais faz a rede decorar o ruído, e é aí que parada e regularização
fazem diferença. O CartPole é totalmente observável: a memória NÃO deve ajudar (diferente do
MiniGrid, que tem visão parcial). A variante fica para mostrar isso.

Estudos (--estudo):
  melhorias  (padrão) as 4 variantes no dataset medio 10% → figuras 1 a 5
  dados      BC com parada + regularização em vários datasets → figura 6
  todos      os dois

Figuras em ./figuras_bc_cartpole/:
  1_parada.png         perda de validação, ponto guardado e retorno ao longo do treino
  2_regularizacao.png  mapas de política sem e com regularização e perda de validação
  3_memoria.png        acurácia contra o dataset e contra o especialista, por variante
  4_comparacao.png     retorno final por variante (média ± desvio entre seeds)
  5_resultado.png      ângulo do poste num episódio: base × melhor variante
  6_dados.png          (estudo dados) qualidade e quantidade dos dados

Uso:
  pip install "minari[create]" gymnasium torch matplotlib
  python bc_cartpole.py
  python bc_cartpole.py --estudo todos
  python bc_cartpole.py --seeds 0 --passos 5000      # versão rápida
  python bc_cartpole.py --so-graficos
"""
import argparse, copy, json, os, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch, torch.nn.functional as F

import cartpole_comum as C

torch.set_num_threads(1)


def configs(k_memoria=4, wd=1e-3):
    return {
        "base":            dict(k=1, dropout=0.0, wd=0.0, ls=0.0, parada=False),
        "+ parada":        dict(k=1, dropout=0.0, wd=0.0, ls=0.0, parada=True),
        "+ regularização": dict(k=1, dropout=0.2, wd=wd, ls=0.1, parada=True),
        "+ memória":       dict(k=k_memoria, dropout=0.2, wd=wd, ls=0.1, parada=True),
    }


# estudo de dados: nome -> (dataset, filtro, fração dos episódios de treino)
DADOS = {
    "expert": ("expert", None, 1.0), "medio": ("medio", None, 1.0), "ruim": ("ruim", None, 1.0),
    "misto": ("misto", None, 1.0), "misto filtrado": ("misto", "sucesso", 1.0), "ruim top 10%": ("ruim", "top10", 1.0),
    "expert 2%": ("expert", None, 0.02), "expert 5%": ("expert", None, 0.05), "expert 10%": ("expert", None, 0.10),
}
COR = {"base": "#8A93A3", "+ parada": "#2A5BA8", "+ regularização": "#B07A00", "+ memória": "#1F7A6D"}


# =============================================================================== dados
def selecionar(D, filtro, fracao, seed=0):
    """Aplica filtro e fração de episódios; separa 10% dos episódios para validação."""
    ok = np.ones(len(D["a"]), bool)
    if filtro == "sucesso":
        ok &= D["sucesso"]
    elif filtro == "top10":
        ok &= D["ret_ep"] >= np.percentile(D["ret_ep"][D["ini"] == np.arange(len(D["a"]))], 90)
    eps = np.unique(D["ep"][ok])
    rng = np.random.default_rng(seed)
    val = set(rng.choice(eps, max(1, len(eps) // 10), replace=False).tolist())
    treino = np.array([e for e in eps if e not in val])
    if fracao < 1:
        treino = rng.choice(treino, max(1, int(round(len(treino) * fracao))), replace=False)
    em = lambda conj: ok & np.isin(D["ep"], np.fromiter(conj, int))
    return np.where(em(treino))[0], np.where(em(val))[0], len(treino)


def dim_entrada(k):
    return k * C.DIM_OBS + (k - 1) * C.N_ACOES


def codificar_hist(D, idx, k):
    """k últimas observações (da mais antiga à atual) e k-1 ações anteriores; no começo do episódio
    repete a primeira observação e usa 'nenhuma ação' (vetor zero)."""
    if k == 1:
        return torch.as_tensor(D["s"][idx])
    ini = D["ini"][idx]; partes = []
    for m in range(k - 1, -1, -1):
        partes.append(torch.as_tensor(D["s"][np.maximum(idx - m, ini)]))
    for m in range(k - 1, 0, -1):
        j = idx - m; valido = torch.as_tensor(j >= ini, dtype=torch.float32)[:, None]
        partes.append(F.one_hot(torch.as_tensor(D["a"][np.maximum(j, ini)]), C.N_ACOES).float() * valido)
    return torch.cat(partes, 1)


def rede(cfg):
    return C.mlp(entrada=dim_entrada(cfg["k"]), dropout=cfg["dropout"])


class Agente:
    """Política gulosa com a mesma memória do treino."""

    def __init__(self, net, k):
        self.net, self.k = net.eval(), k

    def reset(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32)[None]
        self.frames = deque([o] * self.k, maxlen=self.k)
        self.acoes = deque([torch.zeros(1, C.N_ACOES)] * (self.k - 1), maxlen=max(1, self.k - 1))

    def agir(self, obs, primeiro=False):
        if not primeiro:
            self.frames.append(torch.as_tensor(obs, dtype=torch.float32)[None])
        x = torch.cat(list(self.frames) + (list(self.acoes) if self.k > 1 else []), 1)
        with torch.no_grad():
            a = int(self.net(x).argmax())
        if self.k > 1:
            self.acoes.append(F.one_hot(torch.tensor([a]), C.N_ACOES).float())
        return a


def montar(r):
    net = rede(r["cfg"]); net.load_state_dict(r["pesos"])
    return Agente(net, r["cfg"]["k"])


# =============================================================================== treino
def treinar(job):
    nome, ds, filtro, fracao, cfg, passos, seed, paciencia, n_aval, n_curva = job
    D = C.carregar(ds)
    idx_tr, idx_val, n_eps = selecionar(D, filtro, fracao, seed)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    mon = rng.choice(idx_val, min(3000, len(idx_val)), replace=False)
    x_val, a_val = codificar_hist(D, mon, cfg["k"]), torch.as_tensor(D["a"][mon])
    a_esp = torch.as_tensor([C.acao_especialista(o) for o in D["s"][mon]])
    X_tr = codificar_hist(D, idx_tr, cfg["k"]); A_tr = torch.as_tensor(D["a"][idx_tr])

    net = rede(cfg)
    opt = (torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=cfg["wd"]) if cfg["wd"]
           else torch.optim.Adam(net.parameters(), 1e-3))
    log, acum = [], {"perda": [], "acc": []}
    log_cada, curva_cada = 500, max(500, passos // 10)
    melhor = dict(perda=np.inf, passo=0, pesos=None)
    t0 = time.time()
    for t in range(1, passos + 1):
        net.train()
        i = torch.as_tensor(rng.integers(len(idx_tr), size=256))
        logits = net(X_tr[i]); a = A_tr[i]
        perda = F.cross_entropy(logits, a, label_smoothing=cfg["ls"])
        opt.zero_grad(); perda.backward(); opt.step()
        acum["perda"].append(perda.item()); acum["acc"].append((logits.argmax(1) == a).float().mean().item())
        if t % log_cada == 0:
            net.eval()
            with torch.no_grad():
                lv = net(x_val)
                linha = dict(passo=t, perda=float(np.mean(acum["perda"])), acc_treino=float(np.mean(acum["acc"])),
                             perda_val=F.cross_entropy(lv, a_val).item(),
                             acc_val=(lv.argmax(1) == a_val).float().mean().item(),
                             acc_esp=(lv.argmax(1) == a_esp).float().mean().item())
            acum = {"perda": [], "acc": []}
            if linha["perda_val"] < melhor["perda"]:
                melhor = dict(perda=linha["perda_val"], passo=t, pesos=copy.deepcopy(net.state_dict()))
            if n_curva and t % curva_cada == 0:
                linha["retorno"] = C.avaliar(Agente(net, cfg["k"]), n=n_curva, seed=50_000)[0]
            log.append(linha)
            if cfg["parada"] and t - melhor["passo"] >= paciencia:
                break

    passo_final = log[-1]["passo"] if log else passos
    if cfg["parada"] and melhor["pesos"] is not None:
        net.load_state_dict(melhor["pesos"]); guardado = melhor["passo"]
    else:
        guardado = passo_final
    net.eval()
    with torch.no_grad():
        lv = net(x_val)
        val = dict(perda=F.cross_entropy(lv, a_val).item(), acc=(lv.argmax(1) == a_val).float().mean().item(),
                   acc_esp=(lv.argmax(1) == a_esp).float().mean().item())
    ret, rets, acoes = C.avaliar(Agente(net, cfg["k"]), n=n_aval)
    print(f"  {nome:22s} seed {seed}  guardado {guardado:6d} (parou {passo_final:6d})  acc_val={val['acc']:.0%}  "
          f"acc_esp={val['acc_esp']:.0%}  retorno={ret:6.1f}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(nome=nome, dataset=ds, filtro=filtro, fracao=fracao, cfg=cfg, seed=seed, n_eps=int(n_eps),
                n_trans=int(len(idx_tr)), log=log, passo_guardado=guardado, passo_final=passo_final, val=val,
                retorno=ret, retornos=rets.tolist(), acoes=acoes.tolist(),
                pesos={k: v.cpu() for k, v in net.state_dict().items()})


# =============================================================================== gráficos
def gerar_figuras(res, pasta, args, mostrar=False):
    plt = C.estilo_graficos(mostrar)
    os.makedirs(pasta, exist_ok=True)
    numeros = {}
    grupo = lambda nome: [r for r in res if r["nome"] == nome]
    media = lambda rs, f: (float(np.mean([f(r) for r in rs])), float(np.std([f(r) for r in rs])))

    def curva(ax, chave, rotulo, r, **kw):
        L = [l for l in r["log"] if chave in l]
        ax.plot([l["passo"] for l in L], [l[chave] for l in L], label=rotulo, **kw)

    def salvar(fig, nome, titulo):
        fig.suptitle(titulo, fontsize=13); fig.tight_layout(); fig.savefig(f"{pasta}/{nome}", dpi=120)

    variantes = [v for v in configs() if grupo(v)]
    if variantes:
        r0 = grupo(variantes[0])[0]
        ds_nome = f"{r0['dataset']} {r0['fracao']:.0%}" if r0["fracao"] < 1 else r0["dataset"]
        D = C.carregar(r0["dataset"])
        s0 = min(r["seed"] for r in grupo(variantes[0]))
        um = lambda v: next(r for r in grupo(v) if r["seed"] == s0)
        resumo = {v: dict(retorno=media(grupo(v), lambda r: r["retorno"]),
                          acc_val=media(grupo(v), lambda r: r["val"]["acc"]),
                          acc_esp=media(grupo(v), lambda r: r["val"]["acc_esp"]),
                          perda_val=media(grupo(v), lambda r: r["val"]["perda"]),
                          passo=media(grupo(v), lambda r: r["passo_guardado"]),
                          seeds=[r["retorno"] for r in grupo(v)]) for v in variantes}
        numeros["dataset_melhorias"] = ds_nome
        numeros["episodios_treino"] = r0["n_eps"]
        numeros["variantes"] = resumo

        # 1. parada antecipada
        if "base" in variantes and "+ parada" in variantes:
            b, p = um("base"), um("+ parada")
            fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
            curva(ax[0], "perda_val", "base: validação", b, color=COR["base"], lw=2)
            curva(ax[0], "perda", "base: treino", b, color=COR["base"], lw=1.2, ls=":")
            curva(ax[0], "perda_val", "+ parada: validação", p, color=COR["+ parada"], lw=2, ls="--")
            mp = next(l for l in p["log"] if l["passo"] == p["passo_guardado"])
            ax[0].plot(mp["passo"], mp["perda_val"], "*", color=COR["+ parada"], ms=18, mec="k",
                       label=f"rede guardada (passo {p['passo_guardado']})")
            ax[0].set(title="Perda (cross-entropy)", xlabel="passo de treino"); ax[0].legend(fontsize=9)
            curva(ax[1], "retorno", "base", b, color=COR["base"], lw=2, marker="o", ms=4)
            ax[1].axvline(p["passo_guardado"], color=COR["+ parada"], ls="--", lw=1.5,
                          label="passo de menor perda de validação")
            ax[1].set_ylim(0, 540); ax[1].legend(fontsize=9)
            ax[1].set(title=f"Retorno durante o treino da base ({args.curva} episódios)", xlabel="passo de treino",
                      ylabel="retorno")
            sb, sp = resumo["base"]["retorno"], resumo["+ parada"]["retorno"]
            salvar(fig, "1_parada.png", f"1. Parada antecipada ({ds_nome}): retorno {sb[0]:.0f} ± {sb[1]:.0f} "
                                        f"(última rede) → {sp[0]:.0f} ± {sp[1]:.0f} (menor perda de validação)")

        # 2. regularização: o mapa mostra o que a rede decorou
        if "+ regularização" in variantes:
            vs = [v for v in ("base", "+ parada", "+ regularização") if v in variantes]
            fig, ax = plt.subplots(1, len(vs) + 1, figsize=(5.2 * (len(vs) + 1), 4.8))
            for j, v in enumerate(vs):
                r = um(v)
                C.desenhar_mapa(ax[j], montar(r).net, f"{v}\nretorno {r['retorno']:.0f} · concorda c/ especialista "
                                                      f"{r['val']['acc_esp']:.0%}", D)
            xs = np.arange(len(variantes))
            m = [resumo[v]["perda_val"][0] for v in variantes]; e = [resumo[v]["perda_val"][1] for v in variantes]
            b = ax[-1].bar(xs, m, yerr=e, capsize=5, color=[COR[v] for v in variantes])
            ax[-1].bar_label(b, labels=[f"{x:.3f}" for x in m], padding=3)
            ax[-1].set_xticks(xs, variantes, rotation=15)
            ax[-1].set(title="Perda de validação da rede guardada\n(média ± desvio)", ylabel="cross-entropy")
            salvar(fig, "2_regularizacao.png", "2. O que a rede aprendeu: mapas de política (carrinho parado no "
                                               "centro) — a linha tracejada é a fronteira do especialista")

        # 3. memória
        if "+ memória" in variantes:
            fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
            xs, w_ = np.arange(len(variantes)), 0.38
            for dx, chave, nome, c in [(-w_ / 2, "acc_val", "contra o dataset (rótulos ruidosos)", "#8A93A3"),
                                       (w_ / 2, "acc_esp", "contra o especialista (sem ruído)", "#1F7A6D")]:
                m = [resumo[v][chave][0] for v in variantes]; e = [resumo[v][chave][1] for v in variantes]
                b = ax[0].bar(xs + dx, m, w_, yerr=e, capsize=4, color=c, label=nome)
                ax[0].bar_label(b, labels=[f"{x:.1%}" for x in m], padding=3, fontsize=9)
            ax[0].set_xticks(xs, variantes, rotation=15); ax[0].set_ylim(0.5, 1.05)
            ax[0].set(title="Acurácia de validação (média ± desvio)", ylabel="fração"); ax[0].legend(fontsize=9)
            for v in ("+ regularização", "+ memória"):
                if v in variantes:
                    curva(ax[1], "acc_esp", v, um(v), color=COR[v], lw=2)
            ax[1].set(title="Concordância com o especialista durante o treino", xlabel="passo de treino",
                      ylabel="fração")
            ax[1].legend(fontsize=9)
            rm, rr = resumo["+ memória"]["retorno"], resumo.get("+ regularização", {"retorno": (np.nan, 0)})["retorno"]
            salvar(fig, "3_memoria.png", f"3. Memória: o CartPole é totalmente observável, as observações passadas não "
                                         f"trazem informação nova (retorno {rr[0]:.0f} → {rm[0]:.0f})")

        # 4. comparação
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
        xs = np.arange(len(variantes))
        m = [resumo[v]["retorno"][0] for v in variantes]; e = [resumo[v]["retorno"][1] for v in variantes]
        b = ax[0].bar(xs, m, yerr=e, capsize=5, color=[COR[v] for v in variantes], alpha=0.9)
        for x, v in zip(xs, variantes):
            ax[0].scatter([x] * len(resumo[v]["seeds"]), resumo[v]["seeds"], color="#14213D", s=18, zorder=3)
        ax[0].bar_label(b, labels=[f"{x:.0f}" for x in m], padding=3)
        ax[0].set_xticks(xs, variantes, rotation=15); ax[0].set_ylim(0, 560)
        ax[0].set(title=f"Retorno final ({args.episodios} episódios por seed; pontos = seeds)", ylabel="retorno")
        pm = [resumo[v]["passo"][0] for v in variantes]
        b = ax[1].bar(xs, pm, color=[COR[v] for v in variantes])
        ax[1].bar_label(b, labels=[f"{x / 1000:.1f} mil" for x in pm], padding=3)
        ax[1].set_xticks(xs, variantes, rotation=15)
        ax[1].set(title="Passo da rede guardada (média)", ylabel="passo de treino")
        salvar(fig, "4_comparacao.png", f"4. Comparação das variantes no dataset {ds_nome} "
                                        f"({r0['n_eps']} episódios de treino, {len(resumo[variantes[0]]['seeds'])} seeds)")

        # 5. resultado
        melhor = max(variantes, key=lambda v: resumo[v]["retorno"][0])
        fig, ax = plt.subplots(figsize=(11, 4.8))
        for v in dict.fromkeys(["base", melhor]):
            if v not in variantes: continue
            for k_, sd in enumerate((10_000, 10_001, 10_002)):
                S_ep = C.rodar_episodio(montar(um(v)), sd)
                ax.plot(S_ep[:, 2], color=COR[v], lw=1.6, alpha=1 - 0.25 * k_,
                        label=f"{v}: {len(S_ep) - 1} passos (seed {sd})")
        for y in (0.2095, -0.2095):
            ax.axhline(y, color="#999", ls=":", lw=1)
        ax.set(title="Ângulo do poste em 3 episódios", xlabel="passo", ylabel="ângulo θ (rad)"); ax.legend(fontsize=8)
        salvar(fig, "5_resultado.png", f"5. O resultado no ambiente: base × {melhor} (seed de treino {s0})")

    # 6. estudo de dados
    nomes_d = [n for n in DADOS if grupo("dados: " + n)]
    if nomes_d:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
        q = [n for n in ("expert", "medio", "ruim", "misto", "misto filtrado", "ruim top 10%") if n in nomes_d]
        mm = [media(grupo("dados: " + n), lambda r: r["retorno"]) for n in q]
        b = ax[0].bar(range(len(q)), [x[0] for x in mm], yerr=[x[1] for x in mm], capsize=5, color="#1F7A6D")
        ax[0].bar_label(b, labels=[f"{x[0]:.0f}" for x in mm], padding=3)
        ret_ds = []
        for n in q:
            ds, filt, _ = DADOS[n]; Dd = C.carregar(ds); ok = np.ones(len(Dd["a"]), bool)
            if filt == "sucesso": ok &= Dd["sucesso"]
            if filt == "top10":
                ok &= Dd["ret_ep"] >= np.percentile(Dd["ret_ep"][Dd["ini"] == np.arange(len(Dd["a"]))], 90)
            prim = ok & (Dd["ini"] == np.arange(len(Dd["a"])))
            ret_ds.append(float(Dd["ret_ep"][prim].mean()))
        ax[0].scatter(range(len(q)), ret_ds, marker="_", s=900, color="k", lw=2, zorder=5, label="retorno médio do dataset")
        ax[0].set_xticks(range(len(q)), q, rotation=12); ax[0].set_ylim(0, 560); ax[0].legend(fontsize=9)
        ax[0].set(title="Qualidade: retorno do BC por dataset", ylabel="retorno")
        f = [n for n in ("expert 2%", "expert 5%", "expert 10%", "expert") if n in nomes_d]
        eps = [grupo("dados: " + n)[0]["n_eps"] for n in f]
        mf = [media(grupo("dados: " + n), lambda r: r["retorno"]) for n in f]
        ax[1].errorbar(eps, [x[0] for x in mf], yerr=[x[1] for x in mf], fmt="o-", color="#1F7A6D", lw=2, capsize=5)
        ax[1].set_xscale("log"); ax[1].set_ylim(0, 540)
        ax[1].set(title="Quantidade: retorno por nº de episódios do especialista", xlabel="episódios no treino (log)",
                  ylabel="retorno")
        numeros["dados"] = {n: media(grupo("dados: " + n), lambda r: r["retorno"]) for n in nomes_d}
        numeros["retorno_dos_datasets"] = dict(zip(q, ret_ds))
        salvar(fig, "6_dados.png", "6. Dados (BC com parada + regularização): qualidade e quantidade "
                                   "(média ± desvio entre seeds)")

    with open(f"{pasta}/numeros.json", "w", encoding="utf-8") as fh:
        json.dump(numeros, fh, indent=2, ensure_ascii=False)
    if mostrar:
        plt.show()
    plt.close("all")
    return numeros


def main():
    C.utf8_no_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--estudo", default="melhorias", choices=["melhorias", "dados", "todos"])
    p.add_argument("--dataset", default="medio", choices=["expert", "medio", "ruim", "misto"],
                   help="dataset do estudo de melhorias")
    p.add_argument("--fracao", type=float, default=0.10, help="fração dos episódios de treino usada nas melhorias")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--passos", type=int, default=20_000, help="máximo de passos de treino")
    p.add_argument("--paciencia", type=int, default=5_000)
    p.add_argument("--k", type=int, default=4, help="observações empilhadas na variante com memória")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--episodios", type=int, default=50, help="episódios na avaliação final")
    p.add_argument("--curva", type=int, default=10, help="episódios por ponto da curva de retorno (0 = desliga)")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--pasta", default="figuras_bc_cartpole")
    p.add_argument("--mostrar", action="store_true")
    p.add_argument("--so-graficos", action="store_true")
    a = p.parse_args()

    C.garantir_datasets()
    os.makedirs(a.pasta, exist_ok=True)
    arq = os.path.join(a.pasta, f"modelos_{a.estudo}.pt")
    if a.so_graficos:
        res = torch.load(arq, weights_only=False)
    else:
        cf = configs(a.k, a.weight_decay); jobs = []
        if a.estudo in ("melhorias", "todos"):
            for v, cfg in cf.items():
                for s in a.seeds:
                    jobs.append((v, a.dataset, None, a.fracao, cfg, a.passos, s, a.paciencia, a.episodios,
                                 a.curva if s == a.seeds[0] else 0))
        if a.estudo in ("dados", "todos"):
            for n, (ds, filt, fr) in DADOS.items():
                for s in a.seeds:
                    jobs.append(("dados: " + n, ds, filt, fr, cf["+ regularização"], a.passos, s, a.paciencia,
                                 a.episodios, 0))
        print(f"treinando {len(jobs)} modelos ({a.workers} processos; até {a.passos} passos cada)...")
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(treinar, jobs))
        torch.save(res, arq)
    print("gerando gráficos...")
    numeros = gerar_figuras(res, a.pasta, a, a.mostrar)
    print(json.dumps(numeros, indent=2, ensure_ascii=False))
    print(f"figuras salvas em {os.path.abspath(a.pasta)}")


if __name__ == "__main__":
    main()
