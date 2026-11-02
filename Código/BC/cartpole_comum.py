"""
Peças comuns dos scripts de Offline RL no CartPole (bc_cartpole.py, cql_cartpole.py, iql_cartpole.py).

Datasets (gerados uma vez com o DataCollector do Minari e guardados no cache local do Minari):
  cartpole/expert-v0  heurística que equilibra o poste:          50 episódios, retorno 500
  cartpole/medio-v0   heurística com 30% das ações trocadas:     300 episódios, retorno ~100
  cartpole/ruim-v0    heurística com 60% das ações trocadas:    2000 episódios, retorno ~16
  misto               expert + ruim (montado na memória)

O CartPole é totalmente observável (posição, velocidade, ângulo, velocidade angular), tem 2 ações
(0 = empurrar para a esquerda, 1 = para a direita) e recompensa +1 por passo, até 500 passos.
"""
import copy, os
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym

ENV_ID = "CartPole-v1"
GAMMA = 0.99
N_ACOES, DIM_OBS = 2, 4
NOMES_ACOES = ["esquerda", "direita"]
NOMES_OBS = ["posição x", "velocidade v", "ângulo θ", "vel. angular ω"]
DATASETS = {   # nome -> (id no Minari, probabilidade de trocar a ação, nº de episódios)
    "expert": ("cartpole/expert-v0", 0.0, 50),
    "medio": ("cartpole/medio-v0", 0.3, 300),
    "ruim": ("cartpole/ruim-v0", 0.6, 2000),
}


def utf8_no_console():
    import sys
    for fluxo in (sys.stdout, sys.stderr):   # console do Windows (cp1252) não tem alguns caracteres
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def conferir_dispositivo(d):
    """--device: falha logo, antes de treinar, se pedirem GPU numa máquina sem CUDA."""
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {d}: este torch não enxerga CUDA (torch {torch.__version__}). Instale um build com CUDA ou use --device cpu.")


def contexto_pool(d):
    """CUDA não funciona depois de fork: fora da CPU, os processos nascem com spawn."""
    import multiprocessing
    return None if d == "cpu" else multiprocessing.get_context("spawn")


def acao_especialista(obs):
    """Heurística que equilibra o poste: empurra para o lado em que ele está caindo."""
    x, v, theta, omega = obs
    return int(theta + 0.5 * omega + 0.01 * x + 0.1 * v > 0)


# =============================================================================== datasets
def garantir_datasets(nomes=("expert", "medio", "ruim")):
    """Cria os datasets no cache local do Minari, se ainda não existirem."""
    import minari
    from minari import DataCollector
    locais = minari.list_local_datasets()
    for nome in nomes:
        did, p_erro, n_eps = DATASETS[nome]
        if did in locais:
            continue
        print(f"gerando {did} ({n_eps} episódios, {p_erro:.0%} das ações trocadas)...", flush=True)
        rng = np.random.default_rng(0)
        env = DataCollector(gym.make(ENV_ID))
        for ep in range(n_eps):
            obs, _ = env.reset(seed=100_000 + ep); fim = False
            while not fim:
                a = acao_especialista(obs)
                if rng.random() < p_erro:
                    a = 1 - a
                obs, r, term, trunc, _ = env.step(a); fim = term or trunc
        env.create_dataset(dataset_id=did, eval_env=ENV_ID, algorithm_name=f"heurística, troca p={p_erro}",
                           author="Offline RL no CartPole", ref_min_score=9.4, ref_max_score=500.0,
                           description=f"Heurística de equilíbrio com {p_erro:.0%} das ações trocadas")
        env.close()


def carregar(nome):
    """Arrays por transição: s, a, r, s2, d, o retorno descontado G, o episódio, o início do episódio,
    o retorno do episódio e se ele chegou aos 500 passos. Episódio truncado não acabou: G parte de
    1/(1-γ) = 100, o mesmo valor a que convergem os alvos de TD (d = 0)."""
    import minari
    if nome == "misto":
        a, b = carregar("expert"), carregar("ruim")
        b["ep"] = b["ep"] + a["ep"].max() + 1
        b["ini"] = b["ini"] + len(a["a"])
        return {k: np.concatenate([a[k], b[k]]) for k in a}
    ds = minari.load_dataset(DATASETS[nome][0])
    cols = {k: [] for k in ["s", "a", "r", "s2", "d", "G", "ep", "ret_ep", "sucesso"]}
    for i, ep in enumerate(ds.iterate_episodes()):
        T = len(ep.actions); o = ep.observations.astype(np.float32)
        cols["s"].append(o[:-1]); cols["s2"].append(o[1:])
        cols["a"].append(ep.actions.astype(np.int64)); cols["r"].append(ep.rewards.astype(np.float32))
        cols["d"].append(ep.terminations.astype(np.float32))          # truncamento (500 passos) não é terminal
        G = np.zeros(T, np.float32)
        g = 1.0 / (1.0 - GAMMA) if ep.truncations[-1] and not ep.terminations[-1] else 0.0
        for t in range(T - 1, -1, -1):
            g = ep.rewards[t] + GAMMA * g; G[t] = g
        cols["G"].append(G); cols["ep"].append(np.full(T, i))
        cols["ret_ep"].append(np.full(T, float(ep.rewards.sum()), np.float32))
        cols["sucesso"].append(np.full(T, bool(ep.truncations[-1])))
    D = {k: np.concatenate(v) for k, v in cols.items()}
    novo = np.r_[True, D["ep"][1:] != D["ep"][:-1]]
    D["ini"] = np.flatnonzero(novo)[np.cumsum(novo) - 1]
    return D


def resumo_dataset(nome, D):
    n_ep = len(np.unique(D["ep"])); ret = D["ret_ep"][np.r_[True, D["ep"][1:] != D["ep"][:-1]]]
    return f"dataset {nome}: {len(D['a']):,} transições, {n_ep} episódios, retorno médio {ret.mean():.1f}"


def mlp(entrada=DIM_OBS, saida=N_ACOES, h=256, dropout=0.0):
    camadas = [nn.Linear(entrada, h), nn.ReLU()]
    if dropout: camadas.append(nn.Dropout(dropout))
    camadas += [nn.Linear(h, h), nn.ReLU()]
    if dropout: camadas.append(nn.Dropout(dropout))
    camadas.append(nn.Linear(h, saida))
    return nn.Sequential(*camadas)


# =============================================================================== avaliação
def rede_na_cpu(net):
    """A avaliação fica sempre na CPU (simulador, um estado por vez). Treinando na GPU, avalia uma cópia."""
    return net if next(net.parameters()).device.type == "cpu" else copy.deepcopy(net).cpu()


class Agente:
    """Política gulosa: argmax da rede (logits de π ou valores de Q) na observação atual."""

    def __init__(self, net):
        self.net = net.eval()

    def reset(self, obs):
        pass

    def agir(self, obs, primeiro=False):
        with torch.no_grad():
            return int(self.net(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax())


def avaliar(agente, n=30, seed=10_000, passos_aleatorios=0):
    """Avaliação online no simulador. Retorna retorno médio, os retornos e a contagem de ações.
    Com passos_aleatorios > 0, as primeiras ações de cada episódio são sorteadas e só depois o agente
    assume. O sorteio depende só da seed do episódio: todo agente recebe os mesmos estados. Esses passos
    contam no retorno (máximo 500), mas não na contagem de ações."""
    env = gym.make(ENV_ID); rets, acoes = [], np.zeros(N_ACOES)
    for k in range(n):
        obs, _ = env.reset(seed=seed + k); fim, R, prim = False, 0.0, True
        sorteio = np.random.default_rng(seed + k)
        for _ in range(passos_aleatorios):
            if fim: break
            obs, r, term, trunc, _ = env.step(int(sorteio.integers(N_ACOES))); R += r; fim = term or trunc
        agente.reset(obs)
        while not fim:
            a = agente.agir(obs, primeiro=prim); prim = False
            acoes[a] += 1
            obs, r, term, trunc, _ = env.step(a); R += r; fim = term or trunc
        rets.append(R)
    return float(np.mean(rets)), np.array(rets), acoes


def rodar_episodio(agente, seed):
    """Um episódio: devolve as observações visitadas (para desenhar o ângulo ao longo do tempo)."""
    env = gym.make(ENV_ID); obs, _ = env.reset(seed=seed); agente.reset(obs)
    S, fim, prim = [obs], False, True
    while not fim:
        a = agente.agir(obs, primeiro=prim); prim = False
        obs, _, term, trunc, _ = env.step(a); fim = term or trunc; S.append(obs)
    return np.array(S)


def mapa_politica(net, n=81, th_max=0.21, om_max=2.0):
    """Ação escolhida pela rede no plano (ângulo, velocidade angular), com o carrinho parado no centro."""
    th, om = np.meshgrid(np.linspace(-th_max, th_max, n), np.linspace(-om_max, om_max, n))
    obs = np.stack([np.zeros_like(th), np.zeros_like(th), th, om], -1).reshape(-1, 4).astype(np.float32)
    with torch.no_grad():
        a = net.eval()(torch.as_tensor(obs)).argmax(1).numpy().reshape(n, n)
    return th, om, a


def desenhar_mapa(ax, net, titulo, D=None):
    """Mapa de política com a fronteira da heurística (θ + 0,5ω = 0) e, opcional, os estados do dataset."""
    import matplotlib.colors as mcolors
    th_max, om_max = 0.21, 2.0
    if D is not None:   # enquadra a região onde o dataset tem estados (com folga)
        th_max = float(np.clip(1.6 * np.percentile(np.abs(D["s"][:, 2]), 99.5), 0.04, 0.21))
        om_max = float(np.clip(1.6 * np.percentile(np.abs(D["s"][:, 3]), 99.5), 0.3, 2.0))
    th, om, a = mapa_politica(net, th_max=th_max, om_max=om_max)
    ax.pcolormesh(th, om, a, cmap=mcolors.ListedColormap(["#F2C9A8", "#A9C7E8"]), shading="auto", vmin=0, vmax=1)
    x = np.linspace(th.min(), th.max(), 10)
    ax.plot(x, -2 * x, "k--", lw=1.5, label="fronteira do especialista")
    if D is not None:
        sel = np.random.default_rng(0).choice(len(D["a"]), min(1500, len(D["a"])), replace=False)
        ax.scatter(D["s"][sel, 2], D["s"][sel, 3], s=3, color="#14213D", alpha=0.35, label="estados do dataset")
    ax.set(xlim=(th.min(), th.max()), ylim=(om.min(), om.max()), xlabel="ângulo θ (rad)", ylabel="vel. angular ω",
           title=titulo)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.02, 0.03, "laranja = esquerda · azul = direita", transform=ax.transAxes, fontsize=8, color="#333")


def estilo_graficos(mostrar=False):
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12})
    return plt
