"""
Vídeos dos agentes no CartPole, para as apresentações.

Carrega os modelos já treinados (os .pt das pastas figuras_*), roda um episódio de cada agente a partir
do mesmo estado inicial (seed 10000, o mesmo dos gráficos 5_resultado.png) e grava os episódios lado a
lado. Quando um poste cai, o painel congela e mostra o passo da queda; o vídeo acaba quando todos acabam.
Não treina nada: roda em segundos.

Vídeos em ./videos_cartpole/:
  1_politicas_dos_dados.mp4   a heurística com 0%, 30% e 60% das ações trocadas (expert, medio, ruim)
  2_bc_parada.mp4             BC base × + parada (medio 10%)         usa figuras_bc_cartpole/modelos_melhorias.pt
                              (episódio representativo da base: retorno mais perto da mediana da avaliação)
  3_cql_dqn.mp4               DQN offline × CQL (dataset expert)      usa figuras_cql_cartpole/modelos_expert.pt
  4_iql_ruim.mp4              BC × CQL × IQL (dataset ruim)           usa figuras_iql_cartpole/modelos.pt
  5_iql_misto.mp4             BC × CQL × IQL (dataset misto)          idem

Uso (da raiz do projeto):
  pip install "gymnasium[classic-control]" imageio imageio-ffmpeg
  python Código/videos_cartpole.py
  python Código/videos_cartpole.py --gif           # também em GIF (metade do tamanho, 25 quadros/s)
  python Código/videos_cartpole.py --seed 10001    # outro estado inicial
"""
import argparse, os, sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(AQUI, "IQL"), os.path.join(AQUI, "BC")]
import cartpole_comum as C
import bc_cartpole as BC
import iql_cartpole as IQL
import gymnasium as gym

CAB_VIDEO, CAB_PAINEL, SEP = 48, 76, 10
COR_CAIU, COR_OK, COR_TEXTO = "#B04A16", "#1F7A6D", "#14213D"


def fonte(tam):
    """DejaVu Sans Bold, que vem com o matplotlib: tem acentos, α e τ em qualquer sistema."""
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")), tam)


class Heuristica:
    """A política que gerou os dados: o especialista com uma fração p das ações trocadas."""

    def __init__(self, p, seed=0):
        self.p, self.rng = p, np.random.default_rng(seed)

    def reset(self, obs):
        pass

    def agir(self, obs, primeiro=False):
        a = C.acao_especialista(obs)
        return 1 - a if self.rng.random() < self.p else a


# =============================================================================== gravação
def painel(s, F):
    img = Image.fromarray(s["quadro"])
    if s["fim"]:
        img = Image.blend(img, Image.new("RGB", img.size, "white"), 0.4)
    W, H = img.size
    can = Image.new("RGB", (W, H + CAB_PAINEL), "white"); can.paste(img, (0, CAB_PAINEL))
    d = ImageDraw.Draw(can)
    d.text((W / 2, 10), s["rot"], fill=COR_TEXTO, font=F["titulo"], anchor="mt")
    d.text((W / 2, 46), f"passo {s['t']}", fill="#555", font=F["passo"], anchor="mt")
    if s["fim"]:
        motivo = "saiu do trilho" if abs(s["obs"][0]) > 2.4 else "caiu"      # término: |x| > 2,4 ou |θ| > 12°
        msg, cor = (f"{motivo} no passo {s['t']}", COR_CAIU) if s["caiu"] else (f"{s['t']} passos: de pé até o fim", COR_OK)
        d.text((W / 2, CAB_PAINEL + H / 2), msg, fill=cor, font=F["msg"], anchor="mm")
    return can


def compor(paineis, titulo, F):
    W = sum(p.width for p in paineis) + SEP * (len(paineis) - 1); H = paineis[0].height + CAB_VIDEO
    W16, H16 = -(-W // 16) * 16, -(-H // 16) * 16          # o H.264 pede lados múltiplos de 16
    can = Image.new("RGB", (W16, H16), "white"); d = ImageDraw.Draw(can); x = (W16 - W) // 2
    for j, p in enumerate(paineis):
        can.paste(p, (x, CAB_VIDEO)); x += p.width + SEP
        if j < len(paineis) - 1:
            d.line([(x - SEP // 2, CAB_VIDEO), (x - SEP // 2, H16)], fill="#C9CED6", width=2)
    d.text((W16 / 2, 12), titulo, fill=COR_TEXTO, font=F["video"], anchor="mt")
    return can


def gravar(agentes, titulo, arq, seed, fps, segurar, gif, F):
    """Roda os agentes lado a lado, do mesmo estado inicial, e grava um quadro por passo."""
    import imageio.v2 as imageio
    S = []
    for rot, ag in agentes:
        env = gym.make(C.ENV_ID, render_mode="rgb_array"); obs, _ = env.reset(seed=seed); ag.reset(obs)
        S.append(dict(rot=rot, ag=ag, env=env, obs=obs, quadro=env.render(), t=0, fim=False, caiu=False, prim=True))
    escritor = imageio.get_writer(arq, fps=fps, codec="libx264", quality=8, ffmpeg_log_level="error")
    quadros_gif, n = [], 0
    while segurar >= 0:
        img = compor([painel(s, F) for s in S], titulo, F)
        escritor.append_data(np.asarray(img))
        if gif and n % 2 == 0:
            quadros_gif.append(img.resize((img.width // 2, img.height // 2)).quantize(64))
        n += 1
        if all(s["fim"] for s in S):
            segurar -= 1
            continue
        for s in S:
            if s["fim"]:
                continue
            a = s["ag"].agir(s["obs"], primeiro=s["prim"]); s["prim"] = False
            s["obs"], _, term, trunc, _ = s["env"].step(a)
            s["quadro"], s["t"], s["fim"], s["caiu"] = s["env"].render(), s["t"] + 1, term or trunc, term
    escritor.close()
    for s in S:
        s["env"].close()
    if gif:
        quadros_gif[0].save(arq[:-4] + ".gif", save_all=True, append_images=quadros_gif[1:],
                            duration=int(2000 / fps), loop=0)
    print(f"  {arq}: " + " · ".join(f"{s['rot']} {s['t']} passos" for s in S), flush=True)


# =============================================================================== modelos
def carregar_pt(raiz, rel):
    arq = os.path.join(raiz, rel)
    if not os.path.exists(arq):
        print(f"  (sem {rel}: vídeo pulado)")
        return None
    return torch.load(arq, weights_only=False)


def agente_q(pesos):
    """Política gulosa de uma rede Q (DQN ou CQL) salva pelo cql_cartpole.py."""
    net = C.mlp(); net.load_state_dict(pesos)
    return C.Agente(net)


def main():
    C.utf8_no_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raiz", default=".", help="pasta onde estão as figuras_* com os modelos .pt")
    p.add_argument("--pasta", default="videos_cartpole")
    p.add_argument("--seed", type=int, default=10_000, help="estado inicial do episódio (10000 = o dos gráficos)")
    p.add_argument("--fps", type=int, default=50, help="50 = tempo real do CartPole")
    p.add_argument("--segurar", type=float, default=2.0, help="segundos com a imagem parada no fim")
    p.add_argument("--alpha", type=float, default=0.5, help="alpha do CQL no iql_cartpole")
    p.add_argument("--tau", type=float, default=0.7, help="tau do IQL")
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()

    os.makedirs(a.pasta, exist_ok=True)
    F = {"video": fonte(24), "titulo": fonte(24), "passo": fonte(18), "msg": fonte(30)}
    grava = lambda agentes, titulo, nome, seed=a.seed: gravar(agentes, f"{titulo} · episódio seed {seed}",
                                                              os.path.join(a.pasta, nome), seed, a.fps,
                                                              int(a.segurar * a.fps), a.gif, F)
    print("gravando vídeos...")

    grava([("expert: 0% trocadas", Heuristica(0.0, 1)), ("medio: 30% trocadas", Heuristica(0.3, 1)),
           ("ruim: 60% trocadas", Heuristica(0.6, 1))],
          "As políticas que geraram os datasets (π_β)", "1_politicas_dos_dados.mp4")

    res = carregar_pt(a.raiz, "figuras_bc_cartpole/modelos_melhorias.pt")
    if res:
        s0 = min(r["seed"] for r in res if r["nome"] == "base")
        um = lambda v: next(r for r in res if r["nome"] == v and r["seed"] == s0)
        # episódio representativo da base: o de retorno mais perto da mediana dos 50 episódios de avaliação
        rb = np.array(um("base")["retornos"]); sd = 10_000 + int(np.argmin(np.abs(rb - np.median(rb))))
        grava([("BC base (rede do último passo)", BC.montar(um("base"))),
               ("BC + parada antecipada", BC.montar(um("+ parada")))],
              "Behavior Cloning treinado no medio 10%", "2_bc_parada.mp4", seed=sd if a.seed == 10_000 else a.seed)

    res = carregar_pt(a.raiz, "figuras_cql_cartpole/modelos_expert.pt")
    if res:
        a_cql = min([al for al in res if al > 0], key=lambda al: (-res[al]["retorno"], al))   # como no cql_cartpole
        grava([("DQN offline (α = 0)", agente_q(res[0.0]["pesos"])), (f"CQL (α = {a_cql:g})", agente_q(res[a_cql]["pesos"]))],
              "Treinados no dataset expert", "3_cql_dqn.mp4")

    res = carregar_pt(a.raiz, "figuras_iql_cartpole/modelos.pt")
    if res:
        rots = ["BC", IQL.rotulo("CQL", {"alpha": a.alpha}), IQL.rotulo("IQL", {"tau": a.tau})]
        for j, ds in enumerate(("ruim", "misto")):
            rs = [next((r for r in res if r["dataset"] == ds and r["rotulo"] == rot), None) for rot in rots]
            if None in rs:
                print(f"  (faltam modelos do dataset {ds}: vídeo pulado)"); continue
            grava([(r["rotulo"], IQL.Agente.montar(r["algo"], r["pesos"])) for r in rs],
                  f"Treinados no dataset {ds}", f"{4 + j}_iql_{ds}.mp4")
    print(f"vídeos salvos em {os.path.abspath(a.pasta)}")


if __name__ == "__main__":
    main()
