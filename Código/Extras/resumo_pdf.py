"""
Resumo "antes e agora" dos testes em PDF, em duas versões: 1 página e 3 páginas.

Lê só os CSVs de Historico/dados/: não treina, não avalia, não carrega modelos.
Todo número das tabelas vem dos CSVs; o script imprime cada valor usado com a sua fonte.

Uso (da raiz do projeto):
  python Código/Extras/coletar_historico.py     # atualiza os CSVs, se houve rodada nova
  python Código/Extras/resumo_pdf.py            # escreve Historico/resumo_1pagina.pdf e Historico/resumo_3paginas.pdf
"""
import csv, datetime, os, re, statistics as st

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath

from comum_extras import C, COR, TINTA, TINTA2, eixo_limpo

DADOS = "Historico/dados"
SAIDA = {1: "Historico/resumo_1pagina.pdf", 3: "Historico/resumo_3paginas.pdf"}
W, H, M = 595.3, 841.9, 40                      # A4 em pontos e margem
LARG = W - 2 * M
VERDE, VERMELHO, CINZA, REGUA, FUNDO = "#2E7D32", "#B3261E", "#8A8F98", "#D9DBDE", "#F2F0EB"
NB = " "                                   # espaço que não quebra
# Valores dos slides que não estão nos CSVs: Historico/HISTORIA.md, tabela "A mesma medida em cinco momentos"
SLIDES_BC_DADOS = {"misto filtrado": 500, "ruim top 10%": 72}
FONTES = []
MEDIDOR = TextToPath()                          # larguras sem hinting, em pontos: as mesmas do PDF


# ---------- números em pt-BR ----------
def num(x, casas=0, sinal=False):
    s = f"{abs(x):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("−" if x < 0 and s.strip("0,.") else "+" if sinal and x > 0 else "") + s


def pm(v, casas=0): return f"{num(v['media'], casas)} ± {num(v['dp'], casas)}"
def faixa(v, casas=0): return f"{num(v['minimo'], casas)} a {num(v['maximo'], casas)}"
def pct(x): return f"{num(100 * x)}%"


# ---------- leitura dos CSVs (cada valor fica anotado com a fonte) ----------
def ler(nome):
    with open(os.path.join(DADOS, nome), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def anota(valor, fonte):
    FONTES.append((fonte, valor)); return valor


def resumo_de(v): return {"media": st.mean(v), "dp": st.stdev(v) if len(v) > 1 else 0.0, "minimo": min(v), "maximo": max(v)}


class Dados:
    def __init__(s):
        s.resumo, s.comp, s.evo = ler("resumo_seeds.csv"), ler("comparacao_marcos.csv"), ler("evolucao.csv")
        s.sens, s.curvas = ler("sensibilidade.csv"), ler("curvas.csv")

    def seeds(s, dataset, rotulo, script="iql", aval="padrao"):
        l = next(r for r in s.resumo if r["script"] == script and r["dataset"] == dataset and r["rotulo"] == rotulo and r["avaliacao"] == aval)
        return anota({k: float(l[k]) for k in ("media", "dp", "minimo", "maximo")}, f"resumo_seeds.csv  {script} | {dataset} | {rotulo} | {aval}  (n={l['n_seeds']})")

    def marco(s, metodo, dataset, metrica, m):
        l = next(r for r in s.comp if r["metodo"] == metodo and r["dataset"] == dataset and r["metrica"] == metrica)
        return anota(float(l[m]) if l[m] else None, f"comparacao_marcos.csv  {metodo} | {dataset} | {metrica} | {m}")

    def evolucao(s, m, metodo, dataset, metrica):
        v = [float(r["valor"]) for r in s.evo if r["marco"] == m and r["metodo"] == metodo and r["dataset"] == dataset and r["metrica"] == metrica]
        assert v, (m, metodo, dataset, metrica)
        return anota(v, f"evolucao.csv  {m} | {metodo} | {dataset} | {metrica}")

    def sensib(s, bloco, dataset, rotulo):
        v = [float(r["retorno"]) for r in s.sens if r["bloco"] == bloco and r["dataset"] == dataset and r["rotulo"] == rotulo]
        assert len(v) == 5, (bloco, dataset, rotulo, len(v))
        return anota(resumo_de(v), f"sensibilidade.csv  {bloco} | {dataset} | {rotulo}  (5 seeds)")

    def q_final(s, m, dataset, rotulo):
        ls = [r for r in s.curvas if r["marco"] == m and r["dataset"] == dataset and r["rotulo"] == rotulo and r["q_dados"]]
        l = max(ls, key=lambda r: int(r["passo"]))
        return anota(float(l["q_dados"]), f"curvas.csv  {m} | {dataset} | {rotulo} | q_dados no passo {l['passo']}")


# ---------- página: texto com quebra medida, tabelas, caixas ----------
def partes_de(x, cor=TINTA):
    """'texto' ou [('texto', 'bold', '#cor'), ...] -> lista de (texto, peso, cor)."""
    if isinstance(x, str): x = [(x,)]
    return [(p[0], p[1] if len(p) > 1 else "normal", p[2] if len(p) > 2 else cor) for p in x]


class Pagina:
    def __init__(s, plt, n, total, rodape):
        s.fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
        s.y, s.n, s.total, s.rodape, s.medidas = M, n, total, rodape, {}

    def larg(s, txt, tam, peso):
        chave = (txt, tam, peso)
        if chave not in s.medidas:
            s.medidas[chave] = MEDIDOR.get_text_width_height_descent(txt, FontProperties(family="DejaVu Sans", size=tam, weight=peso), ismath=False)[0]
        return s.medidas[chave]

    def espaco(s, tam, peso): return s.larg("a a", tam, peso) - s.larg("aa", tam, peso)

    def quebra(s, partes, tam, largura):
        """Quebra palavra a palavra, medindo a largura real. Devolve linhas de trechos (texto, peso, cor, x) e a largura de cada linha."""
        linhas, atual, x = [], [], 0.0
        for txt, peso, cor in partes:
            txt = txt.replace(" = ", f"{NB}={NB}").replace(" ± ", f"{NB}±{NB}").replace(" a +", f"{NB}a{NB}+")
            txt = re.sub(r"\(caps?\. [^)]*\)", lambda m: m.group(0).replace(" ", NB), txt)
            for p in txt.split(" "):
                if not p: continue
                wp, we = s.larg(p, tam, peso), s.espaco(tam, peso)
                if atual and x + we + wp > largura: linhas.append((atual, x)); atual, x = [], 0.0
                if atual: x += we
                atual.append((p, peso, cor, x)); x += wp
        if atual: linhas.append((atual, x))
        return linhas

    def desenha_linha(s, linha, x0, base, tam):
        trechos = []                                     # junta palavras seguidas com o mesmo estilo
        for p, peso, cor, x in linha:
            if trechos and trechos[-1][1:3] == [peso, cor]: trechos[-1][0] += " " + p
            else: trechos.append([p, peso, cor, x])
        for txt, peso, cor, x in trechos:
            s.fig.text((x0 + x) / W, 1 - base / H, txt.replace(NB, " "), fontsize=tam, fontweight=peso, color=cor, va="baseline", ha="left", family="DejaVu Sans")

    def texto(s, x, tam=9.0, dx=0.0, largura=None, entre=1.36, cor=TINTA, depois=0.0):
        largura = largura or LARG - dx
        for linha, _ in s.quebra(partes_de(x, cor), tam, largura):
            s.desenha_linha(linha, M + dx, s.y + tam * 0.95, tam); s.y += tam * entre
        s.y += depois

    def item(s, x, tam=9.0, dx=0.0, largura=None, marca="•", cor_marca=TINTA2, entre=1.34, depois=3.0):
        largura = largura or LARG - dx
        s.fig.text((M + dx) / W, 1 - (s.y + tam * 0.95) / H, marca, fontsize=tam, color=cor_marca, fontweight="bold", va="baseline", family="DejaVu Sans")
        s.texto(x, tam, dx + tam * 1.25, largura - tam * 1.25, entre, depois=depois)

    def titulo(s, txt, sub=None):
        s.texto([(txt, "bold")], 18, entre=1.25)
        if sub: s.y += 2; s.texto(sub, 8.8, cor=TINTA2, entre=1.38)
        s.y += 4; s.regua(s.y, 1.2, TINTA); s.y += 10

    def secao(s, txt, nota=None, antes=8.0):
        s.y += antes; s.texto([(txt, "bold")], 11.5, entre=1.3)
        if nota: s.texto(nota, 8.2, cor=TINTA2, entre=1.32)
        s.y += 3

    def regua(s, y, lw=0.6, cor=REGUA, x0=0.0, x1=LARG):
        s.fig.add_artist(Line2D([(M + x0) / W, (M + x1) / W], [1 - y / H] * 2, lw=lw, color=cor, transform=s.fig.transFigure))

    def caixa(s, x, tam=10.0, cor_barra=TINTA, pad=8.0):
        linhas = s.quebra(partes_de(x), tam, LARG - 2 * pad - 4)
        alt = len(linhas) * tam * 1.36 + 2 * pad - tam * 0.36
        s.fig.add_artist(Rectangle((M / W, 1 - (s.y + alt) / H), LARG / W, alt / H, transform=s.fig.transFigure, color=FUNDO, lw=0))
        s.fig.add_artist(Rectangle((M / W, 1 - (s.y + alt) / H), 3 / W, alt / H, transform=s.fig.transFigure, color=cor_barra, lw=0))
        y = s.y + pad
        for linha, _ in linhas:
            s.desenha_linha(linha, M + pad + 4, y + tam * 0.95, tam); y += tam * 1.36
        s.y += alt

    def tabela(s, cab, linhas, larguras, alin, tam=8.4, pad=3.6, grupos=()):
        """cab e células: str ou partes ricas. alin: 'l', 'r' ou 'c' por coluna. grupos: índices de linha antes dos quais entra uma régua mais forte."""
        assert abs(sum(larguras) - LARG) < 1, sum(larguras)
        xs = [sum(larguras[:i]) for i in range(len(larguras))]

        def linha_tab(celulas, peso_cab=False):
            qs = [s.quebra(partes_de([(c, "bold", TINTA2)] if peso_cab and isinstance(c, str) else c), tam, larguras[i] - 2 * pad) for i, c in enumerate(celulas)]
            n = max(len(q) for q in qs); alt = n * tam * 1.3 + 2 * pad - tam * 0.3
            for i, q in enumerate(qs):
                for k, (lin, w) in enumerate(q):
                    folga = larguras[i] - 2 * pad - w
                    x = xs[i] + pad + {"l": 0, "r": folga, "c": folga / 2}[alin[i]]
                    s.desenha_linha(lin, M + x, s.y + pad + tam * 0.92 + k * tam * 1.3, tam)
            s.y += alt

        s.regua(s.y, 1.0, TINTA); linha_tab(cab, True); s.regua(s.y, 0.7, TINTA)
        for i, celulas in enumerate(linhas):
            if i and i in grupos: s.regua(s.y, 0.7, CINZA)
            elif i: s.regua(s.y, 0.4)
            linha_tab(celulas)
        s.regua(s.y, 1.0, TINTA); s.y += 2

    def eixos(s, altura, dx=0.0, largura=None, esq=0.0, dir_=0.0):
        largura = largura or LARG - dx
        ax = s.fig.add_axes([(M + dx + esq) / W, 1 - (s.y + altura) / H, (largura - esq - dir_) / W, altura / H]); s.y += altura
        return ax

    def fecha(s, pdf):
        assert s.y <= H - M + 4, f"página {s.n} estourou: y = {s.y:.0f} > {H - M:.0f}"
        s.fig.text(M / W, 1 - (H - M / 2) / H, s.rodape, fontsize=6.8, color=TINTA2, va="baseline", family="DejaVu Sans")
        s.fig.text((W - M) / W, 1 - (H - M / 2) / H, f"{s.n}/{s.total}", fontsize=6.8, color=TINTA2, va="baseline", ha="right", family="DejaVu Sans")
        pdf.savefig(s.fig); print(f"  página {s.n}/{s.total}: ocupa até y = {s.y:.0f} de {H - M:.0f} pt")


# ---------- conteúdo ----------
def ok(txt): return [("✓", "bold", VERDE), (txt,)]
def muda(txt): return [("✗", "bold", VERMELHO), (txt,)]


def numeros(d):
    """Junta, uma vez, tudo o que as páginas usam."""
    n = {"slide": {(m, ds): d.marco(m, ds, "retorno", "0_slides") for m in ("BC", "CQL", "IQL") for ds in ("expert", "medio", "ruim", "misto")}}
    n["agora"] = {(m, ds): d.seeds(ds, r) for m, r in (("BC", "BC"), ("CQL", "CQL (α=0.5)"), ("IQL", "IQL (τ=0.7)")) for ds in ("expert", "medio", "ruim", "misto")}
    n["dados"] = {ds: d.evolucao("4_seeds", "dados", ds, "retorno")[0] for ds in ("expert", "medio", "ruim", "misto")}
    n["tau_slide"] = [d.marco("IQL", "ruim", f"retorno_tau_{t}", "0_slides") for t in ("0.5", "0.7", "0.9")]
    n["tau_agora"] = [d.seeds("ruim", f"IQL (τ={t})") for t in ("0.5", "0.7", "0.9")]
    n["tau_60k"] = d.evolucao("5_longo_60k", "IQL", "ruim", "retorno_tau_0.9")[0]
    n["dqn_ret"] = d.seeds("expert", "DQN (α=0)", "cql")
    n["dqn_vies"] = resumo_de(d.evolucao("4_seeds", "DQN", "expert", "vies"))
    n["dqn_conc"] = d.evolucao("4_seeds", "DQN", "expert", "concordancia")
    n["cql01_vies"] = resumo_de(d.evolucao("4_seeds", "CQL", "expert", "vies"))
    n["cql_igual"] = resumo_de(d.evolucao("4_seeds", "CQL", "ruim", "igual_especialista"))
    n["bc_dados"] = {k: (d.evolucao("8_bc_dados", "BC", k, "estudo_dados.retorno")[0], d.evolucao("8_bc_dados", "BC", k, "estudo_dados.retorno_dp")[0]) for k in SLIDES_BC_DADOS}
    n["q60"] = {m: d.q_final("9_longo_expert", "expert", r) for m, r in (("CQL", "CQL (α=0.5)"), ("IQL", "IQL (τ=0.7)"))}
    n["s15"] = {(r, ds): d.sensib(f"alfa_{ds}", ds, r) for r in ("DQN (α=0)", "CQL (α=0.5)", "CQL (α=2)", "CQL (α=5)") for ds in ("ruim", "misto")}
    n["s60"] = {(r, ds): d.sensib(f"longo_{ds}", ds, r) for r in ("DQN (α=0)", "CQL (α=0.5)", "CQL (α=2)", "CQL (α=5)") for ds in ("ruim", "misto")}
    for r in ("IQL (τ=0.7)", "BC"):
        for ds in ("ruim", "misto"):
            n["s15"][(r, ds)] = d.sensib("parada_n15000", ds, r); n["s60"][(r, ds)] = d.sensib("iql_longo", ds, r)
    n["alto"] = {(r, ds): d.sensib(f"alfa_alto_{ds}", ds, r) for r in ("CQL (α=10)", "CQL (α=20)", "CQL (α=50)") for ds in ("ruim", "misto", "medio")}
    for ds in ("ruim", "misto", "medio"):
        n["alto"][("CQL (α=5)", ds)] = d.sensib(f"alfa_{ds}", ds, "CQL (α=5)")
    n["parada"] = {p: d.sensib(f"parada_n{p}", "misto", "CQL (α=0.5)") for p in (3000, 15000)}
    return n


def linhas_antes_agora(n):
    a, sl = n["agora"], n["slide"]; t_sl, t_ag = n["tau_slide"], n["tau_agora"]; bd = n["bc_dados"]
    dv = n["dqn_vies"]; cm = a[("CQL", "medio")]; cx = a[("CQL", "misto")]
    vies = f"{num(dv['minimo'], sinal=True)} a {num(dv['maximo'], sinal=True)}"
    tau = lambda vs: " / ".join(num(v) for v in vs)
    return [["IQL no ruim (τ = 0,7)", num(sl[("IQL", "ruim")]), pm(a[("IQL", "ruim")]), ok("mantém: acima do CQL em toda seed")],
         ["CQL no medio (α = 0,5)", num(sl[("CQL", "medio")]), pm(cm), ok(f"mantém: dentro da faixa ({faixa(cm)})")],
         ["CQL no ruim (α = 0,5)", num(sl[("CQL", "ruim")]), pm(a[("CQL", "ruim")]), ok("mantém")],
         ["CQL no misto (α = 0,5)", num(sl[("CQL", "misto")]), pm(cx), muda(f"muda: {faixa(cx)} entre seeds")],
         ["BC no misto (regularizado)", num(sl[("BC", "misto")]), pm(a[("BC", "misto")]), muda("muda: o 140 não se reproduz")],
         ["DQN offline no expert, retorno", num(n["slide_dqn"], 1), pm(n["dqn_ret"], 1), ok("mantém: falha em todas as seeds")],
         ["DQN no expert, viés de Q", num(n["slide_vies"], sinal=True), vies, muda("muda: tamanho de Q varia com a seed")],
         ["CQL α = 0,1 no expert, viés", num(n["slide_cql01"]), pm(n["cql01_vies"], 2), muda("muda: régua corrigida (G truncado)")],
         ["IQL no ruim, τ = 0,5 / 0,7 / 0,9", tau(t_sl), tau(v["media"] for v in t_ag), muda("muda: o retorno não cai com τ")],
         ["BC no misto filtrado", num(SLIDES_BC_DADOS["misto filtrado"]), f"{num(bd['misto filtrado'][0])} ± {num(bd['misto filtrado'][1])} (3 seeds)", ok("mantém")],
         ["BC no ruim top 10%", num(SLIDES_BC_DADOS["ruim top 10%"]), f"{num(bd['ruim top 10%'][0])} ± {num(bd['ruim top 10%'][1])} (3 seeds)", muda(f"muda: refeito, dá {num(bd['ruim top 10%'][0])}")]]


def listas(n):
    a, s15, s60 = n["agora"], n["s15"], n["s60"]; dr = n["dqn_ret"]; dv = n["dqn_vies"]; t = n["tau_agora"]
    vies = f"{num(dv['minimo'], sinal=True)} a {num(dv['maximo'], sinal=True)}"
    q60 = num(n['q60']['CQL']) if num(n['q60']['CQL']) == num(n['q60']['IQL']) else f"{num(n['q60']['CQL'])} e {num(n['q60']['IQL'])}"
    manteve = [
        [("DQN offline falha no expert", "bold"), (f"em todas as seeds (retorno de {faixa(dr, 1)}); qualquer α > 0 no CQL conserta: 500 em todas. (caps. 1 e 7)",)],
        [("Dados expert levam todos a 500.", "bold"), (f"No ruim, o BC fica abaixo dos próprios dados ({num(a[('BC', 'ruim')]['media'], 1)} contra {num(n['dados']['ruim'], 1)}); no medio, faz 500 (dados: {num(n['dados']['medio'])}). (cap. 7)",)],
        [("IQL acima do CQL α = 0,5", "bold"), ("no medio, no ruim e no misto, aos 15.000 passos, em todas as seeds. (cap. 7)",)],
        [("IQL no misto: 500", "bold"), ("em todas as seeds, com todos os τ e β testados e até 60.000 passos. (caps. 10 e 11)",)]]
    mudou = [
        [("“Q da ordem de 10⁴” no DQN", "bold"), (f"era uma seed: o viés vai de {vies}. Fica a falha, não o tamanho de Q. (cap. 7)",)],
        [("“O retorno cai quando τ cresce”:", "bold"), (f"com 5 seeds é {' / '.join(num(v['media']) for v in t)}. O τ = 0,9 só cai com treino longo ({num(n['tau_60k'])} aos 60.000 passos, uma seed). (cap. 8)",)],
        [("“Só o IQL volta a 500 no misto”:", "bold"), (f"o CQL α = 0,5 está em {num(n['parada'][3000]['media'])} aos 3.000 passos nas 5 seeds e cai para {num(n['parada'][15000]['media'])} aos 15.000. (caps. 7 e 10)",)],
        [("“O IQL é melhor que o CQL, sempre”:", "bold"), (f"depende de α e do tempo. Com α = 5, o CQL empata no medio e no misto aos 15.000 passos; aos 60.000, cai para {num(s60[('CQL (α=5)', 'misto')]['media'])} no misto e {num(s60[('CQL (α=5)', 'ruim')]['media'])} no ruim, contra {num(s60[('IQL (τ=0.7)', 'misto')]['media'])} e {num(s60[('IQL (τ=0.7)', 'ruim')]['media'])} do IQL. (caps. 10 e 11)",)],
        [("“Q ≈ 53: o CQL é conservador”:", "bold"), (f"Q ≈ 53 mede tempo de treino. Uma fórmula sem penalidade acerta Q, e aos 60.000 passos no expert CQL e IQL chegam a {q60} (máximo: 100). (caps. 6 e 8)",)]]
    return manteve, mudou


FRASE = [("A frase que resiste a tudo:", "bold"), ("o IQL mantém o que aprende. O CQL pode alcançá-lo, mas precisa do α certo e da hora certa de parar, e offline não dá para conferir nenhum dos dois.",)]
ABERTO = ["Vento (cap. 12): só a heurística foi medida; as políticas treinadas não.",
          "medio com 60.000 passos: não testado (com α de 5 a 50 está em 500 aos 15.000).",
          "A causa da queda do CQL não foi isolada (o máximo no alvo é a explicação coerente).",
          "Uma seed só: τ = 0,9 aos 60.000 passos e o expert aos 60.000 passos."]


def legenda_situacao(): return [("✓", "bold", VERDE), ("se mantém", "normal", TINTA2), ("   ✗", "bold", VERMELHO), ("mudou ou precisa ser corrigido no slide", "normal", TINTA2)]


def pagina_unica(plt, n, pdf, rodape):
    p = Pagina(plt, 1, 1, rodape)
    p.titulo("Offline RL no CartPole: antes e agora",
             f"Resumo dos testes em 1 página · {datetime.date.today():%d/%m/%Y}. Retorno no CartPole-v1 (máximo 500), 50 episódios por modelo. "
             "“Slides” é o que as apresentações mostram hoje; “agora” é média ± desvio de 5 seeds (0 a 4) aos 15.000 passos, salvo indicação.")
    p.secao("A mesma medida, antes e agora", antes=0)
    L = linhas_antes_agora(n)
    p.tabela(["Medida", "Slides", "Agora", "Situação"], L, [LARG - 371, 76, 92, 203], "lrrl", tam=8.6)
    p.texto(legenda_situacao(), 7.6, depois=2)
    y0 = p.y; meio = LARG / 2; manteve, mudou = listas(n)
    p.secao("O que se manteve", antes=8)
    for x in manteve: p.item(x, 9.0, largura=meio - 10, marca="✓", cor_marca=VERDE)
    y1, p.y = p.y, y0
    p.y += 8; p.texto([("O que mudou", "bold")], 11.5, dx=meio + 6, entre=1.3); p.y += 3
    for x in mudou: p.item(x, 9.0, dx=meio + 6, marca="✗", cor_marca=VERMELHO)
    p.y = max(p.y, y1) + 8
    p.caixa(FRASE, 10)
    p.secao("Em aberto", antes=10)
    for x in ABERTO: p.item(x, 9.0, depois=1)
    p.fecha(pdf)


def grafico_halteres(ax, n):
    """Slides (vazado) → agora (cheio, média de 5 seeds) com a faixa entre seeds, por dataset e método."""
    eixo_limpo(ax); ax.grid(False); ax.grid(axis="x", color="#E6E6E3", lw=0.8)
    y, rotulos, posicoes = 0.0, [], []
    for ds in ("medio", "ruim", "misto"):
        topo = y
        ax.text(-8, y - 0.55, f"{ds}", fontsize=9.5, fontweight="bold", color=TINTA, ha="right", va="center")
        for m in ("BC", "CQL", "IQL"):
            sl, ag = n["slide"][(m, ds)], n["agora"][(m, ds)]
            ax.plot([sl, ag["media"]], [y, y], color="#B9BDC4", lw=1.0, zorder=1)
            if ag["maximo"] > ag["minimo"]: ax.plot([ag["minimo"], ag["maximo"]], [y + 0.3] * 2, color=COR[m], lw=2.0, alpha=0.75, solid_capstyle="round", zorder=1)
            ax.plot(sl, y, "o", ms=7.5, mfc="white", mec=TINTA2, mew=1.3, zorder=3)
            ax.plot(ag["media"], y, "o", ms=8.5, color=COR[m], mec="white", mew=1.6, zorder=4)
            ax.text(1.015, y, f"{num(sl)} → {num(ag['media'])}", transform=ax.get_yaxis_transform(), fontsize=8.2, color=TINTA, va="center")
            rotulos.append({"BC": "BC", "CQL": "CQL α = 0,5", "IQL": "IQL τ = 0,7"}[m]); posicoes.append(y); y += 1.0
        dmed = n["dados"][ds]
        ax.plot([dmed, dmed], [topo - 0.45, y - 0.55], color=CINZA, lw=1.0, ls=(0, (3, 2)), zorder=0)
        ax.text(dmed + 5, topo - 0.62, f"dados: {num(dmed, 1 if dmed < 50 else 0)}", fontsize=7.2, color=TINTA2, va="center")
        y += 0.9
    ax.set_yticks(posicoes, rotulos, fontsize=8.2, color=TINTA2); ax.set_ylim(y - 0.6, -1.1)
    ax.set_xlim(0, 510); ax.set_xticks(range(0, 501, 100)); ax.tick_params(axis="x", labelsize=8, colors=TINTA2)
    ax.set_xlabel("retorno (máximo 500)", fontsize=8.2, color=TINTA2)
    ax.text(1.015, -1.05, "slides → agora", transform=ax.get_yaxis_transform(), fontsize=7.6, color=TINTA2, va="center", fontweight="bold")
    h = [Line2D([], [], marker="o", ls="", ms=7.5, mfc="white", mec=TINTA2, mew=1.3, label="slides"),
         Line2D([], [], marker="o", ls="", ms=8.5, color=TINTA2, mec="white", label="agora: média de 5 seeds"),
         Line2D([], [], color=TINTA2, lw=2.0, alpha=0.75, label="pior a melhor seed"),
         Line2D([], [], color=CINZA, lw=1.0, ls=(0, (3, 2)), label="média do dataset")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7.8, handletextpad=0.4, columnspacing=1.4)


def pagina_quadro(plt, n, pdf, rodape):
    p = Pagina(plt, 1, 3, rodape)
    p.titulo("Offline RL no CartPole: antes e agora",
             f"Resumo dos testes em 3 páginas · {datetime.date.today():%d/%m/%Y}. Retorno no CartPole-v1 (máximo 500), 50 episódios por modelo, "
             "5 seeds (0 a 4) e 15.000 passos, salvo indicação. Detalhes e fontes em Historico/HISTORIA.md.")
    p.secao("O enredo", antes=0)
    p.texto("Começamos corrigindo detalhes da medição e descobrimos que os números andavam sozinhos entre rodadas: parte do que os slides contam "
            "era sorte de uma seed. Repetimos tudo com cinco seeds e depois variamos α, β, τ e o tempo de treino. A comparação entre CQL e IQL "
            "mudou duas vezes no caminho (caps. 10 e 11).", 9.2, depois=6)
    p.caixa(FRASE, 10)
    p.secao("O resultado principal agora", "Média ± desvio entre 5 seeds. BC regularizado (dropout, weight decay, label smoothing), sem parada antecipada.")
    a = n["agora"]; ds4 = ("expert", "medio", "ruim", "misto")
    L = [["Média do dataset"] + [num(n["dados"][ds], 1 if n["dados"][ds] < 50 else 0) for ds in ds4]]
    L += [[nome] + [pm(a[(m, ds)], 1 if a[(m, ds)]["media"] < 50 else 0) for ds in ds4] for m, nome in (("BC", "BC regularizado"), ("CQL", "CQL α = 0,5"), ("IQL", "IQL τ = 0,7"))]
    p.tabela(["Retorno", "expert", "medio", "ruim", "misto"], L, [LARG - 360, 90, 90, 90, 90], "lrrrr", tam=8.8, grupos=(1,))
    p.secao("Slides → agora, por método e dataset", "No expert, todos os métodos fazem 500, antes e agora. Ponto vazado: o que os slides mostram. Ponto cheio: média de 5 seeds.")
    p.y += 18
    ax = p.eixos(330, esq=88, dir_=72); grafico_halteres(ax, n)
    p.y += 26
    p.texto([("Como ler:", "bold"), ("o IQL subiu no ruim e segue em 500 no medio e no misto; o CQL no misto varia muito entre seeds; o BC no misto ficou abaixo do 140 dos slides, "
             "que não se reproduz em nenhuma rodada.",)], 8.6)
    p.fecha(pdf)


def pagina_momentos(plt, n, d, pdf, rodape):
    p = Pagina(plt, 2, 3, rodape)
    p.secao("A mesma medida em cada momento", "“1ª rodada”: valores citados no PROJECT_GUIDE.md (os arquivos não existem mais). “01/11”: commit 6759143. "
            "“02/11”: correções de medição e BC regularizado (seed 0, 50 episódios). “5 seeds”: seeds 0 a 4.", antes=0)
    a, c = n["agora"], (lambda m, ds, met, mc: d.marco(m, ds, met, mc))
    def col(m, ds, met, casas=0, sinal=False, f=1.0, suf=""):
        return [("—" if (v := c(m, ds, met, mc)) is None else num(v * f, casas, sinal) + suf) for mc in ("0_slides", "1_original", "2_reexecucao", "3_melhorias")]
    tau = lambda mc: "—" if c("IQL", "ruim", "retorno_tau_0.5", mc) is None else " / ".join(num(c("IQL", "ruim", f"retorno_tau_{t}", mc)) for t in ("0.5", "0.7", "0.9"))
    bc10 = [("—" if (v := c("BC", "medio 10%", "base.retorno", mc)) is None else f"{num(v)} ± {num(c('BC', 'medio 10%', 'base.retorno_dp', mc))}") for mc in ("0_slides", "1_original", "2_reexecucao", "3_melhorias")]
    bd, dc = n["bc_dados"], n["dqn_conc"]
    n_50 = sum(1 for v in dc if round(v, 2) == 0.50)
    L = [["IQL no ruim, retorno"] + col("IQL", "ruim", "retorno") + [pm(a[("IQL", "ruim")])],
         ["CQL no medio, retorno"] + col("CQL", "medio", "retorno") + [pm(a[("CQL", "medio")])],
         ["CQL no ruim, retorno"] + col("CQL", "ruim", "retorno") + [pm(a[("CQL", "ruim")])],
         ["CQL no misto, retorno"] + col("CQL", "misto", "retorno") + [pm(a[("CQL", "misto")])],
         ["BC no misto, retorno"] + col("BC", "misto", "retorno") + [pm(a[("BC", "misto")])],
         ["BC base no medio 10%, retorno"] + bc10 + ["—"],
         ["BC no misto filtrado"] + [num(SLIDES_BC_DADOS["misto filtrado"]), "—", "—", "—"] + [f"{num(bd['misto filtrado'][0])} ± {num(bd['misto filtrado'][1])} (3)"],
         ["BC no ruim top 10%"] + [num(SLIDES_BC_DADOS["ruim top 10%"]), "—", "—", "—"] + [f"{num(bd['ruim top 10%'][0])} ± {num(bd['ruim top 10%'][1])} (3)"],
         ["DQN no expert, retorno"] + col("DQN", "expert", "retorno_alfa_0", 1) + [pm(n["dqn_ret"], 1)],
         ["DQN no expert, viés médio de Q"] + col("DQN", "expert", "vies", sinal=True) + [f"{num(n['dqn_vies']['minimo'], sinal=True)} a {num(n['dqn_vies']['maximo'], sinal=True)}"],
         ["CQL α = 0,1 no expert, viés médio"] + [("—" if (v := c("CQL", "expert", "vies", mc)) is None else num(v, 0 if mc == "0_slides" else 1)) for mc in ("0_slides", "1_original", "2_reexecucao", "3_melhorias")] + [pm(n["cql01_vies"], 2)],
         ["DQN no expert, igual aos dados"] + col("DQN", "expert", "concordancia", f=100, suf="%") + [f"{pct(dc[0])} (1) / {pct(0.5)} ({n_50})"],
         ["CQL no ruim, igual ao especialista"] + col("CQL", "ruim", "igual_especialista", f=100, suf="%") + [f"{pct(n['cql_igual']['media'])} ± {num(100 * n['cql_igual']['dp'])}"],
         ["IQL no ruim, τ = 0,5 / 0,7 / 0,9"] + [tau(mc) for mc in ("0_slides", "1_original", "2_reexecucao", "3_melhorias")] + [" / ".join(num(v["media"]) for v in n["tau_agora"])]]
    p.tabela(["Medida", "Slides", "1ª rodada", "01/11", "02/11", "5 seeds"], L, [LARG - 371, 76, 56, 78, 78, 83], "lrrrrr", tam=8.3, grupos=(5, 8))
    p.texto("Retorno máximo 500; viés = Q previsto menos o retorno descontado real (máximo possível de Q: 100). (n) = número de seeds. "
            "Os dois valores do BC filtrado nos slides vêm da tabela de HISTORIA.md, não dos CSVs.", 7.4, cor=TINTA2, depois=2)
    manteve, mudou = listas(n)
    p.secao("O que se manteve", antes=10)
    for x in manteve: p.item(x, 9.4, marca="✓", cor_marca=VERDE, depois=4)
    p.secao("O que caiu ou mudou", antes=10)
    for x in mudou: p.item(x, 9.4, marca="✗", cor_marca=VERMELHO, depois=4)
    p.fecha(pdf)


def pagina_sensibilidade(plt, n, pdf, rodape):
    p = Pagina(plt, 3, 3, rodape)
    p.secao("CQL × IQL depois da sensibilidade (caps. 10 e 11)", "Até o cap. 9, o CQL usava α = 0,5 fixo. Depois: α de 0 a 50 e treino até 60.000 passos, "
            "5 seeds, em GPU. Os números de GPU ficam separados da tabela principal (CPU): a maior diferença entre as médias é de 12 pontos.", antes=0)
    s15, s60 = n["s15"], n["s60"]
    nomes = (("DQN (α=0)", "CQL α = 0 (DQN)"), ("CQL (α=0.5)", "CQL α = 0,5"), ("CQL (α=2)", "CQL α = 2"), ("CQL (α=5)", "CQL α = 5"), ("IQL (τ=0.7)", "IQL τ = 0,7"), ("BC", "BC regularizado"))
    cas = lambda v: 1 if v["media"] < 50 else 0
    L = [[nome] + [pm(s[(r, ds)], cas(s[(r, ds)])) for ds in ("ruim", "misto") for s in (s15, s60)] for r, nome in nomes]
    p.texto([("O α forte só atrasa a queda.", "bold"), ("Retorno, média ± desvio de 5 seeds:",)], 9.2, depois=3)
    p.tabela(["Método", "ruim, 15k", "ruim, 60k", "misto, 15k", "misto, 60k"], L, [LARG - 380, 95, 95, 95, 95], "lrrrr", tam=9.0, grupos=(4,))
    al = n["alto"]
    L2 = [[ds] + [pm(al[(r, ds)], cas(al[(r, ds)])) for r in ("CQL (α=5)", "CQL (α=10)", "CQL (α=20)", "CQL (α=50)")]
          + [pm(n["s15"][("BC", ds)], cas(n["s15"][("BC", ds)])) if ds != "medio" else f"{pm(n['agora'][('BC', 'medio')])} (CPU)"] for ds in ("ruim", "misto", "medio")]
    p.y += 6; p.texto([("Acima de α = 5, o CQL vira BC.", "bold"), ("Retorno aos 15.000 passos:",)], 9.2, depois=3)
    p.tabela(["Dataset", "α = 5", "α = 10", "α = 20", "α = 50", "BC"], L2, [LARG - 420, 84, 84, 84, 84, 84], "lrrrrr", tam=9.0)
    c5m, c5r = s60[("CQL (α=5)", "misto")], s60[("CQL (α=5)", "ruim")]
    p.secao("Leitura", antes=8)
    for x in [[("Aos 15.000 passos, α = 5 empata com o IQL", "bold"), (f"no medio e no misto (500) e chega perto no ruim ({num(s15[('CQL (α=5)', 'ruim')]['media'])} contra {num(s15[('IQL (τ=0.7)', 'ruim')]['media'])}).",)],
              [("Aos 60.000, todo α cai.", "bold"), (f"No misto, α = 5 vai a {num(c5m['media'])} ({faixa(c5m)} entre seeds); no ruim, todos terminam perto de 100. "
                f"O IQL fica em {num(s60[('IQL (τ=0.7)', 'misto')]['media'])} no misto e {num(s60[('IQL (τ=0.7)', 'ruim')]['media'])} no ruim.",)],
              [("Penalidade demais copia os dados:", "bold"), (f"com α = 50 no ruim, o CQL faz {num(al[('CQL (α=50)', 'ruim')]['media'], 1)}, o mesmo do BC. α vai do Q-learning (α = 0) ao BC (α alto).",)],
              [("Escolher α sem jogar não funciona:", "bold"), ("o critério do menor viés de Q erra em 2 dos 3 datasets (cap. 11). Offline, nada no treino avisa o α nem a hora de parar.",)]]:
        p.item(x, 9.2, depois=3)
    p.secao("Em aberto", antes=8)
    for x in ABERTO: p.item(x, 9.0, depois=1)
    p.secao("Frases dos slides que precisam mudar", antes=8)
    t, t_sl = n["tau_agora"], n["tau_slide"]; dv = n["dqn_vies"]; a = n["agora"]
    for x in [[("IQL, slide 11:", "bold"), (f"retorno por τ {' → '.join(num(v) for v in t_sl)} (“queda”). Com 5 seeds: {' / '.join(num(v['media']) for v in t)}, sem queda contínua.",)],
              [("IQL, slide 20:", "bold"), (f"“o IQL é o único que chega a 500 no misto e o melhor no ruim ({num(n['slide'][('IQL', 'ruim')])})”. O CQL também chega a 500 no misto, mas cai com o treino; "
                f"IQL no ruim: {pm(a[('IQL', 'ruim')])}. Trocar a figura pelo quadro de 5 seeds da página 1.",)],
              [("CQL, slides 11, 12 e 16:", "bold"), (f"“Q ≈ 2.700 / 2.450, viés +2.737”. O DQN falha em todas as seeds, mas o viés vai de {num(dv['minimo'], sinal=True)} a {num(dv['maximo'], sinal=True)}: o tamanho depende da seed.",)],
              [("CQL, slides 12 e 16:", "bold"), ("“o limite inferior prometido”. No expert, Q ≈ 53 mede tempo de treino (caps. 6 e 8).",)],
              [("BC, slide 15:", "bold"), (f"BC no misto {num(n['slide'][('BC', 'misto')])} → {pm(a[('BC', 'misto')])} (regularizado); BC filtrado no ruim {num(SLIDES_BC_DADOS['ruim top 10%'])} → "
                f"{num(n['bc_dados']['ruim top 10%'][0])}; IQL no ruim {num(n['slide'][('IQL', 'ruim')])} → {num(a[('IQL', 'ruim')]['media'])}.",)]]:
        p.item(x, 9.0, depois=3)
    p.fecha(pdf)


def main():
    plt = C.estilo_graficos(False)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    d = Dados(); n = numeros(d)
    n["slide_dqn"] = d.marco("DQN", "expert", "retorno_alfa_0", "0_slides")
    n["slide_vies"] = d.marco("DQN", "expert", "vies", "0_slides")
    n["slide_cql01"] = d.marco("CQL", "expert", "vies", "0_slides")
    rodape = f"Fonte: Historico/dados/*.csv · detalhes em Historico/HISTORIA.md · gerado em {datetime.date.today():%d/%m/%Y} por Código/Extras/resumo_pdf.py"
    print("Escrevendo", SAIDA[1])
    with PdfPages(SAIDA[1], metadata={"Title": "Offline RL no CartPole: antes e agora (1 página)"}) as pdf:
        pagina_unica(plt, n, pdf, rodape)
    print("Escrevendo", SAIDA[3])
    with PdfPages(SAIDA[3], metadata={"Title": "Offline RL no CartPole: antes e agora (3 páginas)"}) as pdf:
        pagina_quadro(plt, n, pdf, rodape); pagina_momentos(plt, n, d, pdf, rodape); pagina_sensibilidade(plt, n, pdf, rodape)
    print(f"\nValores usados ({len(FONTES)}), com a fonte:")
    vistos = set()
    for fonte, valor in FONTES:
        if fonte in vistos: continue
        vistos.add(fonte)
        v = {k: round(x, 3) for k, x in valor.items()} if isinstance(valor, dict) else [round(x, 3) for x in valor] if isinstance(valor, list) else valor
        print(f"  {fonte} -> {v}")


if __name__ == "__main__":
    main()
