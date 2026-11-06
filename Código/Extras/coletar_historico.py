"""
Junta num lugar só tudo o que o projeto já mediu, marco por marco, para gerar gráficos depois sem retreinar nada.

Lê e escreve assim:
  - resultados antigos: lidos do commit 6759143 (git show), não da pasta figuras_*. Promover um run novo para
    figuras_* depois não muda o histórico.
  - resultados novos: lidos de experimentos/. O que ainda não existe aparece como "pendente" em marcos.csv.
  - Historico/snapshots/: cópias dos numeros.json, PNGs e logs de cada marco. Nunca são apagadas nem sobrescritas:
    se um arquivo mudou, a versão nova entra ao lado, com o hash no nome.
  - Historico/dados/: CSVs derivados, refeitos a cada coleta a partir dessas fontes.

Só lê resultados: não treina, não avalia modelos, não usa os datasets. Pode rodar quantas vezes quiser:
  python Código/Extras/coletar_historico.py
"""
import csv, glob, hashlib, io, json, os, re, shutil, subprocess

import numpy as np
import torch

COMMIT = "6759143"
RAIZ, SAIDA, EXP = ".", "Historico", "experimentos"
DADOS, FOTOS = os.path.join(SAIDA, "dados"), os.path.join(SAIDA, "snapshots")
SEEDS = {f"iql_s{s}": s for s in range(5)}
SEEDS_CQL = {f"cql_s{s}": s for s in range(5)}

MARCOS = [
    ("0_slides", "antes de 2026-11-01", "Números nos slides", "o que as apresentações ainda mostram", "Apresentações/*.pptx"),
    ("1_original", "antes de 2026-11-01", "Primeira rodada guardada", "os numeros.json originais não existem mais: restam os valores citados no PROJECT_GUIDE", "PROJECT_GUIDE.md §9 e §10"),
    ("2_reexecucao", "2026-11-01", "Reexecução guardada no commit", "mesmo código, rodada nova: mede o quanto os números variam sozinhos (IQL com 30 episódios)", f"{COMMIT}:figuras_bc_cartpole; {COMMIT}:figuras_cql_cartpole; {COMMIT}:figuras_iql_cartpole"),
    ("3_melhorias", "2026-11-02", "Correções na medição e no BC", "G com bootstrap no truncamento, BC regularizado, 50 episódios, teste com ações iniciais sorteadas", f"{EXP}/run_20261102"),
    ("4_seeds", "2026-11-02", "Cinco seeds", "CQL e IQL com seeds 0 a 4", f"{EXP}/iql_s*, {EXP}/cql_s*"),
    ("5_longo_60k", "2026-11-02", "60.000 passos", "a queda do CQL persiste? Q converge para cerca de 100?", f"{EXP}/iql_60k"),
    ("6_varredura", "2026-11-02", "Quanto especialista basta?", "todo o ruim + k episódios do especialista", f"{EXP}/varredura_especialista"),
    ("7_vento", "2026-11-02", "Vento", "políticas prontas avaliadas com vento constante e rajadas", f"{EXP}/vento"),
    ("8_bc_dados", "2026-11-03", "Estudo de dados do BC", "qualidade e quantidade dos dados, com parada antecipada e regularização", f"{EXP}/bc_dados"),
    ("9_longo_expert", "2026-11-03", "60.000 passos no expert", "Q segue a fórmula até perto de 100?", f"{EXP}/iql_60k_expert"),
    ("10_sensibilidade", "2026-11-03", "Sensibilidade, em GPU", "parar antes, α do CQL, β e τ do IQL, com cinco seeds", f"{EXP}/sensibilidade_gpu"),
    ("11_alfa_alto_longo", "2026-11-04", "α alto e 60.000 passos, em GPU", "α acima de 5 e treino longo: o α forte cura a queda ou só a atrasa?", f"{EXP}/sensibilidade_gpu2"),
]

# Sem numeros.json: transcritos das fontes citadas. Mesmos nomes de métrica dos numeros.json, para comparar marcos.
S_IQL, S_CQL, S_BC = "Implicit Q-Learning_CartPole.pptx", "Conservative_Q-Learning_CartPole.pptx", "Behavior_Cloning_CartPole.pptx"
ANTIGOS = [("0_slides", "BC", ds, "retorno", v, f"{S_IQL} slide 17") for ds, v in dict(expert=500, medio=500, ruim=9, misto=140).items()]
ANTIGOS += [("0_slides", "CQL", ds, "retorno", v, f"{S_IQL} slide 17") for ds, v in dict(expert=500, medio=372, ruim=126, misto=191).items()]
ANTIGOS += [("0_slides", "IQL", ds, "retorno", v, f"{S_IQL} slide 17") for ds, v in dict(expert=500, medio=500, ruim=187, misto=500).items()]
ANTIGOS += [("0_slides", "IQL", "ruim", f"retorno_tau_{t}", v, f"{S_IQL} slide 11") for t, v in zip((0.5, 0.7, 0.9), (212, 187, 159))]
ANTIGOS += [("0_slides", "IQL", "ruim", f"V_tau_{t}", v, f"{S_IQL} slide 11") for t, v in zip((0.5, 0.7, 0.9), (8.6, 17.7, 43.9))]
ANTIGOS += [("0_slides", m, "ruim", "igual_especialista", v, f"{S_IQL} slide 19") for m, v in dict(BC=0.01, CQL=0.73, IQL=0.88).items()]
ANTIGOS += [("0_slides", "DQN", "expert", "retorno_alfa_0", 9.5, "PROJECT_GUIDE.md §9"), ("0_slides", "DQN", "expert", "vies", 2737, f"{S_CQL} slide 12"), ("0_slides", "CQL", "expert", "vies", -27, f"{S_CQL} slide 12"),
            ("0_slides", "DQN", "expert", "concordancia", 0.50, "PROJECT_GUIDE.md §9"), ("0_slides", "CQL", "expert", "concordancia", 0.996, "PROJECT_GUIDE.md §9"),
            ("0_slides", "BC", "medio 10%", "base.retorno", 385, f"{S_BC} slide 12"), ("0_slides", "BC", "medio 10%", "base.retorno_dp", 35, f"{S_BC} slide 12")]
G9, G10 = "PROJECT_GUIDE.md §9", "PROJECT_GUIDE.md §10"
ANTIGOS += [("1_original", "BC", "medio 10%", "base.retorno", 356, G9), ("1_original", "BC", "medio 10%", "base.retorno_dp", 90, G9),
            ("1_original", "DQN", "expert", "retorno_alfa_0", 9.3, G9), ("1_original", "DQN", "expert", "vies", 19045, G9),
            ("1_original", "CQL", "expert", "vies", -26.7, G9), ("1_original", "DQN", "expert", "concordancia", 0.36, G9),
            ("1_original", "CQL", "expert", "concordancia", 0.998, G9), ("1_original", "CQL", "medio", "retorno", 430, G9),
            ("1_original", "IQL", "ruim", "retorno", 217, G10), ("1_original", "BC", "misto", "retorno", 92, G10), ("1_original", "CQL", "ruim", "igual_especialista", 0.66, G10)]


def md5(b):
    return hashlib.md5(b).hexdigest()


def git(caminho):
    return subprocess.run(["git", "show", f"{COMMIT}:{caminho}"], capture_output=True, check=True).stdout


def ler(caminho):
    """"6759143:pasta/arq" vem do commit; o resto, do disco."""
    if caminho.startswith(COMMIT + ":"):
        return git(caminho.split(":", 1)[1])
    with open(caminho, "rb") as f:
        return f.read()


manifesto = []
def fonte(marco, caminho):
    b = ler(caminho); manifesto.append(dict(marco=marco, arquivo=caminho, bytes=len(b), md5=md5(b)))
    return b


def guardar(marco, sub, caminho):
    """Cópia para Historico/snapshots. Nunca sobrescreve: versão diferente entra ao lado, com o hash no nome."""
    b = fonte(marco, caminho); nome = os.path.basename(caminho); pasta = os.path.join(FOTOS, marco, sub); os.makedirs(pasta, exist_ok=True)
    alvo = os.path.join(pasta, nome)
    if os.path.exists(alvo):
        with open(alvo, "rb") as f:
            if md5(f.read()) == md5(b):
                return
        r, e = os.path.splitext(nome); alvo = os.path.join(pasta, f"{r}__{md5(b)[:8]}{e}")
    if not os.path.exists(alvo):
        with open(alvo, "wb") as f:
            f.write(b)


def achatar(d, pre=""):
    for k, v in d.items():
        nome = f"{pre}{k}"
        if isinstance(v, dict):
            yield from achatar(v, nome + ".")
        elif isinstance(v, list):
            for j, x in enumerate(v):
                if isinstance(x, (int, float)):
                    yield f"{nome}.{j}", x
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield nome, v


def linhas_numeros(marco, seed, caminho, n_ep):
    """Um numeros.json vira linhas (método, dataset, métrica, valor)."""
    d = json.loads(fonte(marco, caminho)); ds0 = d.get("dataset_analise", d.get("dataset", ""))
    n_ep = d.get("episodios", n_ep); ls = []
    def add(metodo, dataset, metrica, valor):
        ls.append(dict(marco=marco, seed=seed, metodo=metodo, dataset=dataset, metrica=metrica, valor=valor, n_episodios=n_ep, fonte=caminho))
    for nome, v in achatar(d):
        if nome in ("beta", "alpha_cql", "episodios", "episodios_treino", "passos_aleatorios") or nome.startswith("hp_BC"):
            continue
        m = re.fullmatch(r"retorno_(perturbado_)?(BC|CQL.*|IQL.*)\.(\w+)", nome)
        if m:
            add(m[2].split()[0], m[3], "retorno_perturbado" if m[1] else "retorno", v)
        elif nome.startswith("retorno_dataset."):
            add("dados", nome.split(".")[1], "retorno", v)
        elif nome.startswith("retorno_por_tau."):
            add("IQL", ds0, f"retorno_tau_{nome.split('.', 1)[1]}", v)
        elif nome.startswith("V_por_tau."):
            add("IQL", ds0, f"V_tau_{nome.split('.', 1)[1]}", v)
        elif nome.startswith("retorno_por_alfa."):
            al = nome.split(".", 1)[1]; add("DQN" if float(al) == 0 else "CQL", ds0, f"retorno_alfa_{al}", v)
        elif nome.startswith("dados."):                                     # bc_cartpole --estudo dados
            _, n, j = nome.split("."); add("BC", n, "estudo_dados.retorno" + ("_dp" if j == "1" else ""), v)
        elif nome.startswith("retorno_dos_datasets."):
            add("dados", nome.split(".")[1], "retorno", v)
        elif nome.startswith("variantes."):
            _, var, met, *j = nome.split(".")
            if met == "seeds":
                add("BC", "medio 10%", f"{var}.retorno_seed_{j[0]}", v)
            elif j:
                add("BC", "medio 10%", f"{var}.{met}" + ("_dp" if j[0] == "1" else ""), v)
        else:
            m = re.fullmatch(r"(.*?)_?(DQN|CQL|IQL|BC)(\.\d+)?", nome)
            if m:
                add(m[2], ds0, m[1] + (m[3] or ""), v)
            else:
                add("", ds0, nome, v)
    return ls


def modelos(arquivo_bytes):
    """Lista de (rótulo, dataset, resultado) de qualquer um dos três formatos de .pt."""
    r = torch.load(io.BytesIO(arquivo_bytes), weights_only=False)
    if isinstance(r, dict):                                                 # cql_cartpole: {alpha: resultado}
        return [("DQN (α=0)" if a == 0 else f"CQL (α={a:g})", "expert", x, None) for a, x in r.items()]
    if "nome" in r[0]:                                                      # bc_cartpole: variantes por seed
        return [(f"BC {x['nome']}", f"{x['dataset']} {x['fracao']:.0%}" if x["fracao"] < 1 else x["dataset"], x, x["seed"]) for x in r]
    return [(x["rotulo"], x["dataset"], x, None) for x in r]


def ic(x, rng):
    """Bootstrap. Muitas reamostragens: com retornos bimodais (10 ou 500) o percentil anda aos saltos."""
    return np.percentile(rng.choice(x, (50_000, len(x))).mean(1), [2.5, 97.5])


def do_pt(marco, seed, caminho, eps, curvas, ics):
    script = re.search(r"figuras_(bc|cql|iql)_cartpole", caminho)[1]            # CQL α = 0,5 no expert sai de dois scripts
    for rot, ds, r, s in modelos(fonte(marco, caminho)):
        s = seed if s is None else s
        a = np.array(r["retornos"]); p = np.array(r.get("retornos_perturbados", [])); rng = np.random.default_rng(0)
        eps.extend(dict(marco=marco, script=script, seed=s, dataset=ds, rotulo=rot, avaliacao="padrao", episodio=j, retorno=float(v)) for j, v in enumerate(a))
        eps.extend(dict(marco=marco, script=script, seed=s, dataset=ds, rotulo=rot, avaliacao="perturbada", episodio=j, retorno=float(v)) for j, v in enumerate(p))
        for l in r["log"]:
            curvas.append(dict(marco=marco, script=script, seed=s, dataset=ds, rotulo=rot, **l))
        lo, hi = ic(a, rng)
        d = dict(marco=marco, script=script, seed=s, dataset=ds, rotulo=rot, dispositivo=r.get("dispositivo", "cpu"), n_episodios=len(a), retorno=float(a.mean()), ic95_inf=float(lo), ic95_sup=float(hi), frac_abaixo_de_100=float((a < 100).mean()))
        if len(p):
            lp, hp = ic(p, rng); ld, hd = ic(p - a, rng)
            d.update(perturbado=float(p.mean()), perturbado_ic95_inf=float(lp), perturbado_ic95_sup=float(hp), diferenca=float((p - a).mean()), diferenca_ic95_inf=float(ld), diferenca_ic95_sup=float(hd))
        ics.append(d)


def sensibilidade():
    """Marcos 10 e 11, em GPU: ficam em CSVs próprios, sem misturar com os resultados em CPU. Os PNGs não vão para o snapshot."""
    linhas, curvas, marcos = [], [], set()
    pastas = [("10_sensibilidade", p) for p in sorted(glob.glob(f"{EXP}/sensibilidade_gpu/*/figuras_*_cartpole"))]
    pastas += [("11_alfa_alto_longo", p) for p in sorted(glob.glob(f"{EXP}/sensibilidade_gpu2/*/figuras_*_cartpole"))]
    for marco, pasta in pastas:
        run = os.path.basename(os.path.dirname(pasta)); bloco, seed = re.fullmatch(r"(.+)_s(\d)", run).groups(); marcos.add(marco)
        guardar(marco, run, os.path.join(pasta, "numeros.json"))
        beta = json.loads(ler(os.path.join(pasta, "numeros.json"))).get("beta", "")
        r = torch.load(io.BytesIO(fonte(marco, glob.glob(os.path.join(pasta, "modelos*.pt"))[0])), weights_only=True, map_location="cpu")
        if isinstance(r, dict):                                             # cql_cartpole: o .pt não guarda o dataset; vem do nome da pasta
            itens = [("DQN (α=0)" if a == 0 else f"CQL (α={a:g})", bloco.split("_")[-1], "CQL", a, x) for a, x in r.items()]
        else:
            itens = [(x["rotulo"], x["dataset"], x["algo"], x["hp"].get("alpha", x["hp"].get("tau", "")), x) for x in r]
        for rot, ds, algo, ajuste, x in itens:
            a, f = np.array(x["retornos"]), x["log"][-1]; p = x.get("retorno_perturbado", "")
            linhas.append(dict(marco=marco, bloco=bloco, seed=seed, dataset=ds, rotulo=rot, metodo=algo, ajuste=ajuste, beta=beta if algo == "IQL" else "", passos=f["passo"], dispositivo=x.get("dispositivo", "cpu"),
                               n_episodios=len(a), retorno=float(a.mean()), mediana=float(np.median(a)), minimo=float(a.min()), maximo=float(a.max()), retorno_perturbado=p,
                               q_dados=f.get("q_dados", ""), q_outra=f.get("q_outra", ""), gap=f.get("gap", ""), perda_td=f.get("perda_td", ""), concordancia_dados=f.get("concordancia", "")))
            for l in x["log"]:
                if "retorno" in l:
                    curvas.append(dict(marco=marco, bloco=bloco, seed=seed, dataset=ds, rotulo=rot, passo=l["passo"], retorno=l["retorno"], q_dados=l.get("q_dados", ""), q_outra=l.get("q_outra", ""), perda_td=l.get("perda_td", "")))
    salvar("sensibilidade.csv", linhas); salvar("sensibilidade_curvas.csv", curvas)
    return marcos


def salvar(nome, linhas):
    if not linhas:
        return
    os.makedirs(DADOS, exist_ok=True); colunas = list(dict.fromkeys(k for l in linhas for k in l))
    with open(os.path.join(DADOS, nome), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, colunas); w.writeheader(); w.writerows(linhas)


def main():
    ev = [dict(marco=m, seed=0, metodo=me, dataset=ds, metrica=mt, valor=v, n_episodios=30 if m == "0_slides" and "retorno" in mt and ds != "medio 10%" else "", fonte=fo) for m, me, ds, mt, v, fo in ANTIGOS]
    eps, curvas, ics = [], [], []

    # marco 2: fixo no commit, para não mudar se figuras_* for substituída
    ev += linhas_numeros("2_reexecucao", 0, f"{COMMIT}:figuras_iql_cartpole/numeros.json", 30)
    ev += linhas_numeros("2_reexecucao", "0-2", f"{COMMIT}:figuras_bc_cartpole/numeros.json", 50)
    ev += linhas_numeros("2_reexecucao", 0, f"{COMMIT}:figuras_cql_cartpole/numeros.json", 50)
    do_pt("2_reexecucao", 0, f"{COMMIT}:figuras_iql_cartpole/modelos.pt", eps, curvas, ics)
    do_pt("2_reexecucao", 0, f"{COMMIT}:figuras_bc_cartpole/modelos_melhorias.pt", eps, curvas, ics)
    do_pt("2_reexecucao", 0, f"{COMMIT}:figuras_cql_cartpole/modelos_expert.pt", eps, curvas, ics)
    arvore = subprocess.run(["git", "ls-tree", "-r", "--name-only", COMMIT], capture_output=True, text=True, check=True).stdout.split("\n")
    for arq in arvore:
        for pref, marco in (("figuras_iql_cartpole/", "2_reexecucao"), ("figuras_bc_cartpole/", "2_reexecucao"), ("figuras_cql_cartpole/", "2_reexecucao"), ("Output/", "2_reexecucao")):
            if arq.startswith(pref) and arq.endswith((".json", ".png", ".txt")):
                guardar(marco, os.path.dirname(arq), f"{COMMIT}:{arq}")

    # marcos novos: entram quando existirem
    existe = {}
    def novo(marco, seed, pasta, pt, n_ep=50):
        js, arq_pt = os.path.join(pasta, "numeros.json"), os.path.join(pasta, pt)
        if not (os.path.exists(js) and os.path.exists(arq_pt)):
            return
        existe[marco] = True; ev.extend(linhas_numeros(marco, seed, js, n_ep)); do_pt(marco, seed, arq_pt, eps, curvas, ics)
        for f in glob.glob(os.path.join(pasta, "*.json")) + glob.glob(os.path.join(pasta, "*.png")):
            guardar(marco, os.path.relpath(pasta, EXP), f)
    novo("3_melhorias", 0, f"{EXP}/run_20261102/figuras_iql_cartpole", "modelos.pt")
    novo("3_melhorias", 0, f"{EXP}/run_20261102/figuras_cql_cartpole", "modelos_expert.pt")
    for run, s in SEEDS.items():
        novo("4_seeds", s, f"{EXP}/{run}/figuras_iql_cartpole", "modelos.pt")
    for run, s in SEEDS_CQL.items():
        novo("4_seeds", s, f"{EXP}/{run}/figuras_cql_cartpole", "modelos_expert.pt")
    novo("5_longo_60k", 0, f"{EXP}/iql_60k/figuras_iql_cartpole", "modelos.pt")
    novo("8_bc_dados", "0-2", f"{EXP}/bc_dados/figuras_bc_cartpole", "modelos_dados.pt")
    novo("9_longo_expert", 0, f"{EXP}/iql_60k_expert/figuras_iql_cartpole", "modelos.pt")
    for marco, pasta, arqs in (("6_varredura", f"{EXP}/varredura_especialista", ("varredura.csv", "varredura_curvas.csv", "numeros.json", "1_quanto_especialista.png")),
                               ("7_vento", f"{EXP}/vento", ("vento.csv", "vento_episodios.json", "numeros.json", "1_vento_constante.png", "2_rajadas.png"))):
        for a in arqs:
            if os.path.exists(os.path.join(pasta, a)):
                existe[marco] = True; guardar(marco, os.path.basename(pasta), os.path.join(pasta, a))
                if a.endswith(".csv"):
                    os.makedirs(DADOS, exist_ok=True); shutil.copyfile(os.path.join(pasta, a), os.path.join(DADOS, a))
    existe.update(dict.fromkeys(sensibilidade(), True))
    log = f"{EXP}/log_almoco.txt"
    if os.path.exists(log) and "TUDO PRONTO" in ler(log).decode("utf-8", "replace"):       # só a versão final do log
        guardar("4_seeds", "logs", log)
    log = f"{EXP}/log_extras.txt"
    if os.path.exists(log) and "TUDO PRONTO" in ler(log).decode("utf-8", "replace"):
        guardar("8_bc_dados", "logs", log)

    # resumo entre seeds: só runs com o código atual (marcos 3 e 4). Seed 0 nos dois: vale a do marco 4, da mesma máquina das outras
    grupos = {}
    for d in ics:
        if d["marco"] in ("3_melhorias", "4_seeds"):
            grupos.setdefault((d["script"], d["dataset"], d["rotulo"], "padrao"), {})[d["seed"]] = d["retorno"]
            if "perturbado" in d:
                grupos.setdefault((d["script"], d["dataset"], d["rotulo"], "perturbada"), {})[d["seed"]] = d["perturbado"]
    resumo = []
    for (script, ds, rot, av), por_seed in sorted(grupos.items()):
        v = np.array([por_seed[s] for s in sorted(por_seed)])
        resumo.append(dict(script=script, dataset=ds, rotulo=rot, avaliacao=av, n_seeds=len(v), media=float(v.mean()), dp=float(v.std(ddof=1)) if len(v) > 1 else "", minimo=float(v.min()), maximo=float(v.max()), seeds=json.dumps(sorted(por_seed)), valores=json.dumps([round(float(x), 2) for x in v])))

    # um marco por coluna, para ver a mesma métrica mudando
    pivo = {}
    for l in ev:
        if str(l["seed"]) in ("0", "0-2"):
            pivo.setdefault((l["metodo"], l["dataset"], l["metrica"]), {})[l["marco"]] = l["valor"]
    colunas = [m[0] for m in MARCOS[:4]]
    comp = [dict(metodo=k[0], dataset=k[1], metrica=k[2], **{c: v.get(c, "") for c in colunas}) for k, v in sorted(pivo.items(), key=lambda kv: tuple(map(str, kv[0])))]

    feitos = {"0_slides", "1_original", "2_reexecucao"} | set(existe)
    salvar("marcos.csv", [dict(marco=m, data=d, titulo=t, o_que_mudou=o, fonte=f, situacao="concluido" if m in feitos else "pendente") for m, d, t, o, f in MARCOS])
    salvar("evolucao.csv", ev); salvar("comparacao_marcos.csv", comp); salvar("episodios.csv", eps)
    salvar("curvas.csv", curvas); salvar("ic95.csv", ics); salvar("resumo_seeds.csv", resumo)
    vistos, unicos = set(), []
    for m in manifesto:
        if (m["arquivo"], m["md5"]) not in vistos:
            vistos.add((m["arquivo"], m["md5"])); unicos.append(m)
    salvar("manifesto.csv", unicos)
    for nome in sorted(os.listdir(DADOS)):
        print(f"  dados/{nome}: {sum(1 for _ in open(os.path.join(DADOS, nome), encoding='utf-8')) - 1} linhas")
    print("marcos:", ", ".join(f"{m[0]}={'ok' if m[0] in feitos else 'pendente'}" for m in MARCOS))


if __name__ == "__main__":
    main()
