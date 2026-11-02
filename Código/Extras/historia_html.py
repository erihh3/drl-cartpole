"""Gera Historico/historia.html a partir de Historico/HISTORIA.md: o texto só precisa ser editado em um lugar."""
import html, re

ENTRADA, SAIDA = "Historico/HISTORIA.md", "Historico/historia.html"
ESTILO = """
:root { /* ficha técnica: uma coluna de leitura; tabelas com números alinhados; as cores são as das figuras do projeto */
  --fundo:#FBFCFD; --superficie:#EEF2F6; --tinta:#14213D; --tinta2:#4C5568; --linha:#D5DBE3; --destaque:#1F7A6D; --pendente:#8A5F00;
  --display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif; --corpo:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme:dark;
  --fundo:#10161F; --superficie:#1A2330; --tinta:#E8ECF2; --tinta2:#A7B1C2; --linha:#2E3A4B; --destaque:#5CC2B2; --pendente:#E3B341; } }
:root[data-theme="dark"] { color-scheme:dark;
  --fundo:#10161F; --superficie:#1A2330; --tinta:#E8ECF2; --tinta2:#A7B1C2; --linha:#2E3A4B; --destaque:#5CC2B2; --pendente:#E3B341; }
* { box-sizing:border-box; }
body { margin:0; background:var(--fundo); color:var(--tinta); font:17px/1.6 var(--corpo); }
main { max-width:860px; margin:0 auto; padding:40px 20px 72px; display:flex; flex-direction:column; gap:14px; }
h1,h2,h3 { font-family:var(--display); line-height:1.15; text-wrap:balance; margin:0; }
h1 { font-size:40px; font-weight:600; }
h2 { font-size:27px; font-weight:600; margin-top:30px; padding-top:18px; border-top:1px solid var(--linha); }
h3 { font-size:20px; font-weight:600; margin-top:16px; }
p,ul { margin:0; max-width:68ch; }
ul { padding-left:22px; display:flex; flex-direction:column; gap:5px; }
strong { font-weight:600; }
code { font:0.88em var(--mono); background:var(--superficie); padding:1px 5px; border-radius:3px; overflow-wrap:anywhere; }
pre { margin:0; background:var(--superficie); padding:12px 14px; border-radius:4px; overflow-x:auto; }
pre code { background:none; padding:0; }
.tabela { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:15px; font-variant-numeric:tabular-nums; }
th { font-weight:600; text-align:left; border-bottom:2px solid var(--tinta2); padding:6px 12px 6px 0; vertical-align:bottom; }
td { border-bottom:1px solid var(--linha); padding:6px 12px 6px 0; vertical-align:top; }
.pendente { display:inline-block; font:600 13px var(--corpo); color:var(--pendente); border:1px solid currentColor; border-radius:3px; padding:1px 7px; margin-left:8px; vertical-align:middle; }
"""


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)


def titulo(nivel, t):
    t = inline(t)
    t = t.replace(" (pendente)", '<span class="pendente">pendente</span>')
    return f"<h{nivel}>{t}</h{nivel}>"


def converter(md):
    saida, linhas, i = [], md.split("\n"), 0
    while i < len(linhas):
        l = linhas[i]
        if l.startswith("```"):
            j = i + 1
            while not linhas[j].startswith("```"):
                j += 1
            saida.append("<pre><code>" + html.escape("\n".join(linhas[i + 1:j]), quote=False) + "</code></pre>"); i = j + 1
        elif m := re.match(r"(#{1,3}) (.*)", l):
            saida.append(titulo(len(m[1]), m[2])); i += 1
        elif l.startswith("|"):
            j = i
            while j < len(linhas) and linhas[j].startswith("|"):
                j += 1
            cel = lambda x: [c.strip() for c in x.strip().strip("|").split("|")]
            cab = "".join(f"<th>{inline(c)}</th>" for c in cel(linhas[i]))
            corpo = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cel(x)) + "</tr>" for x in linhas[i + 2:j])
            saida.append(f'<div class="tabela"><table><thead><tr>{cab}</tr></thead><tbody>{corpo}</tbody></table></div>'); i = j
        elif l.startswith("- "):
            j = i
            while j < len(linhas) and linhas[j].startswith("- "):
                j += 1
            saida.append("<ul>" + "".join(f"<li>{inline(x[2:])}</li>" for x in linhas[i:j]) + "</ul>"); i = j
        elif l.strip():
            saida.append(f"<p>{inline(l)}</p>"); i += 1
        else:
            i += 1
    return "\n".join(saida)


def main():
    with open(ENTRADA, encoding="utf-8") as f:
        md = f.read()
    nome = "Histórico das melhorias no CartPole"
    fontes = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap"
    pagina = (f'<!DOCTYPE html>\n<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
              f'<title>{nome}</title><link rel="stylesheet" href="{fontes}"><style>{ESTILO}</style></head><body><main>\n{converter(md)}\n</main></body></html>\n')
    with open(SAIDA, "w", encoding="utf-8") as f:
        f.write(pagina)
    print(f"{SAIDA}: {len(pagina) // 1024} KiB")


if __name__ == "__main__":
    main()
