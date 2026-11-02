# Offline RL no CartPole: o histórico das melhorias

Registro do que mudamos no projeto, do que cada mudança fez com os números e do que aprendemos com isso. Atualizado em 2026-11-02.

**Situação:** os capítulos 1 a 6 estão concluídos. Os capítulos 7 a 10 dependem de experimentos ainda não rodados e não têm resultados aqui. As mudanças de código estão no branch `melhorias-avaliacao-iql`, ainda sem commit. Todos os números vêm de `Historico/dados/`, com a fonte de cada um.

## O enredo

Começamos querendo corrigir detalhes da medição. Descobrimos que os próprios números andavam sozinhos entre uma rodada e outra, e que parte do que os slides contavam não se sustenta.

**O que resistiu a todas as rodadas (ainda com uma seed só no CQL e no IQL):**

- O DQN offline diverge no expert: retorno 9,5 e valores de Q dezenas a centenas de vezes acima do máximo possível, que é 100.
- Qualquer α > 0 no CQL conserta isso: retorno 500.
- O dataset expert leva todos os métodos a 500. No ruim, o BC fica abaixo dos próprios dados.
- No ruim e no misto, ao fim de 15.000 passos, o IQL tem o maior retorno.

**O que caiu ou precisa mudar:**

- "O retorno cai quando τ cresce": era 212, 187 e 159 nos slides. Hoje é plano: 198, 204 e 207, com intervalos sobrepostos.
- "O CQL fica abaixo do retorno real, o limite inferior prometido": o valor de Q que sustentava isso reflete o tempo de treino (capítulo 6).
- "Só o IQL volta a 500 no misto": vale ao fim de 15.000 passos, mas o CQL ficou em 500 de 3.000 a 10.500 passos (capítulo 6).

## A mesma medida em quatro momentos

| Medida | Slides | 1ª rodada | Rodada de 01/11 | Melhorias de 02/11 |
|---|---|---|---|---|
| IQL no ruim, retorno | 187 | 217 | 213 | 204 |
| CQL no medio, retorno | 372 | 430 | 384 | 391 |
| CQL no ruim, retorno | 126 | sem registro | 116 | 116 |
| CQL no misto, retorno | 191 | sem registro | 189 | 187 |
| BC no misto, retorno | 140 | 92 | 124 | 97 (regularizado) |
| BC base no medio 10%, retorno | 385 ± 35 | 356 ± 90 | 399 ± 49 | não reexecutado |
| DQN no expert, retorno | 9,5 | 9,3 | 9,5 | 9,5 |
| DQN no expert, viés médio de Q | +2.737 | +19.045 | +25.667 | +25.647 |
| CQL α = 0,1 no expert, viés médio | −27 | −26,7 | −26,7 | −47,1 |
| DQN no expert, concordância com os dados | 50% | 36% | 33% | 33% |
| CQL no ruim, igual ao especialista | 73% | 66% | 65% | 65% |
| IQL no ruim, retorno com τ = 0,5 / 0,7 / 0,9 | 212 / 187 / 159 | sem registro | 208 / 213 / 214 | 198 / 204 / 207 |

O retorno máximo é 500. A coluna "Slides" é o que as apresentações ainda mostram. A coluna "1ª rodada" vem do que o `PROJECT_GUIDE.md` registrou, porque aqueles arquivos não existem mais. A coluna "Rodada de 01/11" é o que está guardado no commit `6759143`. O valor 140 do BC não bate com nenhuma figura salva; o guia supõe que veio de `--estudo dados`.

## Capítulo 1: os números andavam sozinhos

Antes de mudarmos qualquer coisa, a mesma receita já dava números diferentes entre rodadas. Os scripts fixam as seeds, mas os resultados mudam entre máquinas e versões de bibliotecas. Com o mesmo código, seed e máquina, o resultado se repete bit a bit.

O caso mais forte é o viés do DQN: +2.737, +19.045 e +25.667. A divergência é exponencial, então o tamanho final de Q não se repete, mas a conclusão sim. O CQL no medio andou de 372 para 430 e 384.

**O que isso muda:** nenhum número de um único experimento é "o" resultado. É por isso que o capítulo 7 repete tudo com cinco seeds.

## Capítulo 2: corrigindo a régua (o G dos episódios truncados)

**O problema:** o retorno descontado `G`, usado para medir o viés de Q, começava de 0 no fim de todo episódio. Mas um episódio cortado nos 500 passos não acabou: o poste continua de pé.

**A mudança:** episódio truncado agora parte de 1/(1 − γ) = 100, o mesmo valor a que convergem os alvos de TD. A definição de `d` (só término conta como fim) continua igual.

**O efeito:** só muda a medida, nunca o treino, porque `G` só entra nas figuras e no viés.

| Viés médio | Antes | Depois |
|---|---|---|
| DQN no expert | +25.667 | +25.647 |
| CQL α = 0,1 no expert | −26,7 | −47,1 |
| CQL e IQL no ruim | +34,4 e +7,5 | iguais: nenhum episódio do ruim é truncado |

## Capítulo 3: um BC justo

**O problema:** o IQL e o CQL eram comparados com um BC sem nenhuma regularização.

**A mudança:** no `iql_cartpole.py`, o BC agora usa dropout 0,2, AdamW com weight decay 1e-3 e label smoothing 0,1. A parada antecipada continua de fora.

**O efeito:** expert e medio seguem em 500, e o ruim em 9. O misto foi de 124 para 97, mas isso não mostra que regularizar piorou: o intervalo de 95% vai de 48 a 156, porque 9 episódios chegam a 500 e 41 caem por volta do passo 10. É uma seed só de um comportamento bimodal.

## Capítulo 4: mais episódios na avaliação

**A mudança:** o `--episodios` do IQL foi de 30 para 50, igual aos outros scripts.

**O efeito:** as redes de CQL e IQL são idênticas, bit a bit, às da rodada de 01/11, e os 30 primeiros episódios também. Portanto o IQL no ruim ter ido de 213 para 204 vem só dos 20 episódios a mais. Isso dá a escala do ruído de medição com 30 episódios: cerca de 10 pontos.

O mesmo vale para a varredura de τ. Com 50 episódios, os intervalos são 183 a 214, 189 a 220 e 193 a 224.

## Capítulo 5: fora da distribuição dos dados

**A mudança:** `avaliar()` ganhou `passos_aleatorios`. Com k = 5, as cinco primeiras ações de cada episódio são sorteadas, e só depois o agente assume. Todo agente recebe os mesmos estados, e esses passos contam no retorno.

| Treinado no | Método | Padrão | Perturbado | Diferença (IC 95%) |
|---|---|---|---|---|
| expert | BC | 500,0 | 483,4 | −16,6 (−40 a −3) |
| expert | CQL α = 0,5 | 500,0 | 485,4 | −14,6 (−37 a −1) |
| expert | IQL τ = 0,7 | 500,0 | 484,5 | −15,5 (−39 a −2) |
| medio | BC | 500,0 | 488,7 | −11,3 (−32 a 0) |
| medio | CQL α = 0,5 | 390,7 | 407,6 | +16,9 (−19 a 50) |
| medio | IQL τ = 0,7 | 500,0 | 490,2 | −9,8 (−29 a 0) |
| ruim | BC | 8,9 | 12,3 | +3,5 (3 a 4) |
| ruim | CQL α = 0,5 | 116,0 | 119,4 | +3,4 (−2 a 7) |
| ruim | IQL τ = 0,7 | 203,5 | 177,6 | −25,9 (−43 a −10) |
| misto | BC | 97,3 | 32,7 | −64,7 (−123 a −15) |
| misto | CQL α = 0,5 | 187,0 | 189,6 | +2,6 (−7 a 11) |
| misto | IQL τ = 0,7 | 500,0 | 476,0 | −24,0 (−50 a −4) |

**Como ler:**

- **No expert não há diferença entre métodos.** A própria heurística que gerou os dados faz 485,4 nesse teste. A queda de cerca de 15 pontos é o custo da tarefa, não falta de robustez.
- **As quedas claras são BC no misto, IQL no ruim e IQL no misto.** A do BC é a maior.
- **O ganho do BC no ruim é um artefato:** os passos sorteados contam no retorno, e os 50 episódios continuam caindo cedo.

No geral, este teste separa pouco os métodos. Os intervalos são de uma seed só.

## Capítulo 6: o que estávamos lendo errado

### Q ≈ 53 no expert mede tempo de treino, não conservadorismo

Com recompensa 1, sem término e rede-alvo com Polyak 0,005, a conta fechada dá Q(t) = 1 + 0,99 × 100 × (1 − e^(−t/20.000)). Essa fórmula não tem penalidade nenhuma, e mesmo assim acerta o valor de todos os métodos:

| Passo | 3.750 | 7.500 | 11.250 | 15.000 |
|---|---|---|---|---|
| Fórmula | 17,9 | 32,0 | 43,6 | 53,2 |
| CQL α = 0,1 | 17,4 | 31,4 | 43,2 | 52,9 |
| CQL α = 5 | 17,8 | 31,8 | 43,5 | 53,2 |
| IQL τ = 0,7 (sem penalidade) | 17,2 | 31,3 | 43,2 | 53,0 |

Pela fórmula, chegar a 95 levaria cerca de 60.000 passos. Portanto o viés de −47 não sustenta a frase "o limite inferior prometido". O que o α muda de verdade é o valor da outra ação, a que não está nos dados: 7,0, −28,2, −63,6 e −72,3 para α = 0,1, 0,5, 2 e 5.

### O CQL chegou a 500 no misto e perdeu

Retorno a cada 1.500 passos, com 10 episódios por ponto:

| Método e dataset | Retorno ao longo do treino |
|---|---|
| CQL no misto | 216, **500, 500, 500, 500, 500, 500**, 246, 144, 186 |
| IQL no misto | 78, 498, 500, 500, 500, 500, 500, 500, 500, 500 |
| CQL no ruim | 186, 219, 221, 230, 171, 236, 194, 130, 118, 117 |
| IQL no ruim | 16, 160, 176, 187, 187, 184, 194, 205, 187, 198 |
| BC no misto | 66, 112, 156, 107, 58, 59, 10, 10, 10, 10 |

Sem simulador, nenhum sinal do treino avisa quando parar: é um problema central de Offline RL. A causa da queda ainda não foi estabelecida, e isto é uma seed só. O capítulo 7 vai dizer se ela se repete.

### "Os pesos escolhem as transições do especialista" é só parte da história

Depois de 5 ações sorteadas, quase nada perto do agente nos dados do misto veio do especialista. Mesmo assim o IQL concorda com ele. Medido em 50 estados de passagem (o estado em que o agente assume):

| Ações sorteadas | Especialista entre os 50 vizinhos | BC concorda | CQL concorda | IQL concorda |
|---|---|---|---|---|
| 0 | 8% | 24% | 46% | 88% |
| 5 | 0% | 14% | 54% | 98% |

Além disso, o IQL já concorda em 86% no ruim, que não tem nenhum episódio do especialista. Esta é uma explicação plausível, ainda não comprovada.

## Capítulo 7: cinco seeds (pendente)

**A pergunta:** o que continua de pé quando a seed muda? As seeds variam a inicialização e os lotes; os dados e os episódios de avaliação ficam fixos. Os casos em aberto são o ruim, o misto e o CQL no medio, além da queda do CQL.

## Capítulo 8: 60.000 passos (pendente)

**A pergunta:** a queda do CQL persiste com mais treino? Q realmente chega perto de 100, como a fórmula prevê?

## Capítulo 9: quanto especialista basta? (pendente)

**A pergunta:** o misto tem 2,4% de episódios do especialista, mas 43,8% das transições. Ele é fácil demais? A varredura treina BC, CQL e IQL em todo o ruim mais k episódios do especialista, com três seeds:

| Episódios do especialista | 0 | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|---|
| % das transições | 0% | 1,5% | 3,0% | 7,2% | 13,5% | 28,0% | 43,8% |
| % dos episódios | 0% | 0,05% | 0,10% | 0,25% | 0,50% | 1,2% | 2,4% |

k = 0 é o ruim e k = 50 é o misto. Conferimos que o código monta exatamente esses dois datasets, array por array.

## Capítulo 10: vento (pendente)

**A pergunta:** e se o mundo mudar depois da coleta? As políticas prontas são avaliadas com vento, sem retreino. Nenhum destes métodos trata mudança de dinâmica, então todos podem cair juntos: é um tema para limites e trabalhos futuros.

Por enquanto só temos a referência, a heurística que gerou os dados. O vento constante a tira do trilho, e as rajadas fortes derrubam o poste:

| Vento | Retorno da heurística |
|---|---|
| Constante, 0 e 2 N | 500 |
| Constante, 3 / 4 / 5 N | 445 / 351 / 252 |
| Rajadas, σ = 5 / 10 / 15 N | 500 / 456,5 / 137,5 |

Conferimos o script: com vento zero, ele reproduz exatamente os 50 retornos salvos dos 12 modelos da seed 0.

## O que mudou no código

| Arquivo | Mudança | Muda o treino? |
|---|---|---|
| `Código/*/cartpole_comum.py` (3 cópias) | `G` de episódio truncado parte de 100 | Não |
| `Código/*/cartpole_comum.py` | `avaliar(..., passos_aleatorios=0)` | Não |
| `Código/IQL/iql_cartpole.py` | BC regularizado | Só o do BC |
| `Código/IQL/iql_cartpole.py` | `--episodios` de 30 para 50 | Não |
| `Código/IQL/iql_cartpole.py` | teste perturbado em `numeros.json` e `5_resultado.png` | Não |
| `Código/CQL/cql_cartpole.py` | rótulo do eixo da figura 1 | Não |
| Scripts de treino e `Código/*/cartpole_comum.py` | `--device` (padrão `cpu`), para treinar também em `cuda`; a avaliação fica sempre na CPU | Na CPU, não: resultados idênticos bit a bit |
| `.gitignore` | ignora `experimentos/` | Não |
| `Código/Extras/` (novo) | varredura, vento e coleta deste histórico | Não: só importa os scripts principais |

Não mudaram: `Código/videos_cartpole.py`, a definição de `d`, as seeds fixas e as pastas `figuras_*`. No `Código/BC/bc_cartpole.py`, só entrou o `--device`. Os `.pt` antigos continuam carregando, e os pesos são sempre salvos na CPU. Resultados treinados na GPU não são idênticos aos da CPU: todas as seeds de uma comparação devem usar o mesmo dispositivo.

## Os dados, para gerar gráficos depois

Tudo fica em `Historico/dados/`. Nada precisa ser retreinado.

| Arquivo | O que tem | Serve para |
|---|---|---|
| `marcos.csv` | cada marco, data, fonte e situação | a linha do tempo |
| `comparacao_marcos.csv` | cada métrica com uma coluna por marco | a tabela "a mesma medida em quatro momentos" |
| `evolucao.csv` | formato longo: marco, seed, método, dataset, métrica, valor, fonte | qualquer gráfico de antes e depois |
| `curvas.csv` | o log de treino de cada modelo por passo: retorno, Q, V, concordância | a subida e queda do CQL, Q contra a fórmula |
| `episodios.csv` | o retorno de cada episódio avaliado | distribuições e intervalos próprios |
| `ic95.csv` | média e IC 95% de cada modelo, padrão e perturbado | barras de erro |
| `resumo_seeds.csv` | média, desvio, mínimo e máximo entre seeds | a figura multi-seed |
| `manifesto.csv` | arquivo, tamanho e md5 de cada fonte | rastrear de onde veio cada número |

`Historico/snapshots/` guarda os `numeros.json`, os PNGs e os logs de cada marco. `Historico/analises/` guarda os scripts e as saídas das análises do capítulo 6.

**Nada antigo é apagado:** os resultados do commit `6759143` são lidos direto do git, então promover um run para `figuras_*` não muda este histórico. Se um arquivo de snapshot muda, a versão nova entra ao lado, com o hash no nome. Os pesos em `experimentos/` não vão para o git: faça uma cópia se quiser guardá-los.

Para atualizar depois de novos experimentos, sem treinar nada:

```
python Código/Extras/coletar_historico.py
```
