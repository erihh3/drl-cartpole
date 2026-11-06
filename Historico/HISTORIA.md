# Offline RL no CartPole: o histórico das melhorias

Registro do que mudamos no projeto, do que cada mudança fez com os números e do que aprendemos com isso. Atualizado em 2026-11-04.

**Situação:** os capítulos 1 a 11 estão concluídos. O capítulo 12 (vento) ainda não foi rodado. As mudanças de código estão no branch `melhorias-avaliacao-iql`. Todos os números vêm de `Historico/dados/` ou `Historico/analises/`, com a fonte de cada um.

## O enredo

Começamos querendo corrigir detalhes da medição. Descobrimos que os próprios números andavam sozinhos entre uma rodada e outra, e que parte do que os slides contavam não se sustenta. Então repetimos tudo com cinco seeds, para separar o que é resultado do que é sorte de uma rodada. Por fim, variamos os ajustes de cada método e o tempo de treino, e a comparação entre CQL e IQL mudou duas vezes (capítulos 10 e 11). O que ficou: o IQL mantém o que aprende; o CQL pode alcançá-lo, mas precisa do α certo e da hora certa de parar, e offline não dá para conferir nenhum dos dois.

**O que resistiu às cinco seeds:**

- O DQN offline falha no expert em todas: retorno de 9,3 a 9,5, usando uma ação só na avaliação. O que muda muito é o tamanho de Q: o valor final vai de cerca de 300 a 460.000, contra um máximo possível de 100.
- Qualquer α > 0 no CQL conserta isso: 500 em todas as seeds.
- O dataset expert leva todos os métodos a 500. No ruim, o BC fica abaixo dos próprios dados em todas as seeds.
- No medio, no ruim e no misto, ao fim de 15.000 passos, o IQL supera o CQL com α = 0,5 em todas as seeds.
- No misto, o IQL faz 500 em todas as seeds, com todos os τ e β testados e até 60.000 passos.
- O CQL cai com treino longo em todo α testado. Um α mais forte só atrasa a queda (capítulo 11).

**O que caiu ou precisa mudar:**

- "O retorno cai quando τ cresce": era 212, 187 e 159 nos slides. Com uma seed, escrevemos aqui que era plano, e isso também não se sustenta. Com cinco seeds, é 209, 221 e 191: não há queda contínua, mas τ = 0,9 fica atrás de τ = 0,7 em 4 das 5 seeds e, com 60.000 passos, cai para 101 (capítulo 8).
- "O CQL fica abaixo do retorno real, o limite inferior prometido": no expert, o valor de Q reflete o tempo de treino. Aos 60.000 passos, CQL e IQL chegam a 95, como a fórmula previa (capítulos 6 e 8). No ruim, com α = 0,5, o CQL superestima: viés de +34 aos 15.000 passos e +201 aos 60.000 (capítulo 8).
- "O IQL é melhor que o CQL", sem condição: aos 15.000 passos, com α = 5, o CQL empata no medio e no misto e chega perto no ruim, 192 contra 220 (capítulo 10). Aos 60.000, o IQL volta a ficar bem à frente: 500 contra 266 no misto e 244 contra 90 no ruim (capítulo 11).
- "Só o IQL volta a 500 no misto": com α = 0,5, o CQL esteve em 500 nas cinco seeds de 3.000 a 7.500 passos, e depois caiu (capítulo 7). Com α = 5, fica em 500 até 30.000 passos, e depois cai também (capítulo 11).
- "CQL no misto: 189": uma seed só não representa esse caso. As cinco seeds vão de 123 a 466 (capítulo 7).
- "BC no ruim top 10%: 72": refeito com três seeds, dá 83, de 66 a 101. Já "misto filtrado: 500" se confirma (capítulo 9).

## A mesma medida em cinco momentos

| Medida | Slides | 1ª rodada | Rodada de 01/11 | Melhorias de 02/11 | Cinco seeds (média ± desvio) |
|---|---|---|---|---|---|
| IQL no ruim, retorno | 187 | 217 | 213 | 204 | 221 ± 15 |
| CQL no medio, retorno | 372 | 430 | 384 | 391 | 397 ± 62 |
| CQL no ruim, retorno | 126 | sem registro | 116 | 116 | 121 ± 22 |
| CQL no misto, retorno | 191 | sem registro | 189 | 187 | 260 ± 133 |
| BC no misto, retorno | 140 | 92 | 124 | 97 (regularizado) | 94 ± 15 |
| BC base no medio 10%, retorno | 385 ± 35 | 356 ± 90 | 399 ± 49 | não reexecutado | não reexecutado |
| BC no misto filtrado, retorno | 500 | sem registro | sem registro | não rodado | 500 ± 0 (3 seeds) |
| BC no ruim top 10%, retorno | 72 | sem registro | sem registro | não rodado | 83, de 66 a 101 (3 seeds) |
| DQN no expert, retorno | 9,5 | 9,3 | 9,5 | 9,5 | 9,4 ± 0,1 |
| DQN no expert, viés médio de Q | +2.737 | +19.045 | +25.667 | +25.647 | de +231 a +533.830 |
| CQL α = 0,1 no expert, viés médio | −27 | −26,7 | −26,7 | −47,1 | −47,11 ± 0,05 |
| DQN no expert, concordância com os dados | 50% | 36% | 33% | 33% | 33% numa seed, 50% nas outras |
| CQL no ruim, igual ao especialista | 73% | 66% | 65% | 65% | 71% ± 7 |
| IQL no ruim, retorno com τ = 0,5 / 0,7 / 0,9 | 212 / 187 / 159 | sem registro | 208 / 213 / 214 | 198 / 204 / 207 | 209 / 221 / 191 |

O retorno máximo é 500. A coluna "Slides" é o que as apresentações ainda mostram. A coluna "1ª rodada" vem do que o `PROJECT_GUIDE.md` registrou, porque aqueles arquivos não existem mais. A coluna "Rodada de 01/11" é o que está guardado no commit `6759143`. A coluna "Cinco seeds" usa as seeds 0 a 4; a coluna anterior é a seed 0 dela. O valor 140 do BC não bate com nenhuma figura salva. O guia supõe que veio de `--estudo dados`, mas esse estudo, refeito, dá 81 no misto. As duas linhas do BC filtrado usam o `bc_cartpole.py`, com parada antecipada.

## Capítulo 1: os números andavam sozinhos

Antes de mudarmos qualquer coisa, a mesma receita já dava números diferentes entre rodadas. Os scripts fixam as seeds, mas os resultados mudam entre máquinas e versões de bibliotecas. Com o mesmo código, seed e máquina, o resultado se repete bit a bit.

O caso mais forte é o viés do DQN: +2.737, +19.045 e +25.667. A divergência é exponencial, então o tamanho final de Q não se repete, mas a conclusão sim. O CQL no medio andou de 372 para 430 e 384.

**Depois:** o capítulo 7 mostra que, entre as duas máquinas que usamos em 02/11, os resultados foram idênticos. A máquina sozinha não explica estas diferenças.

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

No geral, este teste separa pouco os métodos. Os intervalos são de uma seed só; o capítulo 7 repete o teste com cinco.

## Capítulo 6: o que estávamos lendo errado

### Q ≈ 53 no expert mede tempo de treino, não conservadorismo

Com recompensa 1, sem término e rede-alvo com Polyak 0,005, a conta fechada dá Q(t) = 1 + 0,99 × 100 × (1 − e^(−t/20.000)). Essa fórmula não tem penalidade nenhuma, e mesmo assim acerta o valor de todos os métodos:

| Passo | 3.750 | 7.500 | 11.250 | 15.000 |
|---|---|---|---|---|
| Fórmula | 17,9 | 32,0 | 43,6 | 53,2 |
| CQL α = 0,1 | 17,4 | 31,4 | 43,2 | 52,9 |
| CQL α = 5 | 17,8 | 31,8 | 43,5 | 53,2 |
| IQL τ = 0,7 (sem penalidade) | 17,2 | 31,3 | 43,2 | 53,0 |

Pela fórmula, chegar a 95 levaria cerca de 60.000 passos, e o capítulo 8 confirma isso. Portanto o viés de −47 não sustenta a frase "o limite inferior prometido". O que o α muda de verdade é o valor da outra ação, a que não está nos dados: 7,0, −28,2, −63,6 e −72,3 para α = 0,1, 0,5, 2 e 5.

### O CQL chegou a 500 no misto e perdeu

Retorno a cada 1.500 passos, com 10 episódios por ponto:

| Método e dataset | Retorno ao longo do treino |
|---|---|
| CQL no misto | 216, **500, 500, 500, 500, 500, 500**, 246, 144, 186 |
| IQL no misto | 78, 498, 500, 500, 500, 500, 500, 500, 500, 500 |
| CQL no ruim | 186, 219, 221, 230, 171, 236, 194, 130, 118, 117 |
| IQL no ruim | 16, 160, 176, 187, 187, 184, 194, 205, 187, 198 |
| BC no misto | 66, 112, 156, 107, 58, 59, 10, 10, 10, 10 |

Sem simulador, nenhum sinal do treino avisa quando parar: é um problema central de Offline RL. A causa da queda ainda não foi estabelecida. Esta tabela é de uma seed só; o capítulo 7 mostra que a queda se repete nas cinco.

### "Os pesos escolhem as transições do especialista" é só parte da história

Depois de 5 ações sorteadas, quase nada perto do agente nos dados do misto veio do especialista. Mesmo assim o IQL concorda com ele. Medido em 50 estados de passagem (o estado em que o agente assume):

| Ações sorteadas | Especialista entre os 50 vizinhos | BC concorda | CQL concorda | IQL concorda |
|---|---|---|---|---|
| 0 | 8% | 24% | 46% | 88% |
| 5 | 0% | 14% | 54% | 98% |

Esta medida é de uma seed só. Além disso, o IQL já concorda de 86% a 89% no ruim, nas cinco seeds, e o ruim não tem nenhum episódio do especialista. Esta é uma explicação plausível, ainda não comprovada.

## Um desvio: a GPU ajuda?

Antes da rodada grande, medimos o mesmo trabalho na CPU e na GPU de uma mesma máquina: os 14 modelos do `iql_cartpole.py`, com 2.000 passos e 5 episódios de avaliação.

| Dispositivo | Processos | Tempo total |
|---|---|---|
| CPU | 8 | 38,5 s |
| GPU | 8 | 47,5 s |
| GPU | 4 | 54,7 s |
| CPU | 4 | 55,3 a 56,7 s (três repetições) |

Com 4 processos, houve empate. Com 8, a CPU foi 9 s mais rápida. Foi também a primeira execução real do `--device cuda`, e ela chegou ao fim nos dois casos.

Este teste é curto e inclui a inicialização da GPU, que só acontece uma vez, então a diferença pode diminuir em treinos longos. Mesmo assim, nada aqui indica ganho com a GPU. Por isso, a rodada grande usou a CPU, com 8 processos, e levou cerca de 40 minutos. Na GPU, 4 e 8 processos deram pesos idênticos entre si e diferentes dos da CPU, como esperado.

Por que isso faz sentido? As redes são pequenas (cerca de 67 mil parâmetros, lotes de 256). Num passo sintético do IQL contamos de 396 a 537 operações com tensores, cada uma levando em média de 20 a 26 µs numa thread de CPU, em outra máquina. A explicação mais provável, que não medimos na GPU, é que operações tão pequenas não compensam o custo fixo de disparar cada uma na GPU. Com 8 processos há ainda uma disputa: eles dividem uma GPU, enquanto na CPU cada um tem seu núcleo. A GPU também não liberou o processador: com 8 processos, o tempo de CPU foi de 3 min 20 s sem ela para 4 min 51 s com ela. E o código não foi ajustado para GPU: o IQL chama `.item()` quatro vezes por passo, e cada chamada faz a CPU esperar a GPU. Os dados não são o gargalo: eles ficam na GPU, e a cada passo só viajam os 256 índices do lote. A saída original está em `Historico/analises/teste_velocidade_20261102.txt`, e a contagem, em `Historico/analises/conta_ops_20261102.txt`.

## Capítulo 7: cinco seeds

**A pergunta:** o que continua de pé quando a seed muda? As seeds variam a inicialização e os lotes; os dados e os episódios de avaliação ficam fixos. Cada modelo é avaliado em 50 episódios.

![Retorno de BC, CQL e IQL nos quatro datasets, com um ponto por seed. O IQL fica acima do CQL no medio, no ruim e no misto, e o CQL varia muito no misto.](figuras/1_cinco_seeds.png)

| Retorno (média ± desvio entre seeds) | expert | medio | ruim | misto |
|---|---|---|---|---|
| Média do dataset | 500 | 113 | 16 | 27,8 |
| BC regularizado | 500 ± 0 | 500 ± 0 | 8,9 ± 0,0 | 93,6 ± 14,8 |
| CQL α = 0,5 | 500 ± 0 | 397,1 ± 62,4 | 120,8 ± 22,4 | 260,2 ± 133,4 |
| IQL τ = 0,7 | 500 ± 0 | 500 ± 0 | 221,2 ± 15,1 | 500 ± 0 |

**O que se repete em todas as seeds:**

- O IQL supera o CQL no medio, no ruim e no misto.
- O BC faz 500 no medio e fica abaixo dos dados no ruim.
- Qualquer α > 0 faz 500 no expert, e o DQN offline (α = 0) falha.
- Nos episódios de avaliação, o DQN usa uma ação só: sempre à esquerda dá retorno 9,26 (3 seeds), sempre à direita dá 9,50 (2 seeds).

**O que a seed muda:**

- **O tamanho de Q no DQN.** O Q final nos dados vai de 295 a 443.144, e o viés, de +231 a +533.830. "Q da ordem de 10⁴" era o que uma seed deu. O que se sustenta é que o retorno não depende desse tamanho. Já o CQL com α = 0,1 termina com Q entre 52,8 e 52,9 em todas as seeds.
- **O CQL no misto:** de 123 a 466. O 187 da seed 0 estava na parte baixa da faixa.
- **O CQL no medio:** de 306 a 478. Os 372, 430 e 384 de antes cabem nessa faixa.

### A queda do CQL se repete

![Retorno ao longo do treino. No misto e no ruim, o CQL sobe e depois cai em todas as seeds, enquanto o IQL se mantém. Com 60.000 passos no misto, o CQL não se recupera.](figuras/2_queda_do_cql.png)

No misto, as cinco seeds estão em 500 em todos os pontos de 3.000 a 7.500 passos, e todas terminam abaixo disso. A média cai de 500 para 256. No ruim, a queda vem depois dos 9.000 passos: a média vai de 261 para 122, e nas cinco seeds o último ponto fica abaixo do primeiro. Na média, o IQL não cai em nenhum dos dois.

| CQL no misto, média das 5 seeds | 7.500 | 9.000 | 10.500 | 12.000 | 13.500 | 15.000 |
|---|---|---|---|---|---|---|
| Retorno | 500 | 431 | 400 | 300 | 220 | 256 |
| Q nos dados | 28,9 | 33,7 | 38,3 | 42,3 | 46,1 | 49,9 |
| Q na outra ação | 28,8 | 33,7 | 38,3 | 42,3 | 46,2 | 50,0 |
| Concordância com os dados | 59% | 57% | 59% | 58% | 58% | 58% |
| Perda de TD | 0,52 | 0,68 | 0,74 | 0,77 | 0,91 | 0,94 |

Nada nesses sinais marca o momento da queda: Q e a perda de TD sobem sem salto desde o começo do treino. Com um simulador, pararíamos aos 7.500 passos com retorno 500. Sem ele, nenhum destes números avisaria. A causa da queda continua não estabelecida, e tudo isto vale para α = 0,5, que não foi ajustado por dataset.

### O teste perturbado com cinco seeds

As três quedas claras do capítulo 5 se repetem nas cinco seeds: BC no misto (93,6 para 30,5), IQL no ruim (221,2 para 194,3) e IQL no misto (500 para 477,3). O CQL no misto não muda na média (260,2 para 259,8). No medio, o IQL dá 490,2 em todas as seeds porque 49 episódios chegam a 500 e um deles, sempre o mesmo, acaba no passo 10.

### Duas máquinas, o mesmo resultado

A seed 0 rodou em duas máquinas diferentes, com o mesmo código: os 14 modelos saíram idênticos, pesos inclusive, e as 28 listas de retornos também. Portanto, a máquina sozinha não explica as diferenças do capítulo 1. Os arquivos antigos não registram versões de bibliotecas, então não sabemos o que explica.

## Capítulo 8: 60.000 passos

**As perguntas:** a queda do CQL persiste com mais treino? Q realmente chega perto de 100, como a fórmula prevê? Estas rodadas são de uma seed só (a seed 0): primeiro no ruim e no misto, depois no expert, para testar a fórmula.

| Modelo | 15.000 passos | 60.000 passos |
|---|---|---|
| IQL τ = 0,7 no misto | 500,0 | 500,0 |
| IQL τ = 0,7 no ruim | 203,5 | 246,7 |
| CQL α = 0,5 no misto | 187,0 | 118,5 |
| CQL α = 0,5 no ruim | 116,0 | 95,1 |
| BC no misto | 97,3 | 87,7 |
| BC no ruim | 8,9 | 8,9 |

**A queda persiste.** O CQL não se recupera: de 30.000 a 60.000 passos, ele fica entre 111 e 123 no misto. Mais treino também muda pouco os outros: o ganho do IQL no ruim, de 203,5 para 246,7, é pequeno perto do que a seed já faz com 15.000 passos (de 204 a 244).

**O viés do CQL cresce; o do IQL, não.** No ruim, aos 60.000 passos, o Q do CQL nos dados é 216, acima do máximo possível de 100:

| Viés de Q nos dados do ruim | 15.000 passos | 60.000 passos |
|---|---|---|
| CQL α = 0,5 | +34,4 | +201,4 |
| IQL τ = 0,7 | +7,5 | +7,6 |

O alvo do IQL não usa máximo sobre as ações, e o do CQL usa. Além disso, no ruim as duas ações aparecem nos dados, e seus valores de Q ficam quase iguais. Esta é uma explicação plausível, não testada.

**Q perto de 100? No expert, sim, e a fórmula acerta.** Aos 60.000 passos, CQL e IQL estão a menos de 5 do valor verdadeiro, que é 100, e seguem a fórmula do capítulo 6 o tempo todo:

![Q ao longo de 60.000 passos no expert. As curvas de CQL e IQL para a ação dos dados coincidem com a fórmula e chegam perto de 95. A curva do CQL para a outra ação cai até cerca de −186.](figuras/5_q_e_formula.png)

| No expert, seed 0 | 7.500 | 15.000 | 30.000 | 45.000 | 60.000 |
|---|---|---|---|---|---|
| Fórmula | 32,0 | 53,2 | 77,9 | 89,6 | 95,1 |
| CQL α = 0,5, ação dos dados | 31,7 | 53,0 | 78,0 | 89,8 | 95,4 |
| IQL τ = 0,7, ação dos dados | 31,3 | 53,0 | 78,1 | 89,9 | 95,5 |
| CQL α = 0,5, outra ação | −0,3 | −28,2 | −93,0 | −146,8 | −186,4 |

Isso confirma o capítulo 6: no expert, o Q da ação dos dados mede o tempo de treino, com ou sem penalidade. O que o CQL muda é a outra ação, que segue caindo. O retorno fica em 500 em todos os pontos, nos dois métodos. Os modelos do ruim, retreinados nesta rodada, saíram idênticos aos da anterior, pesos inclusive.

**Com dados misturados, a história é outra.** Nos estados do especialista dentro do misto, medidos com os pesos salvos:

| Q da ação do especialista, no misto | 15.000 passos | 60.000 passos |
|---|---|---|
| Fórmula (vale para o expert sozinho) | 53,2 | 95,1 |
| CQL α = 0,5 | 57,1 | 171,7 |
| IQL τ = 0,7 | 46,1 | 80,9 |

O IQL cresce e fica abaixo da fórmula; o CQL passa bem do máximo. Nesses mesmos estados, a concordância com o especialista anda em sentidos opostos: de 71% para 57% no CQL e de 90% para 93% no IQL.

### τ: nem queda contínua, nem plano

![IQL no ruim. Com 15.000 passos e cinco seeds, τ = 0,7 tem a maior média. Até 60.000 passos, na seed 0, o retorno com τ = 0,9 cai para cerca de 100.](figuras/3_tau.png)

| IQL no ruim | τ = 0,5 | τ = 0,7 | τ = 0,9 |
|---|---|---|---|
| Retorno nos slides | 212 | 187 | 159 |
| Retorno, 15.000 passos, cinco seeds | 208,5 ± 12,0 | 221,2 ± 15,1 | 190,9 ± 17,7 |
| Retorno, 60.000 passos, seed 0 | 225,8 | 246,7 | 101,0 |
| V, 15.000 passos, média de cinco seeds | 8,8 | 17,7 | 43,7 |
| V, 60.000 passos, seed 0 | 8,8 | 17,8 | 88,2 |

Com cinco seeds, τ = 0,7 fica acima de τ = 0,9 em 4 seeds e acima de τ = 0,5 em 4, embora as faixas se sobreponham. A queda contínua dos slides não aparece. Os 60.000 passos acrescentam outra coisa: com τ = 0,9, V segue subindo e o retorno cai, enquanto com τ = 0,5 e 0,7 o V já parou. O "otimismo demais" dos slides tem sustentação, mas aparece em τ = 0,9 e com mais treino. A parte longa é de uma seed só.

## Capítulo 9: quanto especialista basta?

**A pergunta:** o misto tem 2,4% de episódios do especialista, mas 43,8% das transições. Ele é fácil demais? A varredura treina BC, CQL e IQL em todo o ruim mais k episódios do especialista, com três seeds.

![Retorno de BC, CQL e IQL conforme aumentam os episódios do especialista somados ao ruim. O IQL chega a 500 com 25 episódios; o CQL e o BC sobem bem mais devagar.](figuras/4_quanto_especialista.png)

| Episódios do especialista | 0 | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|---|
| % das transições | 0% | 1,5% | 3,0% | 7,2% | 13,5% | 28,0% | 43,8% |
| % dos episódios | 0% | 0,05% | 0,10% | 0,25% | 0,50% | 1,2% | 2,4% |
| BC, média | 8,9 | 8,9 | 8,9 | 13,1 | 51,0 | 61,4 | 91,0 |
| CQL, média | 123 | 111 | 122 | 122 | 155 | 269 | 321 |
| CQL, pior e melhor seed | 97 a 157 | 105 a 120 | 91 a 157 | 93 a 163 | 103 a 223 | 163 a 331 | 187 a 466 |
| IQL, média | 223 | 258 | 331 | 423 | 490 | 500 | 500 |
| IQL, pior e melhor seed | 204 a 244 | 232 a 299 | 269 a 425 | 322 a 500 | 471 a 500 | 500 | 500 |

k = 0 é o ruim e k = 50 é o misto. Conferimos que o código monta exatamente esses dois datasets, array por array. Eles usam as seeds 0, 1 e 2, por isso as médias aqui diferem um pouco das de cinco seeds do capítulo 7. Nos outros níveis, a seed também sorteia quais episódios entram.

**A resposta: para o IQL, o misto é fácil demais.** Com 25 episódios ele já faz 500 nas três seeds, e com 10 (0,5% dos episódios, 13,5% das transições) faz 490. A diferença entre os métodos fica mais visível por volta de k = 5: IQL 423, CQL 122, BC 13.

- **CQL:** não sai do patamar do ruim até k = 5 e só sobe de verdade em k = 25, com muita variação entre seeds.
- **BC:** sobe a partir de k = 10 e chega só a cerca de 90 com 50 episódios. Esse ganho é frágil: com as ações iniciais sorteadas do capítulo 5, o BC faz cerca de 12 em k = 10.

### E se só houvesse os dados bons?

O `bc_cartpole.py --estudo dados` treina o BC, com parada antecipada e regularização, em partes dos dados. São três seeds, que aqui também mudam os episódios escolhidos:

![À esquerda, o retorno do BC por dataset: 500 no expert, no medio e no misto filtrado, 9 no ruim, 81 no misto e 83 no ruim top 10%. À direita, o retorno por número de episódios do especialista, sempre acima de 450.](figuras/6_dados_bc.png)

| Dados do BC | Episódios de treino | Média do dataset | Retorno do BC | Pior e melhor seed |
|---|---|---|---|---|
| ruim | 1.800 | 16,0 | 8,9 | 8,9 |
| misto | 1.845 | 27,8 | 81,2 | 77,8 a 87,7 |
| misto filtrado (só episódios que chegam a 500) | 45 | 500 | 500 | 500 |
| ruim top 10% | 197 | 29,8 | 83,3 | 65,5 a 101,4 |
| 1 episódio do especialista | 1 | 500 | 453,8 | 396,4 a 500 |
| 2 episódios do especialista | 2 | 500 | 493,9 | 483,9 a 500 |
| 4 episódios do especialista | 4 | 500 | 477,2 | 459,8 a 500 |
| 45 episódios do especialista | 45 | 500 | 500 | 500 |

**Para o BC, o problema não é falta de dados bons: é a mistura.** Um único episódio do especialista, sozinho, dá 454. Esse mesmo tipo de episódio somado a todo o ruim dá 8,9 (k = 1 na tabela anterior). A varredura usa o BC sem parada antecipada, mas isso não explica o salto: no ruim e no misto, as duas variantes ficam próximas (8,9 e 8,9; 91 e 81).

**Filtrar resolve, mas alguém precisa filtrar.** No misto, manter só os episódios que chegam a 500 devolve exatamente o expert, e o BC volta a 500. Esse filtro usa o retorno, que o BC sozinho não lê. O IQL faz uma separação parecida sem filtro nenhum: no misto, ele chega a 500 com todos os dados juntos. Já o top 10% do ruim ajuda menos, porque até os melhores episódios ali são ruidosos: o BC faz 83, acima dos 29,8 dos dados selecionados.

## Capítulo 10: a sensibilidade

**As perguntas:** os picos do CQL são reais, ou só ruído de 10 episódios? O α = 0,5 explica a derrota e a queda do CQL? E o IQL depende dos próprios ajustes? São 55 rodadas e 310 modelos, com cinco seeds, todos em GPU (`experimentos/sensibilidade_gpu/`). Os dados estão em `Historico/dados/sensibilidade.csv` e `sensibilidade_curvas.csv`.

**A resposta muda a comparação.** Com α = 0,5, o IQL ganha com folga. Com α = 5, o CQL empata com o IQL no medio e no misto, com 500 em todas as seeds, e chega perto no ruim. Antes, a diferença parecia ser de método. Boa parte dela era o valor de α.

**Depois:** o capítulo 11 mostra que esse empate vale aos 15.000 passos. Aos 60.000, o CQL cai em todo α testado.

### A GPU não muda o quadro

Os pesos em GPU são diferentes dos pesos em CPU, mas o padrão é o mesmo. A maior diferença entre as médias é de 12 pontos, bem menor que a variação entre seeds do CQL no misto:

| Mesmos ajustes, 15.000 passos | CPU | GPU |
|---|---|---|
| BC no ruim | 8,9 ± 0,0 | 8,9 ± 0,0 |
| BC no misto | 94 ± 15 | 88 ± 14 |
| CQL α = 0,5 no ruim | 121 ± 22 | 121 ± 25 |
| CQL α = 0,5 no misto | 260 ± 133 | 248 ± 141 |
| IQL τ = 0,7 no ruim | 221 ± 15 | 220 ± 12 |
| IQL τ = 0,7 no misto | 500 ± 0 | 500 ± 0 |

Como checagem, o CQL com α = 0,5 saiu idêntico nos dois scripts em todas as seeds do ruim e do misto.

### Os picos são reais

Treinar só até N passos recria exatamente o modelo daquele passo. Conferimos isso nos logs da seed 0: de 15.000 e de 60.000 passos, Q, V, concordância e retorno coincidem em todos os pontos em comum (só o `peso_awr`, média da janela de log, difere). Na GPU também: 240 retornos e 160 valores de Q coincidem, sem nenhuma diferença. Assim, cada parada pode ser medida com 50 episódios, em episódios diferentes dos usados durante o treino:

![Retorno com 50 episódios, parando em cinco pontos do treino. No ruim, o CQL fica acima do IQL até 9.000 passos e depois cai. No misto, o CQL cai a partir de 6.000 passos, e o IQL fica em 500.](figuras/7_parada.png)

| Dataset e método | 3.000 | 6.000 | 9.000 | 12.000 | 15.000 |
|---|---|---|---|---|---|
| ruim, CQL α = 0,5 | 237 ± 16 | 241 ± 54 | 255 ± 136 | 136 ± 24 | 121 ± 25 |
| ruim, IQL τ = 0,7 | 155 ± 28 | 180 ± 24 | 195 ± 13 | 211 ± 9 | 220 ± 12 |
| misto, CQL α = 0,5 | 500 ± 0 | 500 ± 1 | 429 ± 127 | 313 ± 131 | 248 ± 141 |
| misto, IQL τ = 0,7 | 498 ± 2 | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 |
| misto, BC | 208 ± 106 | 153 ± 16 | 92 ± 18 | 78 ± 12 | 88 ± 14 |

No ruim, o melhor ponto do CQL supera o melhor ponto do IQL em todas as seeds: de 239 a 464 contra 207 a 235. O ponto de 9.000 passos varia muito (de 126 a 464). Escolher o melhor de cinco pontos ainda favorece um pouco quem varia mais, mas a queda do CQL fica confirmada.

### O α muda a história

![Em cima, retorno final por α: o CQL sobe com α e alcança o IQL no medio e no misto. Embaixo, retorno durante o treino: com α = 5, o CQL não cai.](figuras/8_sensibilidade_alfa.png)

| Retorno final por α | 0 | 0,1 | 0,5 | 2 | 5 | IQL τ = 0,7 |
|---|---|---|---|---|---|---|
| medio | 412 ± 62 | 439 ± 68 | 392 ± 105 | 498 ± 5 | 500 ± 0 | 500 ± 0 (CPU) |
| misto | 174 ± 87 | 196 ± 87 | 248 ± 141 | 490 ± 17 | 500 ± 0 | 500 ± 0 |
| ruim | 117 ± 21 | 113 ± 17 | 121 ± 25 | 136 ± 11 | 192 ± 20 | 220 ± 12 |

- **Até α = 0,5, a penalidade pouco muda o resultado no ruim:** 117, 113 e 121.
- **A partir de α = 2, o retorno sobe:** no misto vai a 490 e 500, e no medio a 498 e 500. No ruim sobe até 192, com faixa de 173 a 218 contra 207 a 235 do IQL.
- **A queda some, ao menos até 15.000 passos.** Com α = 5, a média fica em 500 em todo o treino no medio e a partir de 4.500 passos no misto. No ruim, fica estável por volta de 195. Com α = 2, há uma oscilação no misto (442 aos 13.500 passos).
- **O melhor α do ruim está na borda do que testamos.** Ainda não sabemos o que acontece acima de α = 5.

### O IQL depende bem menos dos ajustes

| IQL (média ± desvio) | τ = 0,5 | τ = 0,7 | τ = 0,9 | β = 0,3 | β = 1 | β = 3 |
|---|---|---|---|---|---|---|
| ruim | 209 ± 12 (CPU) | 221 ± 15 (CPU) | 191 ± 18 (CPU) | 153 ± 23 | 220 ± 12 | 227 ± 15 |
| misto | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 |

No misto, o IQL fez 500 em todas as seeds e em todos os ajustes. No ruim, β = 0,3 derruba o retorno para 153, e β = 3 e β = 1 ficam próximos. O τ no ruim vem da rodada em CPU do capítulo 7.

Por isso, a frase mais segura agora é outra: **o CQL pode chegar onde o IQL chega, mas o resultado depende muito de α, e o do IQL depende bem menos dos ajustes.**

### O DQN só explode no expert

O α = 0 no medio, no ruim e no misto é o DQN offline sem penalidade. Ali, ele termina com Q entre 49 e 70, abaixo do máximo de 100, e com retornos de 412, 117 e 174, todos acima da média dos dados (113, 16 e 27,8). No expert, o mesmo método termina com Q de 295 a 443.144 e retorno 9. A explicação plausível é a cobertura: com ações trocadas ao acaso, as duas ações aparecem em estados parecidos, e os próprios dados corrigem as estimativas. O ruído que atrapalha o BC protege o Q-learning. O DQN também cai durante o treino (de 272 para 117 no ruim), então a penalidade não causa a queda.

**Depois:** o título desta seção só vale até 15.000 passos. Com mais treino, o DQN também explode no ruim e no misto: o ruído atrasa o problema, não o evita (capítulo 11).

### O que α muda por dentro

| Com α = 0 / 0,1 / 0,5 / 2 / 5 | ruim | misto | medio |
|---|---|---|---|
| Igual ao especialista (%) | 67 / 67 / 71 / 79 / 80 | 61 / 66 / 75 / 90 / 89 | 58 / 59 / 63 / 88 / 97 |
| Viés médio de Q nos dados | +42 / +40 / +35 / +24 / +5 | +10 / +4 / +1 / −4 / −16 | +18 / +16 / +12 / +8 / +7 |
| Q final na ação dos dados | 52 / 50 / 45 / 34 / 14 | 59 / 52 / 50 / 45 / 33 | 67 / 65 / 61 / 57 / 57 |

Calculado com os pesos salvos, nos mesmos 5.000 estados de cada dataset.

**Uma hipótese nossa caiu.** Como a penalidade do CQL é uma entropia cruzada com as ações dos dados, e no ruim a maioria delas está errada, supusemos que um α maior puxaria o CQL para a falha do BC. Até α = 5 aconteceu o contrário: a concordância com o especialista sobe de 67% para 80%, e a concordância com os dados fica entre 43% e 45%.

**Depois:** a hipótese volta a valer acima de α = 5. Com α = 50 no ruim, o CQL faz o mesmo retorno do BC (capítulo 11).

**Dá para escolher α sem jogar?** Testamos o candidato óbvio, que usa só os dados: o α com o menor viés de Q. Ele acerta no ruim (α = 5, +5) e no medio (α = 5, +7), mas erra no misto: o menor viés é o de α = 0,5 (+1), que faz 248. Saber que α = 5 é bom exigiu jogar, e isso o Offline RL não pode fazer.

**Depois:** testando α até 50, o critério também erra no ruim (capítulo 11).

### O que continua em aberto

- **O α alto cura a queda ou só a atrasa?** Com α = 5 no misto, o Q chega a 33 aos 15.000 passos, o nível em que o CQL com α = 0,5 começou a cair (34 aos 9.000). Já com α = 2 ele passa desse nível sem queda duradoura. Só um treino mais longo responde.
- **E acima de α = 5?** No ruim, o retorno ainda está subindo na borda da varredura.
- **Por que α alto ajuda?** A causa não foi isolada. O viés médio não explica tudo: no misto, α = 0,5 quase não tem viés e cai mesmo assim.
- **O teste do capítulo 9 usou α = 0,5.** Com α maior, a curva do CQL em "quanto especialista basta" pode mudar.

**Depois:** o capítulo 11 responde às duas primeiras: α alto só atrasa a queda, e acima de α = 5 o resultado piora.

## Capítulo 11: α alto e 60.000 passos

**As perguntas:** o capítulo 10 deixou duas em aberto. O α forte cura a queda do CQL ou só a atrasa? E o que acontece acima de α = 5? São 30 rodadas e 130 modelos, com cinco seeds, todos em GPU (`experimentos/sensibilidade_gpu2/`).

**As respostas:** só atrasa. E acima de α = 5 piora: com a penalidade forte demais, o CQL vira BC. A frase que resiste a tudo isso é esta: **o IQL mantém o que aprende. O CQL pode alcançá-lo, mas precisa do α certo e da hora certa de parar, e offline não dá para conferir nenhum dos dois.**

Como checagem, o treino de 60.000 passos reproduz o de 15.000 nos pontos em comum: 140 retornos e 600 valores de Q iguais, sem nenhuma diferença.

### O α forte só atrasa a queda

![Retorno até 60.000 passos. No misto, o CQL cai cada vez mais tarde conforme α cresce, enquanto o IQL fica em 500. No ruim, todos os α terminam perto de 100, e o IQL se mantém acima de 200.](figuras/10_longo_alfa.png)

| Retorno com 50 episódios | ruim, 15.000 | ruim, 60.000 | misto, 15.000 | misto, 60.000 |
|---|---|---|---|---|
| CQL α = 0 | 117 ± 21 | 103 ± 20 | 174 ± 87 | 109 ± 4 |
| CQL α = 0,5 | 121 ± 25 | 98 ± 17 | 248 ± 141 | 112 ± 9 |
| CQL α = 2 | 136 ± 11 | 106 ± 5 | 490 ± 17 | 129 ± 33 |
| CQL α = 5 | 192 ± 20 | 90 ± 15 | 500 ± 0 | 266 ± 164 |
| IQL τ = 0,7 | 220 ± 12 | 244 ± 124 | 500 ± 0 | 500 ± 0 |
| BC | 8,9 ± 0,0 | 8,9 ± 0,0 | 88 ± 14 | 64 ± 11 |

- **No misto, α maior só muda quando a queda começa.** A média fica em 500 até cerca de 7.500 passos com α = 0,5, 12.000 com α = 2 e 30.000 com α = 5. Com α = 5, o fim ainda varia muito: de 108 a 495.
- **No ruim, todos os α terminam perto de 100:** 103, 98, 106 e 90.
- **O IQL não cai.** No misto, faz 500 em todas as seeds e em todos os pontos. No ruim, a média é 244, mas com uma seed puxando (459): três das cinco terminam abaixo do que tinham aos 15.000 passos, e a mediana vai de 214 para 183. Mesmo assim, a pior seed do IQL (159) fica acima da melhor seed do CQL em qualquer α (133).

### O Q continua crescendo

| Q na ação dos dados, média | ruim, 18.000 | ruim, 30.000 | ruim, 42.000 | ruim, 60.000 | misto, 18.000 | misto, 30.000 | misto, 42.000 | misto, 60.000 |
|---|---|---|---|---|---|---|---|---|
| CQL α = 0 | 65 | 143 | 231 | 363 | 68 | 133 | 268 | 705 |
| CQL α = 0,5 | 55 | 113 | 179 | 276 | 57 | 100 | 184 | 405 |
| CQL α = 2 | 39 | 65 | 109 | 231 | 51 | 69 | 93 | 180 |
| CQL α = 5 | 15 | 16 | 19 | 24 | 38 | 53 | 65 | 99 |
| IQL τ = 0,7 | 17 | 17 | 17 | 17 | 34 | 40 | 46 | 51 |

- **O DQN também explode no ruim e no misto, só que mais tarde.** Aos 30.000 passos já passou do máximo de 100 (143 e 133), e aos 60.000 chega a 363 e 705. O retorno cai para cerca de 100, e não para 9 como no expert.
- **O IQL fica parado:** Q de 17 no ruim, com viés de +7,7 a +7,8 nas cinco seeds. O CQL com α = 0,5 tem viés de +169 a +391 no mesmo ponto.
- **O tamanho de Q não explica tudo.** Com α = 5 no ruim, Q termina em apenas 24, com viés de +15, e o retorno cai para 90 mesmo assim.
- **O que aponta para o máximo no alvo:** o IQL não usa máximo e não cai. Com τ = 0,9, que aproxima V de um máximo, o IQL cai para 101 aos 60.000 passos (capítulo 8, uma seed). É uma explicação coerente, mas a causa não foi isolada.

### Acima de α = 5, o CQL vira BC

![Retorno final por α, de 0 a 50. No ruim, o CQL sobe até α = 5 e cai até o nível do BC em α = 50. No misto, fica em 500 de α = 5 a 10 e depois cai. No medio, fica em 500.](figuras/9_alfa_ate_50.png)

| 15.000 passos | α = 5 | α = 10 | α = 20 | α = 50 | BC |
|---|---|---|---|---|---|
| ruim, retorno | 192,3 ± 19,7 | 175,2 ± 35,1 | 17,0 ± 7,6 | 8,9 ± 0,0 | 8,9 ± 0,0 |
| misto, retorno | 500 ± 0 | 500 ± 0 | 463 ± 34 | 279 ± 27 | 88 ± 14 |
| medio, retorno | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 (CPU) |
| ruim, igual ao especialista | 80% | 60% | 19% | 0,9% | |
| misto, igual ao especialista | 89% | 80% | 66% | 54% | |
| medio, igual ao especialista | 97% | 98% | 98% | 99% | |

**A hipótese do capítulo 10 estava certa, só que em outra faixa.** A penalidade do CQL é uma entropia cruzada com as ações dos dados. Quando ela domina, o Q vira um classificador das ações dos dados. No ruim, com α = 50, o CQL faz 8,9, concorda com o especialista em 0,9% dos estados e com os dados em 60%, a maioria errada. O BC faz 8,9 e concorda com o especialista em 0,6%.

Portanto, α é um botão que vai do Q-learning, em α = 0, ao BC, com α muito alto. Aos 15.000 passos, a janela boa muda com o dataset: no ruim, por volta de α = 5; no misto, de 2 a 20; no medio, de 2 em diante. Nenhum α testado alcança o IQL no ruim: o melhor é 192 contra 220.

### Escolher α sem jogar: pior do que parecia

Com α até 50, o critério do menor viés de Q erra em dois de três datasets. No ruim, escolhe α = 20 (viés de +0,5), que faz 17. No misto, escolhe α = 0,5 (+1), que faz 248. Só acerta no medio, com α = 50 (−0,4), que faz 500, onde o próprio BC já resolve.

### O que continua em aberto

- **A causa da queda** não foi isolada.
- **O medio não foi treinado até 60.000 passos.** Lá, com α de 5 a 50, o CQL está em 500 aos 15.000 passos, mas não sabemos se isso dura.
- **O τ = 0,9 aos 60.000 passos** segue com uma seed só.

## Capítulo 12: vento (pendente)

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
| `.gitignore` | ignora `experimentos/` e `CLAUDE.md` | Não |
| `Código/Extras/` (novo) | varredura, vento, coleta e figuras deste histórico | Não: só importa os scripts principais |

Não mudaram: `Código/videos_cartpole.py`, a definição de `d`, as seeds fixas e as pastas `figuras_*`. No `Código/BC/bc_cartpole.py`, só entrou o `--device`. Os `.pt` antigos continuam carregando, e os pesos são sempre salvos na CPU. Resultados treinados na GPU não são idênticos aos da CPU: todas as seeds de uma comparação devem usar o mesmo dispositivo.

## Os dados, para gerar gráficos depois

Tudo fica em `Historico/dados/`. Nada precisa ser retreinado.

| Arquivo | O que tem | Serve para |
|---|---|---|
| `sensibilidade.csv`, `sensibilidade_curvas.csv` | as 85 rodadas em GPU: paradas, α, β, τ e 60.000 passos, por seed | as figuras dos capítulos 10 e 11 |
| `marcos.csv` | cada marco, data, fonte e situação | a linha do tempo |
| `comparacao_marcos.csv` | cada métrica com uma coluna por marco | a tabela "a mesma medida em cinco momentos" |
| `evolucao.csv` | formato longo: marco, seed, método, dataset, métrica, valor, fonte | qualquer gráfico de antes e depois |
| `curvas.csv` | o log de treino de cada modelo por passo: retorno, Q, V, concordância | a subida e queda do CQL, Q contra a fórmula |
| `episodios.csv` | o retorno de cada episódio avaliado | distribuições e intervalos próprios |
| `ic95.csv` | média e IC 95% de cada modelo, padrão e perturbado | barras de erro |
| `resumo_seeds.csv` | média, desvio, mínimo e máximo entre seeds | a figura das cinco seeds |
| `varredura.csv`, `varredura_curvas.csv` | cada nível k por seed e método: retorno, perturbado e curva de treino | a figura da varredura |
| `manifesto.csv` | arquivo, tamanho e md5 de cada fonte | rastrear de onde veio cada número |

`Historico/snapshots/` guarda os `numeros.json`, os PNGs e os logs de cada marco. `Historico/analises/` guarda os scripts e as saídas das análises dos capítulos 6 a 8. `Historico/figuras/` guarda as figuras deste texto, feitas só a partir desses CSVs.

**Nada antigo é apagado:** os resultados do commit `6759143` são lidos direto do git, então promover um run para `figuras_*` não muda este histórico. Se um arquivo de snapshot muda, a versão nova entra ao lado, com o hash no nome. Os pesos em `experimentos/` não vão para o git: faça uma cópia se quiser guardá-los.

Para atualizar depois de novos experimentos, sem treinar nada:

```
python Código/Extras/coletar_historico.py
python Código/Extras/figuras_historico.py
```
