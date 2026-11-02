# Offline RL on CartPole: project guide

A walkthrough of the code, the folder layout, how to run everything, and what the results mean.
The code and slides are in Portuguese, so Portuguese names are kept and translated where they first appear.

---

## 1. What this project is in one paragraph

We train agents for **CartPole-v1** (Gymnasium) **without ever interacting with the environment during training**. That is **offline RL**: the agent only reads a fixed dataset of transitions `(s, a, r, s', d)` recorded earlier by another policy `πβ`. The environment appears only at the end, to measure the return.

We generate four datasets of different quality from a one-line "expert" heuristic and compare three methods:

| Method | Script | Core idea | Uses reward? |
|---|---|---|---|
| **BC** (Behavior Cloning) | `Código/BC/bc_cartpole.py` | Supervised classification: copy the dataset's action | No |
| **CQL** (Conservative Q-Learning) | `Código/CQL/cql_cartpole.py` | Q-learning plus a penalty that pushes down Q of actions **not** in the data | Yes |
| **IQL** (Implicit Q-Learning) | `Código/IQL/iql_cartpole.py` | Never queries Q of unseen actions; uses an expectile to get a "near-max" over dataset actions, then advantage-weighted BC | Yes |

**Takeaway**
- With good data, all three methods reach 500/500.
- With bad or mixed data, BC copies the mistakes.
- Naive offline Q-learning (DQN) explodes because of extrapolation error.
- CQL and IQL fix this in two different ways. IQL is the most robust on the noisy datasets.

---

## 2. Folder structure

```
LocalOffLine-RL_v2/
├── anottations_final_project.md      # course requirements for the 30-min presentation
├── PROJECT_GUIDE.md                  # this file
│
├── Código/                           # "Code"
│   ├── BC/
│   │   ├── bc_cartpole.py            # Behavior Cloning + 4 variants (early stop, regularization, memory)
│   │   └── cartpole_comum.py         # shared module (identical copy)
│   ├── CQL/
│   │   ├── cql_cartpole.py           # offline DQN (α=0) vs CQL, α sweep
│   │   └── cartpole_comum.py         # identical copy
│   └── IQL/
│       ├── iql_cartpole.py           # IQL vs CQL vs BC on 4 datasets, τ sweep
│       └── cartpole_comum.py         # identical copy
│
├── figuras_bc_cartpole/              # "figures": ORIGINAL results (saved models .pt + PNGs + numeros.json)
├── figuras_cql_cartpole/
├── figuras_iql_cartpole/
│
├── Output/                           # results of the fresh re-run (see section 9)
│   ├── figuras_{bc,cql,iql}_cartpole/
│   └── log_{bc,cql,iql}.txt
│
├── Apresentações/                    # "Presentations" (.pptx)
│   ├── Offline Reinforcement Learning.pptx   # intro: online vs offline RL, the central challenge
│   ├── Sobre o dataset_CartPole.pptx         # environment, expert heuristic, datasets
│   ├── Behavior_Cloning_CartPole.pptx        # BC derivation + results
│   ├── Conservative_Q-Learning_CartPole.pptx # CQL derivation + results
│   └── Implicit Q-Learning_CartPole.pptx     # IQL derivation + results
│
├── Dataset/  Modelos/                # EMPTY, not used by the code (see note below)
```

Notes:
- **`cartpole_comum.py` is copied three times, and the copies are byte-identical.** Each script does `import cartpole_comum as C`. Python finds the module because it sits next to the script. If you change one copy, change all three.
- **Datasets do not live in `Dataset/`.** Minari stores them in its own cache: `~/.minari/datasets/cartpole/{expert,medio,ruim}-v0/` (HDF5). The `misto` dataset is never saved; it is built in memory.
- `numeros.json` ("numbers") in each figure folder holds the key metrics behind the figures. Use it as the source of truth for numbers on slides.
- `modelos*.pt` ("models") holds the trained weights and training logs. `--so-graficos` ("only plots") regenerates the figures from it without retraining.

---

## 3. How to run

### 3.1 Setup (once)

```bash
python -m venv .venv && source .venv/bin/activate        # or use uv / conda
pip install torch gymnasium matplotlib "minari[create,hdf5]"
```

> ⚠️ The scripts' docstrings say `pip install "minari[create]"`. With current Minari (0.5.x) that is **not enough**: dataset creation fails with `ImportError: h5py is not installed`. You also need the `hdf5` extra, as shown above.

No GPU is needed. Everything runs on CPU, and each process uses one thread (`torch.set_num_threads(1)`). Parallelism comes from `ProcessPoolExecutor` (`--workers`).

### 3.2 Run (from the project root)

`--pasta` (output folder) is **relative to the current directory**. Run from the project root so the output lands in `figuras_*/` like the original results.

```bash
python Código/BC/bc_cartpole.py      # ~ BC: 4 variants × 3 seeds on "medio 10%"
python Código/CQL/cql_cartpole.py    # ~ CQL: α ∈ {0, 0.1, 0.5, 2, 5} on "expert"
python Código/IQL/iql_cartpole.py    # ~ IQL vs CQL vs BC on 4 datasets + τ sweep on "ruim"
```

The first run of any script generates the Minari datasets (~30 s). After that they are reused.

Useful flags:

| Flag | Script | Meaning |
|---|---|---|
| `--so-graficos` | all | Skip training and re-plot from the saved `.pt` |
| `--pasta DIR` | all | Output folder (default `figuras_<method>_cartpole`) |
| `--workers N` | all | Parallel processes |
| `--mostrar` | all | Open the figures in a window as well |
| `--passos N` | all | Training steps (BC max 20k; CQL/IQL 15k) |
| `--estudo dados\|todos` | BC | Also run the data-quality/quantity study (figure 6) |
| `--seeds 0 1 2` / `--seed 0` | BC / CQL, IQL | Random seeds |
| `--dataset {expert,medio,ruim,misto}` | BC, CQL | Which dataset |
| `--alfas …` | CQL | α values to sweep |
| `--datasets …`, `--analise ruim`, `--tau`, `--beta`, `--alpha`, `--taus-varredura` | IQL | Datasets, analysis dataset, IQL/CQL hyperparameters |

Quick versions: `python Código/BC/bc_cartpole.py --seeds 0 --passos 5000` and `python Código/IQL/iql_cartpole.py --datasets ruim misto`.

**The code never uses the GPU.** Nothing is moved to `cuda`. It is CPU-only by design: tiny MLPs, and CPU evaluation in the simulator. More CPU cores means faster runs. Runtimes are in section 9.

**You don't need to retrain to present.** `--so-graficos` re-plots the existing `figuras_*` results in seconds.

---

## 4. The problem: CartPole and the datasets (`cartpole_comum.py`)

### 4.1 Environment
- **State** `s ∈ ℝ⁴`: cart position `x`, cart velocity `v`, pole angle `θ`, angular velocity `ω`. It is fully observable.
- **Actions**: `0 = esquerda` (left), `1 = direita` (right). There is no "do nothing".
- **Reward**: +1 per step, up to **500** steps, so **return = number of steps survived**.
- **Termination vs. truncation**:
  - The episode *terminates* when `|θ| > 12°` or `|x| > 2.4`. Then `d = 1` and the future is zeroed in the TD target.
  - It is *truncated* at 500 steps. Then `d = 0` because the pole was still up.
  - The code stores only `terminations` in `d`. This matters: treating truncation as death would teach the agent that balancing for 500 steps leads to death.

### 4.2 The behaviour policy `πβ` (the "expert")
```python
def acao_especialista(obs):            # "expert action"
    x, v, theta, omega = obs
    return int(theta + 0.5 * omega + 0.01 * x + 0.1 * v > 0)   # 1 = push right
```
It pushes toward the side where the pole is falling. On its own it scores 500/500.

### 4.3 The four datasets
Each dataset is the expert with each action **flipped with probability p** (`rng` seed 0, env seeds `100000 + episode`):

| Name ("expert/medium/bad/mixed") | Flip p | Episodes | Transitions | Avg. return |
|---|---|---|---|---|
| `expert` | 0% | 50 | 25,000 | 500.0 |
| `medio` | 30% | 300 | 33,895 | 113.0 |
| `ruim` | 60% | 2,000 | 32,074 | 16.0 |
| `misto` = expert + ruim | n/a | 2,050 | 57,074 | 27.8 |

Two facts drive most of the results:
- At **30%** flips, the correct action is still the **majority** in each state. BC's argmax therefore recovers the expert.
- At **60%** flips, the **wrong** action is the majority. BC learns to do the wrong thing. Only methods that look at the reward (CQL, IQL) can do better than the data.
- In `expert`, each state has only **one** action, because the expert is deterministic. The other action's Q is never trained by any data point. That is the classic **extrapolation error** setup for offline Q-learning.

### 4.4 Function reference (`cartpole_comum.py`)

| Function / constant | What it does |
|---|---|
| `ENV_ID, GAMMA=0.99, N_ACOES=2, DIM_OBS=4` | Constants |
| `DATASETS` | name → (Minari id, flip probability, number of episodes) |
| `garantir_datasets()` ("ensure datasets") | Creates the Minari datasets if missing, using `minari.DataCollector` |
| `carregar(nome)` ("load") | Loads a dataset into flat numpy arrays (see below). Builds `misto` by concatenating expert and ruim |
| `resumo_dataset` | One-line summary (transitions, episodes, mean return) |
| `mlp(entrada, saida, h=256, dropout)` | The network used everywhere: `in → 256 → ReLU → 256 → ReLU → out`, with optional dropout |
| `Agente` | Greedy policy: `argmax net(obs)`. Works for both policy logits and Q-values |
| `avaliar(agente, n, seed)` ("evaluate") | Runs `n` online episodes (seeds `seed + k`). Returns mean return, all returns, action counts |
| `rodar_episodio` ("run episode") | One episode; returns the visited states (used for the θ-over-time plots) |
| `mapa_politica` / `desenhar_mapa` | **Policy map**: the action chosen on a θ × ω grid (cart at rest at the centre), plotted with the expert boundary `θ + 0.5ω = 0` and dataset states |
| `estilo_graficos` / `utf8_no_console` | Matplotlib style; UTF-8 fix for the Windows console |

`carregar()` returns a dict of per-transition arrays:

| Key | Meaning |
|---|---|
| `s, a, r, s2, d` | state, action, reward, next state, terminal flag (terminations only) |
| `G` | discounted return-to-go from this step (γ = 0.99). Used to check whether Q is calibrated |
| `ep` | episode index |
| `ini` | index of the first transition of this transition's episode (used for history stacking) |
| `ret_ep` | total return of the episode this transition belongs to |
| `sucesso` | episode reached 500 steps (truncated) |

---

## 5. Behavior Cloning: `Código/BC/bc_cartpole.py`

**Idea.** Treat the dataset as classification. The input is `s`, the label is the recorded `a`, and the loss is cross-entropy. Minimizing it equals minimizing `KL(πβ ‖ πθ)` on dataset states, so BC copies `πβ`, mistakes included. The reward is never used.

**Default experiment ("melhorias" = improvements).** The dataset is **`medio`, 10% of the training episodes** (≈ 27 episodes). With the full expert dataset, plain BC already scores 500, so there is nothing left to improve. Few, noisy labels make overfitting visible. Each variant adds to the previous one, over 3 seeds:

| Variant | Config |
|---|---|
| `base` | MLP, Adam (lr 1e-3), keeps the network from the **last** step (20k steps) |
| `+ parada` (early stopping) | Validation loss every 500 steps; keep the **best** network; stop after 5k steps without improvement |
| `+ regularização` | `+ parada` plus dropout 0.2, AdamW weight decay 1e-3, label smoothing 0.1 |
| `+ memória` (memory) | `+ regularização` plus input = last k=4 observations + 3 previous actions (one-hot) → 22 inputs |

CartPole is fully observable, so memory **should not** help. That variant shows the point; in MiniGrid, which has partial observability, memory would help.

**Code flow**
1. `main()` builds a list of jobs: (variant × seed).
2. `ProcessPoolExecutor.map(treinar, jobs)` runs them in parallel.
3. `treinar(job)` ("train") does the following:
   - `selecionar` splits episodes: 10% for validation (split **by episode**, so there is no leakage between correlated steps), then keeps `fracao` of the rest.
   - `codificar_hist` builds the inputs, stacking history when k > 1.
   - Loop: batch of 256 → cross-entropy → step.
   - Every 500 steps, log validation loss, accuracy against the **dataset** labels (noisy), and accuracy against the **expert** (noise-free), and track the best checkpoint.
   - For the first seed, a 10-episode return curve every `passos/10` steps.
   - Final evaluation on 50 episodes.
4. Results, including weights, are saved to `modelos_melhorias.pt`. `gerar_figuras` writes the PNGs and `numeros.json`.

**Figures**

| File | Shows |
|---|---|
| `1_parada.png` | Training vs validation loss of `base` (validation loss rises from step 500 = memorizing noise) and the return dropping during training |
| `2_regularizacao.png` | Policy maps: `base` grows "islands" off the expert boundary (memorized noise). Validation loss per variant |
| `3_memoria.png` | Accuracy against the dataset vs against the expert: memory brings nothing new |
| `4_comparacao.png` | Final return per variant (bars = mean ± std, dots = seeds) and the step of the saved network |
| `5_resultado.png` | Pole angle over 3 episodes: `base` vs the best variant |
| `6_dados.png` | *(only with `--estudo dados/todos`)* BC return vs data quality and vs number of expert episodes |

---

## 6. Conservative Q-Learning: `Código/CQL/cql_cartpole.py`

**Problem it solves.** Offline Q-learning uses the target `y = r + γ max_a' Q(s', a')`. The `max` can select an action that **never appears in the data**. Its Q is pure extrapolation, it is never corrected, and the error is bootstrapped through the whole value function. On `expert`, where each state has one action, offline DQN's Q-values **explode** into the thousands, while the real return-to-go is at most 100 (= 1/(1−γ)).

**CQL loss (discrete, CQL(H))**, exactly as in the code:
```python
y      = r + γ (1 - d) Q_target(s', argmax_a' Q(s', a'))          # Double DQN target
perda  = MSE(Q(s,a), y) + α * ( logsumexp_a Q(s,·) − Q(s, a_data) ).mean()
```
- `logsumexp` is a soft max over actions. Minimizing `logsumexp − Q(s, a_data)` pushes **down** all actions, softmax-weighted, and pushes **up** the dataset action.
- With 2 actions, the penalty equals `softplus(−Δ)`, where `Δ = Q(s, a_data) − Q(s, other)`. Its gradient is BC's cross-entropy gradient applied to the Q-values.
- `α = 0` gives plain **offline Double DQN**, the baseline that fails.

**Training details.** Batch 256, Adam 3e-4, 15k steps, Polyak target update `θ_t ← 0.995 θ_t + 0.005 θ`, γ = 0.99, seed 0.

**Default experiment.** Dataset `expert`, α ∈ {0, 0.1, 0.5, 2, 5}. For the figures, "CQL" means the best α > 0 (ties → the smallest α).

**Logged during training.** `q_dados` (Q of the dataset action), `q_outra` (Q of the other action), `gap` (|Q_left − Q_right|), `concordancia` (agreement of argmax Q with the dataset action), TD loss, penalty, and the return every 1.5k steps.

**Figures**

| File | Shows |
|---|---|
| `1_superestimacao.png` | DQN's Q grows exponentially (log scale) past the real return; CQL stays below it (a **lower bound**, as the paper guarantees). Predicted max Q vs real G |
| `2_mecanismo.png` | One concrete state: Q bars, logsumexp line, penalty value; TD loss vs penalty over training |
| `3_gap.png` | CQL separates the two actions; agreement with the dataset goes to ~100% |
| `4_alfa.png` | α sweep: return, final Q values, return curves |
| `5_resultado.png` | Pole angle over one episode, plus the policy maps of DQN vs CQL |

---

## 7. Implicit Q-Learning: `Código/IQL/iql_cartpole.py`

**Idea.** Avoid evaluating Q on unseen actions **at all**. Every loss uses only `(s, a)` pairs from the dataset. Three networks (plus two target Q's):

```python
# 1) V(s): EXPECTILE τ of Q_target(s, a_data)  →  a "near-max" over the actions that appear in the data
u       = min(Q1_t, Q2_t)(s)[a] − V(s)
perda_v = mean(|τ − 1(u<0)| · u²)                # asymmetric L2; τ=0.5 → mean, τ→1 → max

# 2) Q(s,a) ← r + γ(1−d) V(s')                    # no max over actions; the "max" is inside V
perda_q = (MSE(Q1(s)[a], y) + MSE(Q2(s)[a], y)) / 2

# 3) policy: advantage-weighted regression (weighted BC)
w        = clamp(exp(β · (Q_t(s,a) − V(s))), max=100)
perda_pi = mean(w · CE(π(s), a))
```
- **τ** (expectile) slides between SARSA (τ = 0.5, value of `πβ`) and Q-learning restricted to the dataset's support (τ → 1).
- **β** controls how much to trust the advantage. β = 0 is plain BC. Here β = 1, smaller than in MiniGrid, because Q ~ 100 in CartPole gives large advantages.
- The policy is the closed-form solution `π* ∝ πβ · exp(β A)`, projected onto a network. Actions never seen keep zero probability.

**Default experiment.** For each of the 4 datasets: IQL (τ = 0.7), CQL (α = 0.5), BC (plain, 15k steps, **no** early stopping). Plus a τ sweep {0.5, 0.7, 0.9} on the analysis dataset `ruim`. That makes 14 models, seed 0, 30 evaluation episodes each.

**Figures** (analysis dataset `ruim`)

| File | Shows |
|---|---|
| `1_acoes_sem_dados.png` | Q over training for CQL vs IQL; calibration of Q(s, a_data) against real G |
| `2_mecanismo_iql.png` | One state where the data took the wrong action: Q, V, advantage; BC vs IQL action probabilities; histogram of AWR weights for right vs wrong dataset actions |
| `3_politicas.png` | Agreement with the dataset over training; policy maps for BC / CQL / IQL |
| `4_tau.png` | τ sweep: return, V and Q vs τ, return curves vs CQL and BC |
| `5_resultado.png` | **The main results chart**: return per dataset for BC / CQL / IQL, plus one episode on `ruim` |

---

## 8. How the pieces fit together

```
                 acao_especialista (heuristic) + p% action flips
                                    │
                 garantir_datasets → Minari cache (~/.minari, HDF5)
                                    │
                 carregar(nome) → dict of numpy arrays (s, a, r, s2, d, G, ep, ini, …)
                                    │
     ┌──────────────────────────────┼───────────────────────────────┐
 bc_cartpole.py               cql_cartpole.py                 iql_cartpole.py
 (classification)             (TD + conservative penalty)     (expectile V, TD Q, AWR π)
     │                              │                               │
     └─────── ProcessPoolExecutor(treinar, jobs): one process per model ──────┘
                                    │
       avaliar(): online evaluation in CartPole-v1 (the ONLY env contact)
                                    │
       torch.save(results) → modelos*.pt   →   gerar_figuras → PNGs + numeros.json
```

Shared conventions across the three scripts:
- Same MLP (4 → 256 → 256 → out), batch 256, γ = 0.99.
- `log_cada` / `aval_cada` intervals: metrics every 1/40 of training, return every 1/10 (10 episodes, seed 50 000).
- Final evaluation seeds start at 10 000, so they differ from the dataset seeds (100 000+) and the training-curve seeds (50 000).
- Each `treinar` returns a dict with `log`, `retorno`, `retornos`, `acoes`, `pesos`. That dict is what goes into the `.pt` file.

---

## 9. Results of the fresh re-run (2026-11-01)

Setup: 8-core CPU, no GPU, Python 3.12, torch 2.15 (CPU), Minari 0.5.4, three scripts running at the same time. BC and CQL finished and are in `Output/`. IQL was stopped partway: 6 of 14 models done, no figures yet.

**BC (`medio` 10%, 3 seeds): return, mean ± std**

| Variant | Slides | Original `numeros.json` | Re-run |
|---|---|---|---|
| base | 385 ± 35 | 356 ± 90 | 399 ± 49 |
| + parada | 500 | 500 ± 0 | 500 ± 0 |
| + regularização | 500 | 500 ± 0 | 500 ± 0 |
| + memória | 500 | 500 ± 0 | 500 ± 0 |

The improved variants reproduce to three decimals (validation loss and accuracy). Only `base` moves, because a network that has memorized noise is fragile.

**CQL (`expert`, seed 0)**

| Metric | Slides | Original JSON | Re-run |
|---|---|---|---|
| Return α = 0 (DQN) / α ≥ 0.1 | 9.5 / 500 | 9.3 / 500 | 9.5 / 500 |
| DQN mean bias of max Q | +2,737 | +19,045 | +25,667 |
| CQL mean bias | −27 | −26.7 | −26.7 |
| Agreement with dataset, DQN / CQL | 50% / 99.6% | 36% / 99.8% | 33% / 99.8% |

DQN's divergence is exponential, so its final Q magnitude changes from run to run. The conclusion does not: it ends 100–250× above the maximum possible return of 100.

**IQL (partial)**: expert: IQL 500, CQL 500, BC 500 · medio: IQL 500, BC 500, CQL 384.5 (430 in the original).

**Runtime** (8 cores, all three scripts in parallel):

| Script | Models | Time per model | Wall time |
|---|---|---|---|
| BC | 12 | base ~150 s, others 45–75 s | ~6 min |
| CQL | 5 | 100–175 s | ~8 min |
| IQL | 14 | IQL 350–450 s, CQL ~170 s, BC 70–150 s | ~20–25 min (estimate) |

---

## 10. Things to know before presenting

1. **Slides vs. figures.** Several slide numbers come from an earlier run than the PNGs in `figuras_*`:
   - CQL slides 4, 11–13: Q ≈ 2,700, bias +2,737, 50% agreement. The figure says +19,045 and 36%.
   - BC slides 12 and 16: 385 ± 35. The figure says 356 ± 90.
   - IQL slide 19: CQL 73% vs 66% in the JSON.
   - IQL slide 20: IQL on `ruim` 187. The figure shows 217.

   Quote the numbers printed on the figures you show, or say "order of 10⁴" for the DQN.
2. **Slide 6 of `Offline Reinforcement Learning.pptx`** talks about **FourRooms** (MiniGrid). That's left over from an earlier version. Replace it with the CartPole DQN numbers.
3. **The `Sobre o dataset` deck's footer says "x / 19"**, but the deck has 7 slides.
4. **Seeds.** CQL and IQL use **one seed**, so there are no error bars; BC uses 3. Expect a question on this.
5. **BC inside `iql_cartpole.py`** is plain BC (no early stopping), so it is not the same model as in `bc_cartpole.py`. The BC slide 15 value "misto 140" matches neither the IQL figure (92) nor any figure present. It probably came from `--estudo dados`, which has no saved figure.
6. **Course notes mention PettingZoo/BenchMARL.** The project is single-agent: Gymnasium + Minari. Be ready to say why.

---

## 11. Videos for the slides (`Código/videos_cartpole.py`)

The script replays the **already-trained** models from the `figuras_*/*.pt` files. It does no training and takes about 1–2 min on CPU. Agents run side by side from the same initial state, and each panel freezes when its episode ends, labelled "caiu" (fell) or "saiu do trilho" (left the track).

```bash
pip install "gymnasium[classic-control]" imageio imageio-ffmpeg
python Código/videos_cartpole.py          # MP4s in videos_cartpole/
python Código/videos_cartpole.py --gif    # also GIFs
```

| Video | Deck |
|---|---|
| `1_politicas_dos_dados.mp4`: expert / medio / ruim behaviour policies (500 / 233 / 16 steps) | Sobre o dataset |
| `2_bc_parada.mp4`: BC base (falls at 315) vs + parada (500) | BC |
| `3_cql_dqn.mp4`: offline DQN (falls at 9) vs CQL (500) | CQL / intro slide 6 |
| `4_iql_ruim.mp4`: BC 9 · CQL 116 (leaves the track) · IQL 205 | IQL |
| `5_iql_misto.mp4`: BC 500 · CQL 196 · IQL 500 | IQL |

- **Episode seeds:** episode seed 10000 is the same one used in the `5_resultado.png` figures. The BC video instead uses the episode closest to the base model's **median** return (seed 10028), because on seed 10000 both BC models happen to reach 500.
- **Step counts:** they can differ by 1–2 steps from the figures (CQL 116 vs 114, IQL 205 vs 204). The likely cause is the torch/gymnasium versions on the machine that made the figures.

---

## 12. Glossary (PT → EN)

| PT | EN | PT | EN |
|---|---|---|---|
| ação / ações | action(s) | retorno | return |
| esquerda / direita | left / right | perda | loss |
| especialista | expert | passo | (training) step |
| dataset medio / ruim / misto | medium / bad / mixed | parada (antecipada) | early stopping |
| trocar / troca | flip (action noise) | regularização | regularization |
| carregar | load | memória | memory (history stacking) |
| treinar | train | concordância | agreement |
| avaliar | evaluate | viés | bias |
| rede / pesos | network / weights | superestimação | overestimation |
| figuras / gráficos | figures / plots | vantagem | advantage |
| pasta | folder | varredura | sweep |
| so-graficos | only-plots | mapa de política | policy map |
| seeds / semente | random seeds | ângulo do poste | pole angle |
