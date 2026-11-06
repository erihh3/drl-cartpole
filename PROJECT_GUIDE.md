# Offline RL on CartPole: project guide

A walkthrough of the code, the folder layout, how to run everything, and what the results mean.
The code and slides are in Portuguese, so Portuguese names are kept and translated where they first appear.

Last updated 2026-11-06, after the follow-up runs recorded in [`Historico/HISTORIA.md`](Historico/HISTORIA.md) (pt-BR). Section 9 has the current results; section 13 explains the record.

---

## 1. What this project is in one paragraph

We train agents for **CartPole-v1** (Gymnasium) **without ever interacting with the environment during training**. That is **offline RL**: the agent only reads a fixed dataset of transitions `(s, a, r, s', d)` recorded earlier by another policy `πβ`. The environment appears only at the end, to measure the return.

We generate four datasets of different quality from a one-line "expert" heuristic and compare three methods:

| Method | Script | Core idea | Uses reward? |
|---|---|---|---|
| **BC** (Behavior Cloning) | `Código/BC/bc_cartpole.py` | Supervised classification: copy the dataset's action | No |
| **CQL** (Conservative Q-Learning) | `Código/CQL/cql_cartpole.py` | Q-learning plus a penalty that pushes down Q of actions **not** in the data | Yes |
| **IQL** (Implicit Q-Learning) | `Código/IQL/iql_cartpole.py` | Never queries Q of unseen actions; uses an expectile to get a "near-max" over dataset actions, then advantage-weighted BC | Yes |

**Takeaway** (five seeds unless stated)
- With good data (`expert`), all three methods reach 500/500 in every seed.
- At 30% action noise, BC reaches 500, far above the data's 113. At 60%, it copies the wrong majority and scores 8.9, below the data.
- Naive offline Q-learning (DQN) explodes on `expert` because of extrapolation error. Any CQL α > 0 fixes it.
- IQL is the most stable method. It beats CQL (α = 0.5) on `medio`, `ruim` and `misto` in every seed, stays at 500 on `misto` up to 60,000 steps, and depends little on τ and β.
- CQL can match IQL at 15,000 steps with α = 5, but it needs the right α and the right moment to stop, and it falls with longer training at every α tested. Offline, neither choice can be checked.

---

## 2. Folder structure

```
LocalOffLine-RL_v2/
├── README.md                         # short summary and the five-seed results
├── PROJECT_GUIDE.md                  # this file
├── anottations_final_project.md      # course requirements for the 30-min presentation
│
├── Código/                           # "Code"
│   ├── BC/
│   │   ├── bc_cartpole.py            # Behavior Cloning + 4 variants (early stop, regularization, memory)
│   │   └── cartpole_comum.py         # shared module (identical copy)
│   ├── CQL/
│   │   ├── cql_cartpole.py           # offline DQN (α=0) vs CQL, α sweep
│   │   └── cartpole_comum.py         # identical copy
│   ├── IQL/
│   │   ├── iql_cartpole.py           # IQL vs CQL vs BC on 4 datasets, τ sweep, perturbed test
│   │   └── cartpole_comum.py         # identical copy
│   ├── videos_cartpole.py            # side-by-side videos from the saved models (section 11)
│   └── Extras/                       # follow-up studies and the run record (section 13)
│
├── figuras_bc_cartpole/              # "figures": the run the slides use (models .pt + PNGs + numeros.json)
├── figuras_cql_cartpole/
├── figuras_iql_cartpole/
│
├── Output/                           # 2026-11-01 re-run (section 9.3)
│   ├── figuras_{bc,cql}_cartpole/
│   └── log_{bc,cql,iql}.txt
│
├── Historico/                        # "history": record of every run, with CSVs and figures (section 13)
├── experimentos/                     # "experiments": raw runs and weights; gitignored, local only
├── videos_cartpole/                  # MP4/GIF episodes for the slides
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
- **`cartpole_comum.py` is copied three times, and the copies are byte-identical.** Each script does `import cartpole_comum as C`. Python finds the module because it sits next to the script. If you change one copy, change all three, then check `md5sum Código/*/cartpole_comum.py`.
- **Datasets do not live in `Dataset/`.** Minari stores them in its own cache: `~/.minari/datasets/cartpole/{expert,medio,ruim}-v0/` (HDF5). The `misto` dataset is never saved; it is built in memory.
- **`figuras_*` is the run the slides use, not the latest.** It is seed 0 of the code as it was in commit `6759143` (2026-11-01): before the measurement fixes and the regularized BC, with IQL evaluated on 30 episodes. `figuras_bc_cartpole/` and `figuras_cql_cartpole/` are byte-identical to `Output/`. The current numbers are the five-seed runs in section 9.1.
- `numeros.json` ("numbers") in each figure folder holds the key metrics behind the figures. Use it, or `Historico/dados/`, as the source of truth for numbers on slides.
- `modelos*.pt` ("models") holds the trained weights and training logs. `--so-graficos` ("only plots") regenerates the figures from it without retraining.
- `experimentos/` holds every follow-up run (`iql_s0`…`iql_s4`, `cql_s0`…`cql_s4`, `iql_60k`, `sensibilidade_gpu`, …). It is gitignored, so its weights exist only on the machine that ran them. The numbers they produced are copied into `Historico/`, which is tracked.

---

## 3. How to run

### 3.1 Setup (once)

```bash
python -m venv .venv && source .venv/bin/activate        # or use uv / conda
pip install torch gymnasium matplotlib "minari[create,hdf5]"
```

> ⚠️ The scripts' docstrings say `pip install "minari[create]"`. With current Minari (0.5.x) that is **not enough**: dataset creation fails with `ImportError: h5py is not installed`. You also need the `hdf5` extra, as shown above.

No GPU is needed. Training runs on CPU by default, and each process uses one thread (`torch.set_num_threads(1)`). Parallelism comes from `ProcessPoolExecutor` (`--workers`), one process per model. Set `--workers` to about the number of physical cores.

### 3.2 Run (from the project root)

`--pasta` (output folder) is **relative to the current directory**, so run from the project root.

```bash
python Código/BC/bc_cartpole.py      # ~ BC: 4 variants × 3 seeds on "medio 10%"
python Código/CQL/cql_cartpole.py    # ~ CQL: α ∈ {0, 0.1, 0.5, 2, 5} on "expert"
python Código/IQL/iql_cartpole.py    # ~ IQL vs CQL vs BC on 4 datasets + τ sweep on "ruim"
```

> ⚠️ **The default `--pasta` is the tracked `figuras_<method>_cartpole/`.** Running a script without `--pasta` overwrites the results the slides use. For new runs, write to `experimentos/` and keep the inner folder name, so that `videos_cartpole.py --raiz experimentos/<run>` still finds the models:
>
> ```bash
> P=experimentos/meu_run
> python Código/IQL/iql_cartpole.py --pasta $P/figuras_iql_cartpole --workers 8
> ```

The first run of any script generates the Minari datasets (~30 s). After that they are reused.

Useful flags:

| Flag | Script | Meaning |
|---|---|---|
| `--so-graficos` | all | Skip training and re-plot from the saved `.pt` (rewrites the PNGs and `numeros.json` in `--pasta`) |
| `--pasta DIR` | all | Output folder (default `figuras_<method>_cartpole`) |
| `--workers N` | all | Parallel processes (default 4; 5 for CQL) |
| `--mostrar` | all | Open the figures in a window as well |
| `--passos N` | all | Training steps (BC max 20k; CQL/IQL 15k) |
| `--episodios N` | all | Final evaluation episodes (default 50) |
| `--device cpu\|cuda` | all | Where to train (default `cpu`). Evaluation always runs on CPU |
| `--estudo dados\|todos` | BC | Also run the data-quality/quantity study (figure 6) |
| `--seeds 0 1 2` / `--seed 0` | BC / CQL, IQL | Random seeds |
| `--dataset {expert,medio,ruim,misto}` | BC, CQL | Which dataset |
| `--alfas …` | CQL | α values to sweep |
| `--datasets …`, `--analise ruim`, `--tau`, `--beta`, `--alpha`, `--taus-varredura` | IQL | Datasets, analysis dataset, IQL/CQL hyperparameters |

Quick versions: add `--passos 2000` for a smoke test, or `--datasets ruim misto` for a partial IQL run. Always with a scratch `--pasta`.

**Same seed, shorter run = exact prefix.** With the same seed and device, a run with `--passos N` recreates exactly the model a longer run had at step N. We checked this on seed 0 (15k against 60k logs: Q, V, agreement and return all match) and on GPU. That is how the stopping points in section 9.2 were re-scored on 50 episodes.

**Runtime.** With `--workers 8` on an 8-core desktop CPU, the default jobs take roughly 2–3 min (BC), 1 min (CQL) and 5–6 min (IQL). Measured per model on that machine:

| Script | Models | Time per model |
|---|---|---|
| BC | 12 | 20–70 s (`--estudo dados` run) |
| CQL | 5 | ~40 s |
| IQL | 14 | IQL 135–166 s, CQL 47–62 s, BC 50–64 s |

Expect 2–3× longer on a laptop or when several scripts share the CPU (section 9.3 has the slower 2026-11-01 timings). Wall time is never shorter than the slowest single model. The five-seed CPU run took about 40 minutes with 8 processes.

**GPU.** `--device cuda` trains on the GPU, but it did not help. On the same machine, the 14 IQL-script models with 2,000 steps took 38.5 s on CPU and 47.5 s on GPU, both with 8 processes. With 4 processes, CPU and GPU tied (~55 s). The networks are tiny (~67k parameters, batch 256), so the fixed cost of launching each GPU operation most likely dominates (not measured). GPU weights are not bit-identical to CPU weights, so keep every seed of a comparison on the same device. If you add a GPU path:
- On CPU, `--device` changes nothing: results are bit-identical with and without it.
- With `cuda`, the pool uses `spawn` (`C.contexto_pool`), and evaluation runs on a CPU copy of the network (`C.rede_na_cpu`).
- The GTX 1050 (Pascal, sm_61) needs the CUDA 12.6 legacy PyTorch wheel; PyTorch's default CUDA wheels dropped Pascal starting with 2.8.

**You don't need to retrain to present.** `--so-graficos` re-plots the existing `figuras_*` results in seconds. For `figuras_iql_cartpole/`, it takes longer: the old `.pt` has no perturbed returns, so the script evaluates the saved networks again, and `5_resultado.png` gains the perturbed values.

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

Three facts drive most of the results:
- At **30%** flips, the correct action is still the **majority** in each state. BC's argmax therefore recovers the expert.
- At **60%** flips, the **wrong** action is the majority. BC learns to do the wrong thing. Only methods that look at the reward (CQL, IQL) can do better than the data.
- In `expert`, each state has only **one** action, because the expert is deterministic. The other action's Q is never trained by any data point. That is the classic **extrapolation error** setup for offline Q-learning.

`misto` has only 2.4% expert episodes, but they make up 43.8% of its transitions, because expert episodes are long. Section 9.2 shows how much of that IQL actually needs.

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
| `avaliar(agente, n, seed, passos_aleatorios=0)` ("evaluate") | Runs `n` online episodes (seeds `seed + k`). Returns mean return, all returns, action counts. With `passos_aleatorios = k`, the first k actions of each episode are random, then the agent takes over (the **perturbed test**). The random actions depend only on the episode seed, so every agent sees the same states; those steps count in the return |
| `rodar_episodio` ("run episode") | One episode; returns the visited states (used for the θ-over-time plots) |
| `mapa_politica` / `desenhar_mapa` | **Policy map**: the action chosen on a θ × ω grid (cart at rest at the centre), plotted with the expert boundary `θ + 0.5ω = 0` and dataset states |
| `conferir_dispositivo` / `contexto_pool` / `rede_na_cpu` | `--device` support: fail early if CUDA is missing; `spawn` pool for GPU; CPU copy of a network for evaluation |
| `estilo_graficos` / `utf8_no_console` | Matplotlib style; UTF-8 fix for the Windows console |

`carregar()` returns a dict of per-transition arrays:

| Key | Meaning |
|---|---|
| `s, a, r, s2, d` | state, action, reward, next state, terminal flag (terminations only) |
| `G` | discounted return-to-go from this step (γ = 0.99). Used only to check whether Q is calibrated; it never enters training. For an episode truncated at 500 steps, the recursion starts from 1/(1−γ) = 100 instead of 0, the value the TD targets converge to, because the pole is still up |
| `ep` | episode index |
| `ini` | index of the first transition of this transition's episode (used for history stacking) |
| `ret_ep` | total return of the episode this transition belongs to |
| `sucesso` | episode reached 500 steps (truncated) |

The `G` bootstrap was added on 2026-11-02. It changes only the measured bias, never the training: the CQL α = 0.1 bias on `expert` went from −26.7 to −47.1, the DQN bias barely moved (+25,667 → +25,647), and nothing on `ruim` changed, because no `ruim` episode is truncated.

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

**The seed also changes the data.** `selecionar()` uses the seed to choose the validation episodes and the 10% training subset, so BC seeds vary the data as well as the initialization.

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
4. Results, including weights, are saved to `modelos_melhorias.pt` (`modelos_dados.pt` / `modelos_todos.pt` for the other `--estudo`). `gerar_figuras` writes the PNGs and `numeros.json`.

**Figures**

| File | Shows |
|---|---|
| `1_parada.png` | Training vs validation loss of `base` (validation loss rises from step 500 = memorizing noise) and the return dropping during training |
| `2_regularizacao.png` | Policy maps: `base` grows "islands" off the expert boundary (memorized noise). Validation loss per variant |
| `3_memoria.png` | Accuracy against the dataset vs against the expert: memory brings nothing new |
| `4_comparacao.png` | Final return per variant (bars = mean ± std, dots = seeds) and the step of the saved network |
| `5_resultado.png` | Pole angle over 3 episodes: `base` vs the best variant |
| `6_dados.png` | *(only with `--estudo dados/todos`)* BC return vs data quality and vs number of expert episodes. The 2026-11-03 run is in `experimentos/bc_dados/` and `Historico/figuras/6_dados_bc.png` |

---

## 6. Conservative Q-Learning: `Código/CQL/cql_cartpole.py`

**Problem it solves.** Offline Q-learning uses the target `y = r + γ max_a' Q(s', a')`. The `max` can select an action that **never appears in the data**. Its Q is pure extrapolation, it is never corrected, and the error is bootstrapped through the whole value function. On `expert`, where each state has one action, offline DQN's Q-values **explode** far above the real return-to-go, which is at most 100 (= 1/(1−γ)). How far depends on the seed: the final Q on the data ranges from 295 to 443,144 over five seeds.

**CQL loss (discrete, CQL(H))**, exactly as in the code:
```python
y      = r + γ (1 - d) Q_target(s', argmax_a' Q(s', a'))          # Double DQN target
perda  = MSE(Q(s,a), y) + α * ( logsumexp_a Q(s,·) − Q(s, a_data) ).mean()
```
- `logsumexp` is a soft max over actions. Minimizing `logsumexp − Q(s, a_data)` pushes **down** all actions, softmax-weighted, and pushes **up** the dataset action.
- With 2 actions, the penalty equals `softplus(−Δ)`, where `Δ = Q(s, a_data) − Q(s, other)`. Its gradient is BC's cross-entropy gradient applied to the Q-values. So a very large α turns CQL into a classifier of the dataset's actions, which is BC (section 9.2).
- `α = 0` gives plain **offline Double DQN**, the baseline that fails.

**Training details.** Batch 256, Adam 3e-4, 15k steps, Polyak target update `θ_t ← 0.995 θ_t + 0.005 θ`, γ = 0.99, seed 0 by default (`--seed`).

**Default experiment.** Dataset `expert`, α ∈ {0, 0.1, 0.5, 2, 5}. For the figures, "CQL" means the best α > 0 (ties → the smallest α, currently 0.1).

**Logged during training.** `q_dados` (Q of the dataset action), `q_outra` (Q of the other action), `gap` (|Q_left − Q_right|), `concordancia` (agreement of argmax Q with the dataset action), TD loss, penalty, and the return every 1.5k steps.

**Figures**

| File | Shows |
|---|---|
| `1_superestimacao.png` | DQN's Q grows far past the real return (log scale); CQL stays below it. Predicted max Q vs real G |
| `2_mecanismo.png` | One concrete state: Q bars, logsumexp line, penalty value; TD loss vs penalty over training |
| `3_gap.png` | CQL separates the two actions; agreement with the dataset goes to ~100% |
| `4_alfa.png` | α sweep: return, final Q values, return curves |
| `5_resultado.png` | Pole angle over one episode, plus the policy maps of DQN vs CQL |

> ⚠️ **Don't read figure 1 as "CQL is a lower bound".** On `expert`, CQL's Q on the dataset action (≈ 53 at 15k steps) is explained by training time alone. With reward 1, no terminations and Polyak 0.005, `Q(t) = 1 + 0.99 × 100 × (1 − e^(−t/20,000))`, a formula with no penalty term. It matches CQL at every α and IQL too, and all of them reach ~95 at 60,000 steps. What α really changes is the Q of the action not in the data: 7.0, −28.2, −63.6 and −72.3 at 15k steps for α = 0.1, 0.5, 2 and 5. On `ruim`, CQL with α = 0.5 actually overestimates: bias +34 at 15k steps and +201 at 60k.

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

**Default experiment.** For each of the 4 datasets: IQL (τ = 0.7, β = 1), CQL (α = 0.5), and BC. Plus a τ sweep {0.5, 0.7, 0.9} on the analysis dataset `ruim`. That makes 14 models, seed 0 by default, all with 15k steps.
- **BC** is regularized like `bc_cartpole.py`'s `+ regularização` (dropout 0.2, AdamW weight decay 1e-3, label smoothing 0.1, `BC_HP`), but with **no** early stopping, so it is still not the best BC variant.
- **Evaluation:** 50 episodes (seeds 10000–10049, the same as the other scripts), then the same 50 with the first 5 actions random (`PASSOS_ALEATORIOS = 5`).
- **Old `.pt` files still load.** `Agente.montar` detects whether the BC network has dropout layers, and `retorno_perturbado()` evaluates the saved network when the `.pt` has no perturbed return.

Until 2026-11-02 the BC here was plain (Adam, no regularization) and IQL used 30 evaluation episodes. `figuras_iql_cartpole/` still comes from that version.

**Figures** (analysis dataset `ruim`)

| File | Shows |
|---|---|
| `1_acoes_sem_dados.png` | Q over training for CQL vs IQL; calibration of Q(s, a_data) against real G |
| `2_mecanismo_iql.png` | One state where the data took the wrong action: Q, V, advantage; BC vs IQL action probabilities; histogram of AWR weights for right vs wrong dataset actions |
| `3_politicas.png` | Agreement with the dataset over training; policy maps for BC / CQL / IQL |
| `4_tau.png` | τ sweep: return, V and Q vs τ, return curves vs CQL and BC |
| `5_resultado.png` | **The main results chart**: return per dataset for BC / CQL / IQL, with the perturbed return in parentheses (white diamonds), plus one episode on `ruim` |

`numeros.json` also stores `hp_BC`, `episodios`, `passos_aleatorios` and `retorno_perturbado_<model>`.

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
       avaliar(): online evaluation in CartPole-v1 (the ONLY env contact; always on CPU)
                                    │
       torch.save(results) → modelos*.pt   →   gerar_figuras → PNGs + numeros.json
                                    │
       Código/Extras/coletar_historico.py → Historico/dados/*.csv → figures, HTML, PDFs
```

Shared conventions across the three scripts:
- Same MLP (4 → 256 → 256 → out), batch 256, γ = 0.99.
- `log_cada` / `aval_cada` intervals: metrics every 1/40 of training, return every 1/10 (10 episodes, seed 50 000).
- Final evaluation seeds start at 10 000, so they differ from the dataset seeds (100 000+) and the training-curve seeds (50 000). Final evaluation uses 50 episodes in all three scripts.
- Fixed RNG seeds: monitor sample 123, figure samples 7 and 0, dataset flips 0.
- Each `treinar` returns a dict with `log`, `retorno`, `retornos`, `acoes`, `pesos` (plus `dispositivo`, and in IQL `retorno_perturbado`, `retornos_perturbados`, `passos_aleatorios`). That dict is what goes into the `.pt` file. New keys get added; old ones are never renamed or removed, so old `.pt` files keep loading.

---

## 9. Results

### 9.1 Five seeds (current)

CQL and IQL with seeds 0–4, CPU, 15,000 steps, 50 evaluation episodes per model. The seed changes the initialization and the minibatches; the data and the evaluation episodes stay fixed. Source: `Historico/dados/resumo_seeds.csv` (from `experimentos/iql_s*` and `experimentos/cql_s*`), HISTORIA chapter 7.

**Return, mean ± std over 5 seeds**

| Dataset | Data | BC (regularized) | CQL (α = 0.5) | IQL (τ = 0.7) |
|---|---|---|---|---|
| `expert` | 500.0 | 500 ± 0 | 500 ± 0 | 500 ± 0 |
| `medio` | 113.0 | 500 ± 0 | 397.1 ± 62.4 | 500 ± 0 |
| `ruim` | 16.0 | 8.9 ± 0.0 | 120.8 ± 22.4 | 221.2 ± 15.1 |
| `misto` | 27.8 | 93.6 ± 14.8 | 260.2 ± 133.4 | 500 ± 0 |

What holds in every seed:
- IQL beats CQL (α = 0.5) on `medio`, `ruim` and `misto`.
- BC scores 500 on `medio` and stays below the data on `ruim`.
- On `expert`, offline DQN (α = 0) scores 9.26–9.50 and uses a single action in every evaluation episode (always left: 9.26, 3 seeds; always right: 9.50, 2 seeds). Every α > 0 scores 500.

What the seed changes:
- **The size of DQN's Q.** Final Q on the data: 295 to 443,144; mean bias +231 to +533,830. "Q of order 10⁴" was one seed. CQL with α = 0.1 ends at Q 52.8–52.9 in every seed (bias −47.11 ± 0.05).
- **CQL on `misto`:** 123 to 466 (seed 0 gave 187). **CQL on `medio`:** 306 to 478.

Seed 0 ran on two different machines with the same code: the 14 models came out identical, weights included. Machines alone do not explain the run-to-run differences in section 9.3; the old runs did not record library versions.

**Perturbed test (first 5 actions random), mean over 5 seeds**

| Dataset | BC | CQL (α = 0.5) | IQL (τ = 0.7) |
|---|---|---|---|
| `expert` | 500 → 485.7 | 500 → 484.7 | 500 → 485.9 |
| `medio` | 500 → 486.3 | 397.1 → 397.8 | 500 → 490.2 |
| `ruim` | 8.9 → 12.3 | 120.8 → 126.4 | 221.2 → 194.3 |
| `misto` | 93.6 → 30.5 | 260.2 → 259.8 | 500 → 477.3 |

The heuristic that generated the data scores 485.4 on the perturbed `expert` test, so the ~15-point drop there is the cost of the task, not a lack of robustness. The clear drops are BC on `misto` (the largest), IQL on `ruim` and IQL on `misto`. BC's gain on `ruim` is an artifact: the random steps count in the return. Overall, this test separates the methods very little.

### 9.2 Follow-up studies

Five seeds unless stated. "GPU" runs used the same scripts with `--device cuda`; with the same settings their means stay within 12 points of the CPU runs, so they are kept out of the main table.

| Question | Answer | HISTORIA |
|---|---|---|
| Are CQL's mid-training peaks real? | Yes. Re-scored on 50 episodes (GPU): on `ruim`, CQL α = 0.5 gives 237 / 241 / 255 at 3k / 6k / 9k steps, then 121 at 15k. On `misto`, 500 at 3k–6k, then 248. IQL climbs steadily to 220 (`ruim`) and stays at 500 (`misto`) | ch. 6, 7, 10 |
| Does any training signal mark CQL's drop? | No. On `misto`, Q and the TD loss rise smoothly through it, and agreement with the data stays at 57–59% | ch. 7 |
| Does α explain CQL's loss to IQL? | Partly, at 15k steps (GPU): α = 5 gives 500 ± 0 on `medio` and `misto` and 192 ± 20 on `ruim`, against IQL's 220 ± 12 | ch. 10 |
| Does a high α cure the drop? | No, it delays it. On `misto`, the mean stays at 500 until about 7,500 steps (α = 0.5), 12,000 (α = 2) and 30,000 (α = 5). At 60k (GPU), α = 5 gives 266 ± 164 on `misto` and 90 ± 15 on `ruim`, against IQL's 500 ± 0 and 244 ± 124 | ch. 11 |
| What happens above α = 5? | CQL turns into BC. At α = 50 on `ruim` it scores 8.9 and agrees with the expert in 0.9% of states (BC: 8.9, 0.6%) | ch. 11 |
| Can α be chosen from the data (lowest Q bias)? | No. That rule picks α = 20 on `ruim` (17) and α = 0.5 on `misto` (248); it is right only on `medio`, where BC already scores 500 | ch. 10, 11 |
| Does IQL depend on its settings? | Little. On `misto`, 500 in every seed for τ ∈ {0.5, 0.7, 0.9} and β ∈ {0.3, 1, 3}. On `ruim`, τ = 0.5 / 0.7 / 0.9 give 209 / 221 / 191 (CPU) and β = 0.3 / 1 / 3 give 153 / 220 / 227 (GPU) | ch. 7, 8, 10 |
| Does IQL hold with long training? | Yes, on average. At 60k, `misto` stays at 500 in every seed and point; `ruim` averages 244, pulled up by one seed (459), with a median of 183. Its worst seed (159) still beats CQL's best at any α (133). Q stays at 17 on `ruim`, bias +7.7 | ch. 11 |
| Does offline DQN fail on noisy data too? | Not at 15k: Q 49–70 and returns 412 / 117 / 174 on `medio` / `ruim` / `misto`, all above the data. Plausibly the noise puts both actions in similar states. Later it explodes too: Q 363 (`ruim`) and 705 (`misto`) at 60k, return ~100 | ch. 10, 11 |
| How much expert data is enough? | All of `ruim` plus k expert episodes (3 seeds). IQL: 490 with k = 10 (0.5% of episodes), 500 with k = 25. At k = 25, CQL (α = 0.5) 269, BC 61. Around k = 5: IQL 423, CQL 122, BC 13 | ch. 9 |
| Is BC short of good data, or hurt by the mix? | The mix. One expert episode alone gives BC 454. The same kind of episode added to all of `ruim` gives 8.9. `misto` filtered to 500-return episodes gives 500; `ruim` top 10% gives 83 (66–101); full `misto`, 81 (`--estudo dados`, 3 seeds) | ch. 9 |
| τ on `ruim` | No steady fall with τ: 209 / 221 / 191 (five seeds). τ = 0.9 trails τ = 0.7 in 4 of 5 seeds, and at 60k (seed 0) it falls to 101 while V keeps rising (88). "Too much optimism" shows up only at τ = 0.9 with long training | ch. 8 |

**Still open:** the cause of CQL's late drop (a max in the TD target is the coherent explanation, not isolated: IQL has no max and does not fall; τ = 0.9, closer to a max, falls); `medio` at 60k; the 60k τ sweep has one seed; the wind test (chapter 12).

### 9.3 The 2026-11-01 re-run (kept for the record)

Setup: 8-core CPU, no GPU, Python 3.12, torch 2.15 (CPU), Minari 0.5.4, three scripts running at the same time. BC and CQL finished and are in `Output/` (and in `figuras_bc_cartpole/`, `figuras_cql_cartpole/`). IQL was stopped partway: 6 of 14 models done, no figures. This run used the code before the 2026-11-02 changes (old `G`, plain BC in the IQL script, 30 IQL evaluation episodes).

"Original `numeros.json`" below is the first run, whose files are no longer in the repo; its values survive only in this table and in `Historico/dados/comparacao_marcos.csv`.

**BC (`medio` 10%, 3 seeds): return, mean ± std**

| Variant | Slides | Original `numeros.json` | Re-run |
|---|---|---|---|
| base | 385 ± 35 | 356 ± 90 | 399 ± 49 |
| + parada | 500 | 500 ± 0 | 500 ± 0 |
| + regularização | 500 | 500 ± 0 | 500 ± 0 |
| + memória | 500 | 500 ± 0 | 500 ± 0 |

The improved variants reproduce to three decimals (validation loss and accuracy). Only `base` moves, because a network that has memorized noise is fragile. The default BC study was not re-run after this (only `--estudo dados` was); the BC script only gained `--device`.

**CQL (`expert`, seed 0)**

| Metric | Slides | Original JSON | Re-run |
|---|---|---|---|
| Return α = 0 (DQN) / α ≥ 0.1 | 9.5 / 500 | 9.3 / 500 | 9.5 / 500 |
| DQN mean bias of max Q | +2,737 | +19,045 | +25,667 |
| CQL mean bias | −27 | −26.7 | −26.7 |
| Agreement with dataset, DQN / CQL | 50% / 99.6% | 36% / 99.8% | 33% / 99.8% |

DQN's divergence is exponential, so its final Q magnitude changes from run to run (and, as section 9.1 shows, from seed to seed). The conclusion does not: it ends far above the maximum possible return of 100. The CQL bias of −26.7 uses the old `G`; with the truncation fix it is −47.1.

**IQL (partial)**: expert: IQL 500, CQL 500, BC 500 · medio: IQL 500, BC 500, CQL 384.5 (430 in the original).

**Runtime** (8 cores, all three scripts in parallel):

| Script | Models | Time per model | Wall time |
|---|---|---|---|
| BC | 12 | base ~150 s, others 45–75 s | ~6 min |
| CQL | 5 | 100–175 s | ~8 min |
| IQL | 14 | IQL 350–450 s, CQL ~170 s, BC 70–150 s | ~20–25 min (estimate) |

---

## 10. Things to know before presenting

1. **Quote the five-seed numbers.** The slides use numbers from runs older than any figure, and `figuras_*` is seed 0 of the older code. Take numbers from `Historico/dados/resumo_seeds.csv` (or the figure you actually show). How the main ones moved:

   | Measure | Slides | Five seeds |
   |---|---|---|
   | IQL on `ruim` | 187 | 221 ± 15 |
   | CQL on `medio` / `ruim` / `misto` | 372 / 126 / 191 | 397 ± 62 / 121 ± 22 / 260 ± 133 |
   | BC on `misto` | 140 | 94 ± 15 |
   | BC base, `medio` 10% (3 seeds) | 385 ± 35 | 399 ± 49 (2026-11-01, not re-run) |
   | BC on `ruim` top 10% (3 seeds) | 72 | 83 (66 to 101) |
   | DQN on `expert`: return / mean Q bias | 9.5 / +2,737 | 9.4 ± 0.1 / +231 to +533,830 |
   | CQL α = 0.1 on `expert`, mean bias | −27 | −47.11 ± 0.05 |
   | DQN agreement with the data | 50% | 33% in one seed, 50% in the others |
   | CQL on `ruim`, agreement with the expert | 73% | 71% ± 7 |
   | IQL on `ruim`, τ = 0.5 / 0.7 / 0.9 | 212 / 187 / 159 | 209 / 221 / 191 |

   For the DQN, say "far above the maximum of 100, from hundreds to hundreds of thousands depending on the seed". The BC "misto 140" matches no saved run: the IQL-script BC gives 94, and `--estudo dados` gives 81.
2. **Claims that no longer hold** (HISTORIA, "O que caiu"):
   - "The return falls as τ grows": there is no steady fall (see 9.2).
   - "CQL stays below the real return, the promised lower bound": on `expert` that is training time (section 6), and on `ruim` CQL overestimates.
   - "IQL beats CQL", unconditionally: at 15k steps, CQL with α = 5 ties on `medio` and `misto`. At 60k, IQL is clearly ahead again.
   - "Only IQL gets back to 500 on `misto`": CQL with α = 0.5 was at 500 in all five seeds from 3,000 to 7,500 steps, then fell.
   - "CQL on `misto`: 189": one seed. The five seeds range from 123 to 466.

   The statement that survives everything: **IQL keeps what it learns; CQL can reach it, but needs the right α and the right moment to stop, and offline data can check neither.**
3. **Slide 6 of `Offline Reinforcement Learning.pptx`** talks about **FourRooms** (MiniGrid). That's left over from an earlier version. Replace it with the CartPole DQN numbers.
4. **The `Sobre o dataset` deck's footer says "x / 19"**, but the deck has 7 slides.
5. **Seeds.** CQL and IQL now have five seeds, and BC three. Single-seed results remain: the 60k τ sweep on `ruim` and the 60k runs on `expert`. The sensitivity and 60k α runs were trained on GPU.
6. **BC inside `iql_cartpole.py`** is regularized but has no early stopping, so it is not the best BC variant from `bc_cartpole.py`. Say so when comparing.
7. **Course notes mention PettingZoo/BenchMARL.** The project is single-agent: Gymnasium + Minari. Be ready to say why, and link offline multi-agent RL as future work.

---

## 11. Videos for the slides (`Código/videos_cartpole.py`)

The script replays the **already-trained** models from the `figuras_*/*.pt` files (or from `--raiz DIR`). It does no training and takes about 1–2 min on CPU. Agents run side by side from the same initial state, and each panel freezes when its episode ends, labelled "caiu" (fell) or "saiu do trilho" (left the track).

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

- **Which models:** the videos use the seed-0 models in `figuras_*`, so the IQL-deck videos show the plain BC of the older code. Each video is a single episode, not an average.
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
| perturbado / passos aleatórios | perturbed / random (initial) steps | sensibilidade | sensitivity |
| rodada | run | queda / cair | drop / fall |
| histórico / marco | record / milestone | dispositivo | device |
| experimentos | experiments | vento / rajadas | wind / gusts |
| expectil | expectile | pendente / concluído | pending / done |

---

## 13. The run record (`Historico/`) and follow-up studies (`Código/Extras/`)

**Rule: older results are never deleted.** New experiments go in their own folders, without editing the main scripts, and `Historico/` keeps the record of how every number moved. Results from commit `6759143` are read directly from git, so promoting a run into `figuras_*` later does not change the record. If a snapshot file changes, the new version is added next to it, with its hash in the name.

**`Historico/HISTORIA.md`** (pt-BR; also `historia.html`, `resumo_1pagina.pdf` and `resumo_3paginas.pdf`) tells the story in chapters:

| Ch. | Topic |
|---|---|
| 1 | The numbers moved on their own between runs |
| 2 | Fixing the measurement: `G` for truncated episodes |
| 3 | A fair BC: regularization in the IQL script |
| 4 | More evaluation episodes (30 → 50): the measurement noise is ~10 points |
| 5 | Off the data distribution: the perturbed test |
| 6 | What we were reading wrong: Q ≈ 53, CQL's early 500 on `misto`, the AWR weights |
| (detour) | Does the GPU help? No |
| 7 | Five seeds |
| 8 | 60,000 steps; Q against the formula; τ |
| 9 | How much expert data is enough; the BC data study |
| 10 | Sensitivity (GPU): stopping points, α, β, τ |
| 11 | High α and 60,000 steps (GPU) |
| 12 | Wind (pending: only the heuristic's reference values exist) |

**Folders**

| Path | Contents |
|---|---|
| `Historico/dados/` | Every measured value as CSV, per seed: `resumo_seeds.csv`, `curvas.csv` (training logs), `episodios.csv` (each episode's return), `ic95.csv`, `varredura*.csv`, `sensibilidade*.csv`, `comparacao_marcos.csv`, `evolucao.csv`, `marcos.csv`, `manifesto.csv` (file, size and md5 of each source) |
| `Historico/figuras/` | The record's figures, drawn only from those CSVs |
| `Historico/snapshots/` | `numeros.json`, PNGs and logs of each milestone |
| `Historico/analises/` | One-off analysis scripts and their output (chapters 6–8, the GPU speed test) |

**Scripts in `Código/Extras/`** (they import the main scripts and never edit them)

| Script | What it does |
|---|---|
| `varredura_especialista.py` | All of `ruim` plus k expert episodes (k = 0, 1, 2, 5, 10, 25, 50) for BC, CQL and IQL. Trains only what is missing in `--pasta`; `--plano` lists the jobs |
| `teste_vento.py` | Evaluates the saved policies with constant wind or gusts, without retraining. With zero wind it reproduces the saved returns exactly |
| `coletar_historico.py` | Gathers every saved result into `Historico/dados/` and `snapshots/`. It only reads results: no training, no datasets |
| `figuras_historico.py` | Redraws `Historico/figuras/` from the CSVs |
| `historia_html.py` | Rebuilds `Historico/historia.html` from `HISTORIA.md` |
| `resumo_pdf.py` | Rebuilds the 1-page and 3-page PDF summaries from the CSVs |
| `comum_extras.py` | Shared helpers for the scripts above |

After a new run (the order matters):

```bash
python Código/Extras/coletar_historico.py
python Código/Extras/figuras_historico.py
python Código/Extras/historia_html.py
python Código/Extras/resumo_pdf.py
```

> ⚠️ `coletar_historico.py` reads the raw runs in `experimentos/`, which only exist on the machine that ran them. On a fresh clone it would mark those milestones as pending. There, use the CSVs as they are and run only the last three scripts.
